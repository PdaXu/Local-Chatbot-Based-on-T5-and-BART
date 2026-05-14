import requests
import speech_recognition as sr
import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
import pyttsx3
import psutil
from pynvml import *  # 用于获取GPU信息
import logging
import time
import gradio as gr

# 配置日志
logging.basicConfig(filename='model_inference.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 初始化语音引擎
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# 初始化Vosk模型和队列
vosk_model = Model(r"D:\study\MGW\ChatAPP\frontend\vosk")
q = queue.Queue()

# 初始化GPU库
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # 假设只有一个GPU

# 初始化CPU和GPU使用率数据
cpu_data = []
gpu_data = []
max_data_points = 100  # 最多显示100个数据点

# 初始化网络流量数据
prev_network_io = psutil.net_io_counters()

chat_history_data = []

def query_model(user_input, model_name):
    # 记录开始时间
    start_time = time.time()

    # 根据模型选择不同端口
    if model_name == 'T5':
        url = "http://127.0.0.1:5000/chat"
    else:
        url = "http://127.0.0.1:5001/chat"

    try:
        response = requests.post(url, json={"text": user_input}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            reply = data.get('reply', '（后端未返回回复）')
        else:
            reply = f"错误：HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        reply = f"连接错误：{e}"

    # 记录结束时间并计算推理时间
    end_time = time.time()
    inference_time = end_time - start_time

    # 记录推理时间到日志
    logging.info(f"模型: {model_name}, 推理时间: {inference_time:.4f} 秒")

    return reply

def handle_send(user_input, model_name, history):
    if not user_input:
        return history
    history = history + [[user_input, None]]
    reply = query_model(user_input, model_name)
    history[-1][1] = f"{model_name}: {reply}"

    chat_history_data.append({
        "role": "用户",
        "model": model_name,
        "text": user_input
    })
    chat_history_data.append({
        "role": model_name,
        "model": model_name,
        "text": reply
    })

    return history

def start_voice_capture():
    q.queue.clear()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))

    stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback)
    rec = KaldiRecognizer(vosk_model, 16000)
    stream.start()
    return stream, rec

def stop_voice_capture(stream, rec):
    stream.stop()
    stream.close()
    result = ""
    while not q.empty():
        data = q.get()
        if rec.AcceptWaveform(data):
            result_json = json.loads(rec.Result())
            result += result_json.get("text", "")
    if not result:
        result_json = json.loads(rec.FinalResult())
        result = result_json.get("text", "")
    return result

def voice_input(model_name, history):
    stream, rec = start_voice_capture()
    result = stop_voice_capture(stream, rec)
    if result:
        history = handle_send(result, model_name, history)
    else:
        history = history + [["（语音识别失败）", None]]
    return history

def show_voice_dialog():
    voice_names = [voice.name for voice in voices]
    def on_voice_select(voice_index):
        voice_id = voices[voice_index].id
        engine.setProperty('voice', voice_id)
    with gr.Blocks() as dialog:
        gr.Markdown("选择语音")
        voice_dropdown = gr.Dropdown(choices=voice_names, label="语音")
        select_button = gr.Button("选择")
        select_button.click(on_voice_select, inputs=voice_dropdown)
    dialog.launch()

def show_history():
    history_text = ""
    for record in chat_history_data:
        role = record["role"]
        model = record["model"]
        text = record["text"]
        history_text += f"{role}（{model}）：{text}\n"
    return history_text

def export_to_txt():
    try:
        with open("chat_history.txt", "w", encoding="utf-8") as f:
            for record in chat_history_data:
                role = record["role"]
                model = record["model"]
                text = record["text"]
                f.write(f"{role}（{model}）：{text}\n")
        return "聊天记录已保存到 chat_history.txt"
    except Exception as e:
        return f"导出失败：{str(e)}"

def show_monitor_info():
    cpu_percent = psutil.cpu_percent(interval=None)
    gpu_info = nvmlDeviceGetUtilizationRates(handle)
    gpu_percent = gpu_info.gpu

    # 更新CPU和GPU使用率数据
    cpu_data.append(cpu_percent)
    gpu_data.append(gpu_percent)

    # 保持数据长度不超过最大数据点
    if len(cpu_data) > max_data_points:
        cpu_data = cpu_data[-max_data_points:]
    if len(gpu_data) > max_data_points:
        gpu_data = gpu_data[-max_data_points:]

    # 获取网络流量和内存占用率
    current_network_io = psutil.net_io_counters()
    network_send = current_network_io.bytes_sent - prev_network_io.bytes_sent
    network_recv = current_network_io.bytes_recv - prev_network_io.bytes_recv
    prev_network_io = current_network_io

    memory_percent = psutil.virtual_memory().percent

    info_text = f"CPU使用率: {cpu_percent}%\nGPU使用率: {gpu_percent}%\n网络发送: {network_send} 字节/秒\n网络接收: {network_recv} 字节/秒\n内存占用率: {memory_percent}%"
    return info_text

def show_logs():
    encodings = ['utf-8', 'gbk']  # 尝试的编码列表
    logs = None
    for encoding in encodings:
        try:
            with open('model_inference.log', 'r', encoding=encoding) as f:
                logs = f.read()
            break  # 如果成功读取，跳出循环
        except UnicodeDecodeError:
            continue  # 尝试下一个编码
        except FileNotFoundError:
            return "未找到模型推理日志文件。"
        except Exception as e:
            return f"读取日志失败：{str(e)}"

    if logs is None:
        return "无法使用支持的编码读取日志文件。"
    return logs

def clear_logs():
    try:
        with open('model_inference.log', 'w', encoding='utf-8') as f:
            f.write("")
        return "日志已清空。"
    except Exception as e:
        return f"清除失败：{str(e)}"

def show_about():
    about_text = "作者：徐潘登\n学号：202121335056\n系统名称：基于深度学习的语音文本交互系统\n版本：1.7v"
    return about_text

with gr.Blocks() as demo:
    gr.Markdown("基于深度学习的语音文本交互系统")

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(choices=['T5', 'BART'], label="模型选择")
            voice_button = gr.Button("声音选择")
            history_button = gr.Button("历史记录")
            export_button = gr.Button("导出聊天记录")
            monitor_button = gr.Button("系统监控")
            log_button = gr.Button("查看日志")
            clear_log_button = gr.Button("清除日志")
            about_button = gr.Button("关于")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot()
            with gr.Row():
                input_box = gr.Textbox(placeholder="输入消息...")
                send_button = gr.Button("发送")
                voice_input_button = gr.Button("语音输入")

    send_button.click(handle_send, inputs=[input_box, model_selector, chatbot], outputs=chatbot)
    voice_input_button.click(voice_input, inputs=[model_selector, chatbot], outputs=chatbot)
    voice_button.click(show_voice_dialog)
    history_button.click(show_history, outputs=gr.Textbox(label="聊天历史记录"))
    export_button.click(export_to_txt, outputs=gr.Textbox(label="导出结果"))
    monitor_button.click(show_monitor_info, outputs=gr.Textbox(label="系统监控信息"))
    log_button.click(show_logs, outputs=gr.Textbox(label="模型推理日志"))
    clear_log_button.click(clear_logs, outputs=gr.Textbox(label="清除日志结果"))
    about_button.click(show_about, outputs=gr.Textbox(label="关于信息"))

demo.launch()