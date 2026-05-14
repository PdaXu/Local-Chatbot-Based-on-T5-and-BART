from flask import Flask, render_template, request, jsonify
import requests
import psutil
from pynvml import *
import logging
import time
import socket

# 指定自定义的模板目录
app = Flask(__name__, template_folder=r'D:\study\MGW\ChatAPP\frontend\templates')

# 配置日志
logging.basicConfig(filename='model_inference.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 初始化GPU库
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # 假设只有一个GPU

# 初始化网络流量数据
prev_network_io = psutil.net_io_counters()

@app.route('/')
def index():
    # 渲染指定的 HTML 文件
    return render_template('index1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('text')
    model_name = request.json.get('model')

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

    return jsonify({'reply': reply})

@app.route('/monitor', methods=['GET'])
def monitor():
    cpu_percent = psutil.cpu_percent(interval=None)
    gpu_info = nvmlDeviceGetUtilizationRates(handle)
    gpu_percent = gpu_info.gpu

    # 获取网络流量和内存占用率
    current_network_io = psutil.net_io_counters()
    network_send = current_network_io.bytes_sent - prev_network_io.bytes_sent
    network_recv = current_network_io.bytes_recv - prev_network_io.bytes_recv
    prev_network_io = current_network_io

    memory_percent = psutil.virtual_memory().percent

    return jsonify({
        'cpu_percent': cpu_percent,
        'gpu_percent': gpu_percent,
        'network_send': network_send,
        'network_recv': network_recv,
        'memory_percent': memory_percent
    })

def get_ip_address():
    """获取本地IP地址"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == '__main__':
    # 获取本地IP地址
    host_ip = get_ip_address()
    port = 5000
    
    # 打印启动信息
    print("\n" + "="*60)
    print(f"智能对话助手已成功启动！")
    print("="*60)
    print(f"访问地址:")
    print(f"  • 本地: http://localhost:{port}")
    print(f"  • 网络: http://{host_ip}:{port} (同一局域网内可用)")
    print("="*60 + "\n")
    
    # 启动应用
    app.run(debug=True, host='0.0.0.0', port=port)