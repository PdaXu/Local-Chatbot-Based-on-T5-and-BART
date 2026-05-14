import requests
import speech_recognition as sr
import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
from chat_db import init_db, save_record, get_all_records
import pyttsx3
import os
import psutil
from pynvml import *  # 用于获取GPU信息
import logging
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QComboBox, QLabel, QHBoxLayout, QScrollArea, QFrame, QSplitter, QDialog, QTextBrowser, QListWidget, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTextCursor, QPixmap, QPainter, QRegion, QBitmap, QImage, QColor, QIcon, QMovie
import pyqtgraph as pg


# 配置日志
logging.basicConfig(filename='model_inference.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


class ChatAppWrapper:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[0].id)
        self.vosk_model = Model(r"D:\study\MGW\ChatAPP\frontend\vosk")
        self.q = queue.Queue()
        init_db()
        self.loading_label = None
        self.loading_movie = None
        self.chat_history_data = []
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)
        self.cpu_data = []
        self.gpu_data = []
        self.max_data_points = 100
        self.prev_network_io = psutil.net_io_counters()

    def create_message_bubble(self, text, is_user=True, model=None):
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bubble.setContentsMargins(15, 10, 15, 10)
        bubble.setMaximumWidth(500)

        if is_user:
            bubble.setStyleSheet("""
            QLabel {
                background: #95EC69;
                border-radius: 10px;
                color: #000000;
                border: 1px solid #7BCF53;
            }
        """)
            bubble.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        else:
            bubble.setStyleSheet("""
            QLabel {
                background: white;
                border-radius: 10px;
                color: #000000;
                border: 1px solid #E4E7ED;
            }
        """)
            bubble.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        if is_user:
            pixmap = QPixmap(r"D:\study\MGW\ChatAPP\icons\img3.png")
        elif model == 'T5':
            pixmap = QPixmap(r"D:\study\MGW\ChatAPP\icons\img1.png")
        else:
            pixmap = QPixmap(r"D:\study\MGW\ChatAPP\icons\img2.png")

        pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

        size = min(pixmap.width(), pixmap.height())
        mask = QBitmap(size, size)
        painter = QPainter(mask)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(0, 0, size, size, Qt.white)
        painter.setBrush(Qt.black)
        painter.drawEllipse(0, 0, size, size)
        painter.end()
        pixmap.setMask(mask)

        avatar = QLabel()
        avatar.setPixmap(pixmap)
        avatar.setFixedSize(size, size)

        if not is_user:
            voice_play_button = QPushButton()
            voice_play_button.setFixedSize(20, 20)
            voice_play_button.setStyleSheet("""
            QPushButton {
                border-radius: 10px;
                background-color: #0099ff;
            }
            QPushButton:hover {
                background-color: #007acc;
            }
            """)
            voice_play_button.clicked.connect(lambda: self.play_text_to_speech(text))
        else:
            voice_play_button = None

        return bubble, avatar, voice_play_button

    def append_chat(self, text, sender="user", model=None):
        is_user = sender == "user"
        bubble, avatar, voice_play_button = self.create_message_bubble(text, is_user, model)

        self.chat_history_data.append({
            "role": "用户" if is_user else (model or "模型"),
            "model": model or "",
            "text": text
        })

        row = QHBoxLayout()

        outer_row = QHBoxLayout()

        if is_user:
            outer_row.addStretch(1)

            user_content_layout = QHBoxLayout()
            user_content_layout.addWidget(bubble)
            if voice_play_button:
                user_content_layout.addWidget(voice_play_button, alignment=Qt.AlignTop)
                user_content_layout.addSpacing(5)
            else:
                user_content_layout.addSpacing(10)

            outer_row.addLayout(user_content_layout)
            outer_row.addWidget(avatar)
        else:
            model_label = QLabel(model or "模型")
            model_label.setStyleSheet("""
                color: #666666;
                font-size: 11px;
                font-weight: bold;
                margin-left: 5px;
                margin-bottom: 2px;
            """)
            model_label.setFixedHeight(16)

            content_layout = QVBoxLayout()
            content_layout.addWidget(model_label, alignment=Qt.AlignLeft)
            content_layout.addWidget(bubble)

            bubble_and_button_layout = QHBoxLayout()
            bubble_and_button_layout.addLayout(content_layout)
            if voice_play_button:
                bubble_and_button_layout.addWidget(voice_play_button, alignment=Qt.AlignTop | Qt.AlignLeft)
                bubble_and_button_layout.addSpacing(5)
            else:
                bubble_and_button_layout.addSpacing(10)
            bubble_and_button_layout.addStretch(1)

            outer_row.addWidget(avatar)
            outer_row.addSpacing(5)
            outer_row.addLayout(bubble_and_button_layout)
            outer_row.addStretch(1)

        # 这里假设 self.chat_layout 已经在某个地方初始化
        # self.chat_layout.insertLayout(self.chat_layout.count() - 1, outer_row)

    def handle_send(self, input_box, chat_layout, model_selector):
        user_input = input_box.text().strip()
        if not user_input:
            return
        self.append_chat(user_input, sender="user")
        model = model_selector.currentText()

        self.show_loading_animation(chat_layout)

        reply = self.query_model(user_input, model)

        self.hide_loading_animation()

        self.append_chat(reply, sender="bot", model=model)
        input_box.clear()

        save_record(model, user_input, reply)

    def query_model(self, user_input, model_name):
        start_time = time.time()

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

        end_time = time.time()
        inference_time = end_time - start_time

        logging.info(f"模型: {model_name}, 推理时间: {inference_time:.4f} 秒")

        return reply

    def toggle_recording(self, voice_input_button, input_box, chat_layout, model_selector):
        if not hasattr(self, 'recording'):
            self.recording = False
        if not self.recording:
            voice_input_button.setText("停止录音")
            self.recording = True
            self.start_voice_capture()
        else:
            voice_input_button.setText("语音输入")
            self.recording = False
            result = self.stop_voice_capture()
            if result:
                input_box.setText(result)
                self.handle_send(input_box, chat_layout, model_selector)
            else:
                input_box.setText("（语音识别失败）")

    def start_voice_capture(self):
        self.q.queue.clear()

        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.q.put(bytes(indata))

        self.stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                        channels=1, callback=callback)
        self.rec = KaldiRecognizer(self.vosk_model, 16000)
        self.stream.start()

    def stop_voice_capture(self):
        self.stream.stop()
        self.stream.close()
        result = ""
        while not self.q.empty():
            data = self.q.get()
            if self.rec.AcceptWaveform(data):
                result_json = json.loads(self.rec.Result())
                result += result_json.get("text", "")
        if not result:
            result_json = json.loads(self.rec.FinalResult())
            result = result_json.get("text", "")
        return result

    def show_history(self):
        records = get_all_records()

        dialog = QDialog()
        dialog.setWindowTitle("聊天历史记录")
        dialog.resize(600, 400)
        dialog.setStyleSheet("background-color: #e6f7ff;")

        layout = QVBoxLayout()
        browser = QTextBrowser()
        browser.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 5px; padding: 10px;")

        for record in records:
            model, user_input, reply, timestamp = record[1], record[2], record[3], record[4]
            browser.append(f"[{timestamp}] [{model}]")
            browser.append(f"你：{user_input}")
            browser.append(f"{model} 回复：{reply}")
            browser.append("-" * 50)

        layout.addWidget(browser)
        dialog.setLayout(layout)
        dialog.exec_()

    def show_about(self):
        dialog = QDialog()
        dialog.setWindowTitle("关于作者")
        dialog.resize(300, 200)
        dialog.setStyleSheet("background-color: #e6f7ff;")

        layout = QVBoxLayout()
        browser = QTextBrowser()
        browser.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 5px; padding: 10px;")
        browser.append("作者：徐潘登")
        browser.append("学号：202121335056")
        browser.append("系统名称：基于深度学习的语音文本交互系统")
        browser.append("版本：1.7v")

        layout.addWidget(browser)
        dialog.setLayout(layout)
        dialog.exec_()

    def show_loading_animation(self, chat_layout):
        self.loading_label = QLabel()
        self.loading_movie = QMovie(r"D:\study\MGW\ChatAPP\icons\loading.gif")
        self.loading_label.setMovie(self.loading_movie)
        self.loading_movie.start()
        self.loading_label.setAlignment(Qt.AlignCenter)
        chat_layout.insertWidget(chat_layout.count() - 1, self.loading_label)

    def hide_loading_animation(self):
        if self.loading_label:
            self.loading_movie.stop()
            self.loading_label.deleteLater()
            self.loading_label = None
            self.loading_movie = None

    def play_text_to_speech(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def show_voice_dialog(self):
        dialog = QDialog()
        dialog.setWindowTitle("选择语音")
        dialog.resize(300, 200)
        dialog.setStyleSheet("background-color: #e6f7ff;")

        layout = QVBoxLayout()
        list_widget = QListWidget()
        for voice in self.voices:
            list_widget.addItem(voice.name)

        def on_item_clicked(item):
            index = list_widget.row(item)
            voice_id = self.voices[index].id
            self.engine.setProperty('voice', voice_id)
            dialog.accept()

        list_widget.itemClicked.connect(on_item_clicked)
        layout.addWidget(list_widget)
        dialog.setLayout(layout)
        dialog.exec_()

    def export_chat_records(self, chat_history_data):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(None, "导出聊天记录", "",
                                                   "文本文件 (*.txt);;图片文件 (*.png)", options=options)
        if file_name:
            if _.startswith("文本文件"):
                if not file_name.endswith('.txt'):
                    file_name += '.txt'
                self.export_to_txt(file_name, chat_history_data)
            elif _.startswith("图片文件"):
                if not file_name.endswith('.png'):
                    file_name += '.png'
                # self.export_to_image(file_name)  # 这里需要获取聊天区域的截图，暂时保留注释

    def export_to_txt(self, filename, chat_history_data):
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                for record in chat_history_data:
                    role = record["role"]
                    model = record["model"]
                    text = record["text"]
                    if role == "用户":
                        f.write(f"用户（{model}）：{text}\n")
                    else:
                        f.write(f"{model}：{text}\n")
            QMessageBox.information(None, "导出成功", f"聊天记录已保存到 {filename}")
        except Exception as e:
            QMessageBox.warning(None, "导出失败", f"发生错误：{str(e)}")

    def show_monitor_dialog(self):
        self.monitor_dialog = QDialog()
        self.monitor_dialog.setWindowTitle("系统性能监控")
        self.monitor_dialog.resize(600, 400)
        self.monitor_dialog.setStyleSheet("background-color: #e6f7ff;")

        layout = QVBoxLayout()

        self.cpu_plot = pg.PlotWidget(title="CPU使用率")
        self.gpu_plot = pg.PlotWidget(title="GPU使用率")
        self.cpu_curve = self.cpu_plot.plot(pen='r')
        self.gpu_curve = self.gpu_plot.plot(pen='b')

        layout.addWidget(self.cpu_plot)
        layout.addWidget(self.gpu_plot)

        self.network_send_label = QLabel()
        self.network_recv_label = QLabel()
        self.memory_usage_label = QLabel()

        layout.addWidget(self.network_send_label)
        layout.addWidget(self.network_recv_label)
        layout.addWidget(self.memory_usage_label)

        self.monitor_dialog.setLayout(layout)
        self.monitor_dialog.show()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_monitor_info)
        self.timer.start(1000)

    def update_monitor_info(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        gpu_info = nvmlDeviceGetUtilizationRates(self.handle)
        gpu_percent = gpu_info.gpu

        self.cpu_data.append(cpu_percent)
        self.gpu_data.append(gpu_percent)

        if len(self.cpu_data) > self.max_data_points:
            self.cpu_data = self.cpu_data[-self.max_data_points:]
        if len(self.gpu_data) > self.max_data_points:
            self.gpu_data = self.gpu_data[-self.max_data_points:]

        self.cpu_curve.setData(self.cpu_data)
        self.gpu_curve.setData(self.gpu_data)

        current_network_io = psutil.net_io_counters()
        network_send = current_network_io.bytes_sent - self.prev_network_io.bytes_sent
        network_recv = current_network_io.bytes_recv - self.prev_network_io.bytes_recv
        self.prev_network_io = current_network_io

        memory_percent = psutil.virtual_memory().percent

        self.network_send_label.setText(f"网络发送: {network_send} 字节/秒")
        self.network_recv_label.setText(f"网络接收: {network_recv} 字节/秒")
        self.memory_usage_label.setText(f"内存占用率: {memory_percent}%")

    def show_logs(self):
        encodings = ['utf-8', 'gbk']
        logs = None
        for encoding in encodings:
            try:
                with open('model_inference.log', 'r', encoding=encoding) as f:
                    logs = f.read()
                break
            except UnicodeDecodeError:
                continue
            except FileNotFoundError:
                QMessageBox.warning(None, "日志文件不存在", "未找到模型推理日志文件。")
                return
            except Exception as e:
                QMessageBox.warning(None, "读取日志失败", f"发生错误：{str(e)}")
                return

        if logs is None:
            QMessageBox.warning(None, "读取日志失败", "无法使用支持的编码读取日志文件。")
            return

        dialog = QDialog()
        dialog.setWindowTitle("模型推理日志")
        dialog.resize(600, 400)
        dialog.setStyleSheet("background-color: #e6f7ff;")

        layout = QVBoxLayout()
        log_text = QTextEdit()
        log_text.setReadOnly(True)
        log_text.setPlainText(logs)
        log_text.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 5px; padding: 10px; font-family: monospace;")

        clear_button = QPushButton("清除日志")
        clear_button.setStyleSheet("""
        QPushButton {
            padding: 8px 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: white;
            margin-top: 10px;
        }
        QPushButton:hover {
            background-color: #f0f0f0;
        }
        """)
        clear_button.clicked.connect(lambda: self.clear_logs(log_text))

        layout.addWidget(log_text)
        layout.addWidget(clear_button)
        dialog.setLayout(layout)
        dialog.exec_()

    def clear_logs(self, log_text_widget):
        try:
            with open('model_inference.log', 'w', encoding='utf-8') as f:
                f.write("")
            log_text_widget.setPlainText("")
            QMessageBox.information(None, "清除成功", "日志已清空。")
        except Exception as e:
            QMessageBox.warning(None, "清除失败", f"发生错误：{str(e)}")