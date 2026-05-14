# frontend/chat_new.py
import sys
import requests
import speech_recognition as sr
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QComboBox, QLabel, QHBoxLayout, QScrollArea, QFrame, QSplitter, QDialog, QTextBrowser, QListWidget, QFileDialog, QScrollArea, QFrame, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTextCursor, QPixmap, QPainter, QRegion, QBitmap, QImage, QColor
import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
from chat_db import init_db, save_record, get_all_records
from PyQt5.QtGui import QIcon, QMovie
import pyttsx3
import os
import psutil
import pyqtgraph as pg
from pynvml import *  # 用于获取GPU信息
import logging
import time

# 配置日志
logging.basicConfig(filename='model_inference.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.engine = pyttsx3.init()

        # 获取系统中可用的语音列表
        self.voices = self.engine.getProperty('voices')
        # 设置默认语音，这里选择第一个语音
        self.engine.setProperty('voice', self.voices[0].id)

        self.init_ui()
        self.vosk_model = Model(r"D:\study\MGW\ChatAPP\frontend\vosk")
        self.q = queue.Queue()
        init_db()  # 初始化数据库
        self.loading_label = None
        self.loading_movie = None
        self.chat_history_data = []
        # 初始化GPU库
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)  # 假设只有一个GPU

        # 初始化CPU和GPU使用率数据
        self.cpu_data = []
        self.gpu_data = []
        self.max_data_points = 100  # 最多显示100个数据点

        # 初始化网络流量数据
        self.prev_network_io = psutil.net_io_counters()

    def init_ui(self):
        self.setWindowTitle('基于深度学习的语音文本交互系统')
        self.resize(900, 600)  # 适当增大窗口宽度以容纳侧栏

        # 设置标题栏颜色为蓝色
        self.setStyleSheet("QWidget#centralWidget { background-color: #e6f7ff; }")

        # 设置标题栏图标
        icon = QIcon(r"D:\study\MGW\ChatAPP\icons\img4.png")
        self.setWindowIcon(icon)

        main_layout = QVBoxLayout(self)

        # 侧栏布局
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar.setStyleSheet("background-color: #f0f0f0; padding: 10px;")

        # 模型选择
        self.model_selector = QComboBox()
        self.model_selector.addItems(['T5', 'BART'])
        self.model_selector.setStyleSheet("""
        QComboBox {
            padding: 5px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: white;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left-width: 1px;
            border-left-color: #ccc;
            border-left-style: solid;
            border-top-right-radius: 5px;
            border-bottom-right-radius: 5px;
        }
        QComboBox::down-arrow {
            image: url(down_arrow.png);
            width: 10px;
            height: 10px;
        }
        """)
        sidebar_layout.addWidget(self.model_selector)

        # 语音选择按钮
        self.voice_button = QPushButton("声音选择")
        self.voice_button.setStyleSheet("""
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
        sidebar_layout.addWidget(self.voice_button)
        self.voice_button.clicked.connect(self.show_voice_dialog)

        # 历史记录按钮
        self.history_button = QPushButton("历史记录")
        self.history_button.setStyleSheet("""
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
        sidebar_layout.addWidget(self.history_button)
        self.history_button.clicked.connect(self.show_history)

        # 导出按钮
        self.export_button = QPushButton("导出聊天记录")
        self.export_button.setStyleSheet("""
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
        sidebar_layout.addWidget(self.export_button)
        self.export_button.clicked.connect(self.export_chat_records)

        # 系统监控按钮
        self.monitor_button = QPushButton("系统监控")
        self.monitor_button.setStyleSheet("""
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
        sidebar_layout.addWidget(self.monitor_button)
        self.monitor_button.clicked.connect(self.show_monitor_dialog)

        # 日志查看按钮
        self.log_button = QPushButton("查看日志")
        self.log_button.setStyleSheet("""
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
        sidebar_layout.addWidget(self.log_button)
        self.log_button.clicked.connect(self.show_logs)

        sidebar_layout.addStretch(1)  # 占位拉伸

        # 关于按钮
        self.about_button = QPushButton("关于")
        self.about_button.setStyleSheet("""
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
        sidebar_layout.addWidget(self.about_button)
        self.about_button.clicked.connect(self.show_about)

        # 主聊天区域布局
        main_content = QWidget()
        main_content_layout = QVBoxLayout(main_content)

        # 聊天显示区域（富文本）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ccc; border-radius: 5px;")
        scroll.setFixedSize(700, 500)

        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.addStretch(1)  # 占位拉伸

        scroll.setWidget(self.chat_widget)
        main_content_layout.addWidget(scroll)

        # 输入行和发送按钮
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("输入消息...")
        self.input_box.setStyleSheet("padding: 10px; font-size: 14px; border: 1px solid #ccc; border-radius: 5px;")

        self.send_button = QPushButton("发送")
        self.send_button.setStyleSheet("""
        QPushButton {
            background-color: #0099ff;
            color: white;
            padding: 10px 20px;
            font-weight: bold;
            border: none;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #007acc;
        }
        """)

        self.voice_input_button = QPushButton("语音输入")
        self.voice_input_button.setStyleSheet("""
        QPushButton {
            padding: 10px 20px;
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: white;
        }
        QPushButton:hover {
            background-color: #f0f0f0;
        }
        """)
        self.voice_input_button.setCheckable(True)

        input_layout.addWidget(self.input_box, 6)
        input_layout.addWidget(self.send_button, 2)
        input_layout.addWidget(self.voice_input_button, 2)

        main_content_layout.addLayout(input_layout)

        self.send_button.clicked.connect(self.handle_send)
        self.recording = False
        self.voice_input_button.clicked.connect(self.toggle_recording)

        # 使用QSplitter分隔侧栏和主聊天区域
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(main_content)
        main_layout.addWidget(splitter)

    def create_message_bubble(self, text, is_user=True, model=None):
        # 创建QQ风格消息气泡
        bubble = QLabel()
        bubble.setText(text)  # 设置文本
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

        # 加载头像
        if is_user:
            pixmap = QPixmap(r"D:\study\MGW\ChatAPP\icons\img3.png")
        elif model == 'T5':
            pixmap = QPixmap(r"D:\study\MGW\ChatAPP\icons\img1.png")
        else:
            pixmap = QPixmap(r"D:\study\MGW\ChatAPP\icons\img2.png")

        # 将头像缩放
        pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

        # 将头像设置为圆形
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

        # 创建语音播放按钮
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
        # 将 \n 替换为 <br>
        text = text.replace('\n', '<br>')
        is_user = sender == "user"
        bubble, avatar, voice_play_button = self.create_message_bubble(text, is_user, model)

        self.chat_history_data.append({
            "role": "用户" if is_user else (model or "模型"),
            "model": model or "",
            "text": text
        })

        row = QHBoxLayout()

        outer_row = QHBoxLayout()  # 最外层横向布局

        if is_user:
            # 用户消息：右对齐，头像在右边
            outer_row.addStretch(1)

            # 用户消息气泡和按钮布局
            user_content_layout = QHBoxLayout()
            user_content_layout.addWidget(bubble)
            if voice_play_button:
                user_content_layout.addWidget(voice_play_button, alignment=Qt.AlignTop)
                user_content_layout.addSpacing(5)  # 按钮和头像之间的间距
            else:
                user_content_layout.addSpacing(10)  # 没有按钮时的间距

            outer_row.addLayout(user_content_layout)
            outer_row.addWidget(avatar)
        else:
            # 模型名标签
            model_label = QLabel(model or "模型")
            model_label.setStyleSheet("""
                color: #666666;
                font-size: 11px;
                font-weight: bold;
                margin-left: 5px;
                margin-bottom: 2px;
            """)
            model_label.setFixedHeight(16)

            # 垂直布局：模型名 + 气泡
            content_layout = QVBoxLayout()
            content_layout.addWidget(model_label, alignment=Qt.AlignLeft)
            content_layout.addWidget(bubble)

            # 横向布局：气泡 + 播放按钮
            bubble_and_button_layout = QHBoxLayout()
            bubble_and_button_layout.addLayout(content_layout)
            if voice_play_button:
                bubble_and_button_layout.addWidget(voice_play_button, alignment=Qt.AlignTop | Qt.AlignLeft)
                bubble_and_button_layout.addSpacing(5)  # 按钮和拉伸之间的间距
            else:
                bubble_and_button_layout.addSpacing(10)  # 没有按钮时的间距
            bubble_and_button_layout.addStretch(1)

            outer_row.addWidget(avatar)
            outer_row.addSpacing(5)
            outer_row.addLayout(bubble_and_button_layout)
            outer_row.addStretch(1)

        self.chat_layout.insertLayout(self.chat_layout.count() - 1, outer_row)


    def handle_send(self):
        user_input = self.input_box.text().strip()
        if not user_input:
            return
        self.append_chat(user_input, sender="user")
        model = self.model_selector.currentText()

        # 显示加载动画
        self.show_loading_animation()

        reply = self.query_model(user_input, model)

        # 隐藏加载动画
        self.hide_loading_animation()

        self.append_chat(reply, sender="bot", model=model)
        self.input_box.clear()

        # 保存记录
        save_record(model, user_input, reply)

    def query_model(self, user_input, model_name):
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

    def toggle_recording(self):
        if not self.recording:
            self.voice_input_button.setText("停止录音")
            self.recording = True
            self.start_voice_capture()
        else:
            self.voice_input_button.setText("语音输入")
            self.recording = False
            self.stop_voice_capture()

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
        if result:
            self.input_box.setText(result)
            self.handle_send()
        else:
            self.input_box.setText("（语音识别失败）")

    def show_history(self):
        records = get_all_records()

        dialog = QDialog(self)
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
        dialog = QDialog(self)
        dialog.setWindowTitle("关于作者")
        dialog.resize(300, 200)
        dialog.setStyleSheet("background-color: #e6f7ff;")

        layout = QVBoxLayout()
        browser = QTextBrowser()
        browser.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 5px; padding: 10px;")
        browser.append("作者：PdaXu")
        #browser.append("作者：徐潘登")
        #browser.append("学号：202121335056")
        browser.append("系统名称：基于深度学习的语音文本交互系统")
        browser.append("版本：1.7v")

        layout.addWidget(browser)
        dialog.setLayout(layout)
        dialog.exec_()

    def show_loading_animation(self):
        self.loading_label = QLabel(self)
        self.loading_movie = QMovie(r"D:\study\MGW\ChatAPP\icons\loading.gif")  # 请替换为实际的加载动画 GIF 路径
        self.loading_label.setMovie(self.loading_movie)
        self.loading_movie.start()
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, self.loading_label)

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
        dialog = QDialog(self)
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

    def export_chat_records(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "导出聊天记录", "",
                                                   "文本文件 (*.txt);;图片文件 (*.png)", options=options)
        if file_name:
            if _.startswith("文本文件"):
                if not file_name.endswith('.txt'):
                    file_name += '.txt'
                self.export_to_txt(file_name)
            elif _.startswith("图片文件"):
                if not file_name.endswith('.png'):
                    file_name += '.png'
                self.export_to_image(file_name)

    def export_to_txt(self, filename):
        if not filename:
            return
        try:
            with open(filename, "w", encoding="utf-8") as f:
                for record in self.chat_history_data:
                    role = record["role"]
                    model = record["model"]
                    text = record["text"]
                    if role == "用户":
                        f.write(f"用户（{model}）：{text}\n")
                    else:
                        f.write(f"{model}：{text}\n")
            QMessageBox.information(self, "导出成功", f"聊天记录已保存到 {filename}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"发生错误：{str(e)}")

    def export_to_image(self, file_name):
        # 获取聊天区域的截图
        pixmap = self.chat_widget.grab()
        image = pixmap.toImage()
        # 保存为图片
        image.save(file_name)

    def show_monitor_dialog(self):
        self.monitor_dialog = QDialog(self)
        self.monitor_dialog.setWindowTitle("系统性能监控")
        self.monitor_dialog.resize(600, 400)
        self.monitor_dialog.setStyleSheet("background-color: #e6f7ff;")

        layout = QVBoxLayout()

        # 创建CPU和GPU使用率的绘图窗口
        self.cpu_plot = pg.PlotWidget(title="CPU使用率")
        self.gpu_plot = pg.PlotWidget(title="GPU使用率")
        self.cpu_curve = self.cpu_plot.plot(pen='r')
        self.gpu_curve = self.gpu_plot.plot(pen='b')

        layout.addWidget(self.cpu_plot)
        layout.addWidget(self.gpu_plot)

        # 添加用于显示网络发送、网络接收和内存占用率的标签
        self.network_send_label = QLabel()
        self.network_recv_label = QLabel()
        self.memory_usage_label = QLabel()

        layout.addWidget(self.network_send_label)
        layout.addWidget(self.network_recv_label)
        layout.addWidget(self.memory_usage_label)

        self.monitor_dialog.setLayout(layout)
        self.monitor_dialog.show()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_monitor_info)
        self.timer.start(1000)  # 每秒更新一次

    def update_monitor_info(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        gpu_info = nvmlDeviceGetUtilizationRates(self.handle)
        gpu_percent = gpu_info.gpu

        # 更新CPU和GPU使用率数据
        self.cpu_data.append(cpu_percent)
        self.gpu_data.append(gpu_percent)

        # 保持数据长度不超过最大数据点
        if len(self.cpu_data) > self.max_data_points:
            self.cpu_data = self.cpu_data[-self.max_data_points:]
        if len(self.gpu_data) > self.max_data_points:
            self.gpu_data = self.gpu_data[-self.max_data_points:]

        # 更新绘图
        self.cpu_curve.setData(self.cpu_data)
        self.gpu_curve.setData(self.gpu_data)

        # 获取网络流量和内存占用率
        current_network_io = psutil.net_io_counters()
        network_send = current_network_io.bytes_sent - self.prev_network_io.bytes_sent
        network_recv = current_network_io.bytes_recv - self.prev_network_io.bytes_recv
        self.prev_network_io = current_network_io

        memory_percent = psutil.virtual_memory().percent

        # 更新标签文本
        self.network_send_label.setText(f"网络发送: {network_send} 字节/秒")
        self.network_recv_label.setText(f"网络接收: {network_recv} 字节/秒")
        self.memory_usage_label.setText(f"内存占用率: {memory_percent}%")

    def show_logs(self):
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
                QMessageBox.warning(self, "日志文件不存在", "未找到模型推理日志文件。")
                return
            except Exception as e:
                QMessageBox.warning(self, "读取日志失败", f"发生错误：{str(e)}")
                return

        if logs is None:
            QMessageBox.warning(self, "读取日志失败", "无法使用支持的编码读取日志文件。")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("模型推理日志")
        dialog.resize(600, 400)
        dialog.setStyleSheet("background-color: #e6f7ff;")

        layout = QVBoxLayout()
        log_text = QTextEdit()
        log_text.setReadOnly(True)
        log_text.setPlainText(logs)
        log_text.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 5px; padding: 10px; font-family: monospace;")

        # 添加清除日志按钮
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
            QMessageBox.information(self, "清除成功", "日志已清空。")
        except Exception as e:
            QMessageBox.warning(self, "清除失败", f"发生错误：{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())