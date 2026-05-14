# frontend/chat_new.py
import sys
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QComboBox, QLabel, QHBoxLayout
)

class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle('T5 / BART 聊天应用')
        self.resize(600, 800)

        # 总布局
        layout = QVBoxLayout()

        # 聊天记录框
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        layout.addWidget(self.chat_history)

        # 输入框和发送按钮布局
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.send_button = QPushButton('发送')
        input_layout.addWidget(self.input_box, 4)
        input_layout.addWidget(self.send_button, 1)
        layout.addLayout(input_layout)

        # 模型选择下拉框
        self.model_selector = QComboBox()
        self.model_selector.addItems(['T5', 'BART'])
        layout.addWidget(QLabel('选择模型:'))
        layout.addWidget(self.model_selector)

        self.setLayout(layout)
        self.send_button.clicked.connect(self.handle_send)

    def handle_send(self):
        user_input = self.input_box.text().strip()
        if not user_input:
            return
        
        self.chat_history.append(f"你：{user_input}")
        selected_model = self.model_selector.currentText()
        reply = self.query_model(user_input, selected_model)
        self.chat_history.append(f"{selected_model} 回复：{reply}")
        self.input_box.clear()

    def query_model(self, user_input, model_name):
        # 根据模型选择不同端口
        if model_name == 'T5':
            url = "http://127.0.0.1:5000/chat"
        else:
            url = "http://127.0.0.1:5001/chat"

        try:
            response = requests.post(url, json={"text": user_input}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('reply', '（后端未返回回复）')
            else:
                return f"错误：HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"连接错误：{e}"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatApp()
    window.show()
    sys.exit(app.exec_())
