# Local-Chatbot-Based-on-T5-and-BART
本项目基于深度学习技术，设计并实现了一套本地离线语音文本交互系统。系统集成 T、BART 双中文生成模型、Vosk 离线语音识别引擎、PyQt5 图形界面与 Flask 后端服务，完整支持语音输入、文本问答、模型动态切换、聊天历史管理、记录导出及系统实时监控等核心功能。系统全程无需联网、可在普通本地电脑独立部署运行，兼顾实用性、安全性与良好交互体验。受限于个人计算机硬件资源，模型训练规模与迭代次数有限，导致生成能力受限，暂无法生成长篇幅、复杂连贯的回答内容。

一、T5 后端环境（glm）
conda create -n glm python=3.9 -y
conda activate glm
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install transformers flask flask-cors safetensors
二、BART 后端环境（T5Bot）
conda create -n T5Bot python=3.9 -y
conda activate T5Bot
pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install transformers flask flask-cors safetensors
三、前端 PyQt5 环境（chatapp）
conda create -n chatapp python=3.9 -y
conda activate chatapp
pip install PyQt5 requests vosk sounddevice psutil pillow

Speech-QA-System/
├── app_debug.log              
├── chat_history.db
├── model_inference.log
├── start_servers.bat  
│
├── backend/                
│   ├── bart_chatbot.py
│   ├── bart_part.py
│   ├── bart_server.py
│   ├── t5_chatbot.py
│   ├── t5_part.py
│   └── t5_server.py
│
├── frontend/           
│   ├── app.py
│   ├── chat_new_beautiful3.py
│   ├── gradio.py
│   └── vosk/            
│
├── icons/                  
│   └── pictures
│
└── models
    ├── BART
    └── T5

以下是模型下载
通过网盘分享的文件：T5(1).zip等2个文件
链接: https://pan.baidu.com/s/1zGqQgANs7aXq1g6Jm85Bng?pwd=1111 提取码: 1111
