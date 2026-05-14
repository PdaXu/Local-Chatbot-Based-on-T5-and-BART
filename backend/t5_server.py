# backend/t5_server.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 初始化应用
app = Flask(__name__)

# 加载模型（只加载一次）
tokenizer = AutoTokenizer.from_pretrained(r'D:\study\MGW\ChatAPP\models\T5')
model = AutoModelForSeq2SeqLM.from_pretrained(r'D:\study\MGW\ChatAPP\models\T5')

@app.route('/chat', methods=['POST'])
def chat():
    # 从POST请求中拿到文本
    data = request.get_json()
    user_text = data.get('text', '')

    if not user_text:
        return jsonify({'reply': '（没有输入内容）'})

    # 模型推理
    inputs = tokenizer(user_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 返回JSON格式的回复
    return jsonify({'reply': reply})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)  # 本地开一个小服务器，监听5000端口
