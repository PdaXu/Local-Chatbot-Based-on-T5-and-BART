# backend/bart_server.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 初始化应用
app = Flask(__name__)

# 加载BART模型（只加载一次）
tokenizer = AutoTokenizer.from_pretrained(r'D:\study\MGW\ChatAPP\models\BART')
model = AutoModelForSeq2SeqLM.from_pretrained(r'D:\study\MGW\ChatAPP\models\BART')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_text = data.get('text', '')

    if not user_text:
        return jsonify({'reply': '（没有输入内容）'})

    inputs = tokenizer(user_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100,do_sample=True,top_k=50,top_p=0.95)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'reply': reply})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001)  # 注意BART用5001端口
