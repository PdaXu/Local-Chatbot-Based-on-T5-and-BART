# backend/bart_chatbot.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(r'D:\study\MGW\ChatAPP\models\BART')
model = AutoModelForSeq2SeqLM.from_pretrained(r'D:\study\MGW\ChatAPP\models\BART')

def chat_bart(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

if __name__ == "__main__":
    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        bot_reply = chat_bart(user_input)
        print("BART回复：", bot_reply)
