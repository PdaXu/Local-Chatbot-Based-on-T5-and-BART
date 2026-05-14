# backend/t5_part.py
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(r'D:\study\MGW\ChatAPP\models\T5')
model = AutoModelForSeq2SeqLM.from_pretrained(r'D:\study\MGW\ChatAPP\models\T5')

def chat_t5(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

if __name__ == "__main__":
    try:
        user_input = sys.stdin.read().strip()
        if not user_input:
            print("（未接收到输入）")
        else:
            bot_reply = chat_t5(user_input)
            print(bot_reply)
    except Exception as e:
        print(f"程序异常：{str(e)}")
