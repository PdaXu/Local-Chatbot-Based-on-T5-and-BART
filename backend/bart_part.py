# backend/bart_chatbot.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
from pathlib import Path

def load_model():
    model_path = Path(__file__).parent.parent / "models" / "bart"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    return tokenizer, model

tokenizer, model = load_model()

def chat_bart(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs,
                              max_new_tokens=50,
                              do_sample=True,   # 开启采样
                              top_k=50,         # 从top50个词中采样
                              top_p=0.9,        # 或使用nucleus sampling
                              temperature=0.8,  # 降低随机性
                              num_beams=2       # 不用beam search，随机更强
                              )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    user_input = sys.stdin.read().strip()
    print(chat_bart(user_input))