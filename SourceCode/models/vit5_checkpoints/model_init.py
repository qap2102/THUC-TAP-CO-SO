# src/model_init.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def check_env():
    model_name = "VietAI/vit5-base" # Bạn có thể đổi thành "vinai/bartpho-word"
    print(f"[*] Đang tải Tokenizer cho {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"[*] Đang tải Model {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    print(f"[+] Thiết lập thành công! Thiết bị đang dùng: {device.upper()}")
    return tokenizer, model

if __name__ == "__main__":
    check_env()