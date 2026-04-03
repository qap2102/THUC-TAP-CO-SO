import torch
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Lấy đường dẫn đến thư mục chứa file main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TRỎ ĐÚNG VÀO THƯ MỤC CHỨA CÁC FILE JSON VÀ SAFETENSORS
# Cấu trúc của bạn: models \ vit5_checkpoints nằm cùng cấp hoặc trong SourceCode
MODEL_PATH = os.path.join(BASE_DIR, "models", "vit5_checkpoints")

# Kiểm tra thực tế xem Python có thấy thư mục này không
if not os.path.exists(MODEL_PATH):
    print(f"LỖI: Không tìm thấy thư mục model tại: {MODEL_PATH}")
    # Nếu main.py nằm ngoài SourceCode, hãy thử:
    # MODEL_PATH = os.path.join(BASE_DIR, "SourceCode", "models", "vit5_checkpoints")
else:
    print(f"Đã nạp thành công Model từ: {MODEL_PATH}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

def run_spell_checker():
    print("--- ĐANG NẠP MÔ HÌNH VI T5 ---")
    # Nạp tokenizer và model từ local
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    
    test_sentences = [
        "Học sinh ddang học bài tasng lớp.",
        "Hnoay tơi ddi h0c ở truwờng PTIT."
    ]

    for text in test_sentences:
        # Thêm prefix 'gec: ' để kích hoạt khả năng sửa lỗi
        inputs = tokenizer("gec: " + text, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                max_length=128, 
                num_beams=4, 
                early_stopping=True
            )
        
        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Lỗi: {text}")
        print(f"Sửa: {corrected}")
        print("-" * 30)

if __name__ == "__main__":
    run_spell_checker()