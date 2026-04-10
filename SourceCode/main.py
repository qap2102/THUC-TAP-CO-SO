import torch
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ====== CONFIG ======
MODEL_PATH = "models/vit5_checkpoints"   # đổi nếu cần
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====== LOAD MODEL ======
if not os.path.exists(MODEL_PATH):
    raise ValueError(f"Không tìm thấy model tại: {MODEL_PATH}")

print(f"[INFO] Loading model from: {MODEL_PATH}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()


# ====== FUNCTION SỬA LỖI ======
def correct_text(text):
    prefixes = [
        "gec: ",
        "fix: ",
        "sửa lỗi chính tả: ",
        ""
    ]

    for p in prefixes:
        input_text = p + text

        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"[{p}] -> {result}")


# ====== TEST ======
def run_test():
    print("\n===== TEST SPELL CHECKER =====\n")

    test_sentences = [
        "Học sinh ddang học bài tasng lớp.",
        "Hnoay tơi ddi h0c ở truwờng PTIT.",
        "hom nay toi di hoc"
    ]

    for text in test_sentences:
        corrected = correct_text(text)

        print("Input :", text)
        print("Output:", corrected)
        print("-" * 40)
    
    print("RAW:", outputs[0])


# ====== MAIN ======
if __name__ == "__main__":
    run_test()