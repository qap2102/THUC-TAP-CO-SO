from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_PATH = "models/vit5_finetuned"

# load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("=== TEST MODEL ===")

while True:
    text = input("\nNhập câu sai: ")

    if text.strip() == "":
        break

    # ⚠️ QUAN TRỌNG: phải có prefix giống lúc train
    input_text = "gec: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    # generate
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

    # decode
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("→", result)