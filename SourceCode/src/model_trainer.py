import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    MBartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)



# ===== CONFIG =====
MODEL_NAME = "vinai/bartpho-syllable"
OUTPUT_DIR = "models/bartpho_finetuned"
MAX_LEN = 128

# ===== LOAD DATA VÀ LÀM SẠCH =====
df = pd.read_csv("data/processed/train_data.csv")

print(f"Tổng số mẫu ban đầu: {len(df)}")
df = df.dropna(subset=['source', 'target'])
df = df.drop_duplicates()
df = df[df['source'].str.strip() != df['target'].str.strip()]
print(f"Số mẫu sau khi làm sạch: {len(df)}")

# Prefix cho task (BARTpho không bắt buộc dùng prefix như T5)
df["input"] = df["source"]

dataset = Dataset.from_pandas(df)

# ===== TOKENIZER =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    model_inputs = tokenizer(
        example["input"],
        max_length=MAX_LEN,
        truncation=True
    )

    labels = tokenizer(
        example["target"],
        max_length=MAX_LEN,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(tokenize_function, batched=True)

# ===== SPLIT DATA =====
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ⚠️ set format torch (tránh bug ngầm)
train_dataset.set_format(type="torch")
eval_dataset.set_format(type="torch")

# ===== MODEL =====
print("Đang tải mô hình BARTpho...")
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

# 5. Data Collator cho Dynamic Padding (Tăng tốc xử lý câu ngắn)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)



# ===== TRAINING CONFIG =====
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    
    # 1. Tăng tốc độ tính toán bằng Mixed Precision (BẮT BUỘC)
    fp16=True, 
    
    # 2. Gộp Batch để chạy nhanh hơn
    per_device_train_batch_size=4,   # Giữ ở mức 4-8 để không bị Out of Memory
    gradient_accumulation_steps=8,    # Chỉ cập nhật sau mỗi 32 mẫu (4x8) giúp giảm số bước lặp (steps)
    
    num_train_epochs=5,               # 10 là hơi nhiều cho fine-tuning, 3-5 là đủ
    eval_strategy="steps",            # Nên eval theo số bước thay vì đợi hết cả epoch
    eval_steps=500,
    save_steps=500,
    
    # 3. Tắt cái này khi train để nhanh hơn
    predict_with_generate=False,      # Khi eval chỉ tính Loss thôi, dùng generate cực kỳ chậm
    
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    
    # 4. Tối ưu việc nạp dữ liệu từ CPU sang GPU
    dataloader_pin_memory=True,
    dataloader_num_workers=2,

    # 6. Gom các câu có độ dài gần nhau để giảm padding dư thừa
    group_by_length=True,
    
    # 7. Optimizer nhanh hơn (nếu có GPU)
    optim="adamw_torch_fused"
)

# ===== TRAINER =====
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ===== TRAIN =====
trainer.train()

# ===== SAVE =====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training completed!")