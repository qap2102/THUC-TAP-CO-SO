import os
import sys

# Ép hệ thống tìm thư viện ở ổ E trước
sys.path.insert(0, "E:/python_libs")

import torch
from transformers import MBartForConditionalGeneration, AutoTokenizer
import gradio as gr

# Cấu hình cache và token Hugging Face
HF_TOKEN = "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"
CACHE_DIR = "E:/hf_cache" 
os.environ["HF_HOME"] = CACHE_DIR

model_id = "qap0310/TTCS"

# Tải cấu hình tự động GPU/CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Hệ thống đang chạy trên: {device}")

print("🔄 Đang tải Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, cache_dir=CACHE_DIR)

print("🔄 Đang tải Model...")
model = MBartForConditionalGeneration.from_pretrained(model_id, token=HF_TOKEN, cache_dir=CACHE_DIR).to(device)

# --- HÀM TỰ TÌM CÁC TỪ BỊ SỬA LỖI ---
def get_diff_highlights(original_text, corrected_text):
    """
    So sánh từng từ giữa câu gốc và câu đã sửa để tìm ra các từ thay đổi
    Trả về danh sách tuple theo định dạng của gr.HighlightedText: (từ, nhãn)
    """
    orig_words = original_text.strip().split()
    corr_words = corrected_text.strip().split()
    
    highlighted_data = []
    
    # Duyệt song song qua từng cặp từ (giả định cấu trúc câu không bị đảo lộn từ ngữ)
    for i in range(max(len(orig_words), len(corr_words))):
        # Nếu câu gốc dài hơn (bị thừa từ)
        if i >= len(corr_words):
            highlighted_data.append((orig_words[i], "Xóa"))
            continue
        # Nếu câu mới dài hơn (bị thiếu từ ban đầu)
        if i >= len(orig_words):
            highlighted_data.append((corr_words[i], f"Thêm"))
            continue
            
        w_orig = orig_words[i]
        w_corr = corr_words[i]
        
        # Làm sạch dấu câu để so sánh chính xác từ cốt lõi
        clean_orig = w_orig.strip(",.?!;:\"()").lower()
        clean_corr = w_corr.strip(",.?!;:\"()").lower()
        
        if clean_orig != clean_corr or w_orig != w_corr:
            # Nếu từ bị thay đổi, hiển thị dạng: hoc -> Học
            highlighted_data.append((w_corr, f"{w_orig} -> {w_corr}"))
        else:
            # Nếu từ viết đúng, giữ nguyên không gắn nhãn màu
            highlighted_data.append((w_corr, None))
            
    return highlighted_data


# --- HÀM SUY LUẬN CHÍNH ---
def predict(text):
    if not text.strip():
        return "", [], "0.00%"
        
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=256,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Giải mã văn bản kết quả
    generated_tokens = outputs.sequences[0]
    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Tính danh sách các từ khác biệt để highlight màu sắc
    diff_data = get_diff_highlights(text, decoded_text)
    
    # Tính toán % độ tự tin (Confidence Score)
    scores = outputs.scores
    probs = []
    for i, token_id in enumerate(generated_tokens[1:]):
        if i >= len(scores):
            break
        token_probs = torch.softmax(scores[i][0], dim=-1)
        probs.append(token_probs[token_id].item())
        
    if probs:
        confidence = sum(probs) / len(probs) * 100
        confidence_str = f"{confidence:.2f}%"
    else:
        confidence_str = "0.00%"
        
    return decoded_text, diff_data, confidence_str


# --- KHỞI TẠO GIAO DIỆN GRADIO DEMO ---
print("🚀 Đang khởi chạy giao diện Web UI...")
with gr.Blocks(title="Hệ thống Sửa lỗi chính tả Tiếng Việt") as demo:
    gr.Markdown("# 📝 Hệ thống Demo Báo cáo Thực tập Cơ sở")
    gr.Markdown("### Mô hình Seq2Seq fine-tuned trên nền BARTpho chuyên sửa lỗi chính tả và thêm dấu.")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(lines=4, placeholder="Nhập văn bản tiếng Việt không dấu hoặc sai chính tả...", label="Văn bản đầu vào")
            submit_btn = gr.Button("Xử lý sửa lỗi", variant="primary")
            clear_btn = gr.ClearButton(value="Xóa dữ liệu")
            
        with gr.Column():
            output_box = gr.Textbox(lines=4, label="Kết quả xử lý mô hình (Văn bản thuần)")
            
            # Giao diện hiển thị chi tiết các từ được gợi ý sửa đổi
            output_highlight = gr.HighlightedText(
                label="Bản đồ gợi ý sửa lỗi chi tiết (Từ gốc -> Từ sửa)",
                combine_adjacent=False,
                color_map={"Xóa": "red", "Thêm": "green"} # Các từ được sửa sẽ tự động có màu sắc nổi bật khác
            )
            
            output_confidence = gr.Textbox(label="Độ tự tin của mô hình (%)")

    # Thiết lập sự kiện tương tác
    submit_btn.click(
        fn=predict, 
        inputs=input_box, 
        outputs=[output_box, output_highlight, output_confidence]
    )
    clear_btn.add([input_box, output_box, output_highlight, output_confidence])

if __name__ == "__main__":
    demo.launch(share=False)