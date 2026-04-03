import os
import unicodedata
from underthesea import word_tokenize
from collections import Counter

# --- CẤU HÌNH ĐƯỜNG DẪN (Khớp với cấu trúc SourceCode của bạn) ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "gold_data_wikipedia.txt")
MODEL_DIR = os.path.join(CURRENT_DIR, "models")
DICT_PATH = os.path.join(MODEL_DIR, "dictionary.txt")

# Đảm bảo thư mục models tồn tại
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def build_dictionary(min_freq=2):
    print(f"[*] Đang xử lý file: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print("[!] Lỗi: Không tìm thấy dữ liệu đầu vào. Hãy chạy main.py trước!")
        return

    # 1. Đọc dữ liệu
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_words = []
    print("[*] Đang tách từ và làm sạch dữ liệu...")
    
    for line in lines:
        # Chuẩn hóa Unicode và đưa về chữ thường để tránh trùng lặp do viết hoa
        line = unicodedata.normalize('NFC', line.lower().strip())
        
        # Tách từ tiếng Việt chuẩn
        # format="text" giúp giữ các từ ghép như "học_sinh"
        tokens = word_tokenize(line, format="text").split()
        
        for token in tokens:
            # Thay thế dấu gạch dưới bằng khoảng trắng
            word = token.replace('_', ' ').strip()
            
            # Chỉ giữ lại nếu từ có chứa ít nhất 1 chữ cái (loại bỏ số/dấu câu thuần túy)
            if any(c.isalpha() for c in word) and len(word) > 1:
                all_words.append(word)

    # 2. Đếm tần suất xuất hiện
    word_counts = Counter(all_words)
    
    # 3. Loại bỏ trùng lặp bằng set() và lọc theo tần suất tối thiểu
    # set() sẽ tự động xóa các phần tử giống hệt nhau
    unique_vocab = set()
    for word, count in word_counts.items():
        if count >= min_freq:
            unique_vocab.add(word)

    # 4. Sắp xếp danh sách từ theo thứ tự bảng chữ cái (A-Z)
    final_vocab = sorted(list(unique_vocab))

    # 5. Ghi ra file dictionary.txt
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        for word in final_vocab:
            f.write(word + "\n")
            
    print("-" * 50)
    print(f"THÀNH CÔNG! Đã tạo từ điển tại: {DICT_PATH}")
    print(f"Số lượng từ vựng duy nhất (không trùng lặp): {len(final_vocab)}")
    print("=" * 50)

if __name__ == "__main__":
    build_dictionary(min_freq=2) # Bạn có thể tăng min_freq lên nếu muốn từ điển khắt khe hơn