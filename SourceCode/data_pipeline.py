import os
import sys

# Đảm bảo Python tìm thấy thư mục src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))
from src.data_scraper import DataScraper
from src.noising_system import AdvancedVietnameseNoiser

# --- CẤU HÌNH ĐƯỜNG DẪN CHUẨN ---
GOLD_PATH = os.path.join(BASE_DIR, "data", "raw", "gold_data_wikipedia.txt")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "train_data.csv")
REPORT_PATH = os.path.join(BASE_DIR, "data", "processed", "data_report.csv")
DICTIONARY_PATH = os.path.join(BASE_DIR, "resources", "dictionary.txt")

def main():
    # Bước 1: Thu thập dữ liệu (Tuần 3)
    if not os.path.exists(GOLD_PATH):
        print("\n=== BƯỚC 1: THU THẬP DỮ LIỆU CHUẨN ===")
        scraper = DataScraper(report_path=REPORT_PATH)
        sources = {
            "Wikipedia AI": "https://vi.wikipedia.org/wiki/Tr%C3%AD_tu%E1%BB%87_nh%C3%A2n_t%E1%BA%A1o",
            "VnExpress": "https://vnexpress.net/chu-tich-fpt-ai-se-thay-doi-moi-nganh-nghe-4724567.html"
        }
        final_data = []
        for name, url in sources.items():
            final_data.extend(scraper.process_url(url, name))
        
        os.makedirs(os.path.dirname(GOLD_PATH), exist_ok=True)
        with open(GOLD_PATH, "w", encoding="utf-8") as f:
            for line in final_data: f.write(line + "\n")
        scraper.save_report()

    # Bước 2: Tạo dữ liệu lỗi (Tuần 4)
    print("\n=== BƯỚC 2: TẠO DỮ LIỆU HUẤN LUYỆN (NOISING) ===")
    noiser = AdvancedVietnameseNoiser()
    noiser.create_dataset(GOLD_PATH, TRAIN_CSV_PATH)

    # Bước 3: Sẵn sàng cho Tuần 5 (Setup Model)
    print("\n=== BƯỚC 3: SẴN SÀNG CHO TUẦN 5 ===")
    print(f"Dữ liệu huấn luyện đã sẵn sàng tại: {TRAIN_CSV_PATH}")
    print("Tiếp theo: Mở src/model_trainer.py để khởi tạo ViT5.")

if __name__ == "__main__":
    main()