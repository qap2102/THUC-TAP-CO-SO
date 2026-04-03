import os
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from underthesea import sent_tokenize, word_tokenize

# --- CẤU HÌNH ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
GOLD_PATH = os.path.join(DATA_DIR, "gold_data_wikipedia.txt")
REPORT_PATH = os.path.join(DATA_DIR, "data_report.csv")

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

class DataScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.stats = []

    def clean_text(self, text):
        # Loại bỏ các ký tự rác và chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        # Loại bỏ link URL ẩn trong text
        text = re.sub(r'http\S+', '', text)
        return text

    def process_url(self, url, source_name):
        print(f"[*] Đang xử lý {source_name}: {url}")
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            
            # Lấy toàn bộ text trong các thẻ p
            content = " ".join([p.get_text() for p in soup.find_all('p')])
            sentences = sent_tokenize(content)
            
            clean_sentences = []
            word_count = 0
            
            for s in sentences:
                s_clean = self.clean_text(s)
                # Chỉ lấy câu có từ 5 đến 50 từ (tránh câu quá ngắn hoặc quá dài/nhiễu)
                words = s_clean.split()
                if 5 <= len(words) <= 50:
                    clean_sentences.append(s_clean)
                    word_count += len(words)
            
            # Lưu thống kê cho báo cáo
            self.stats.append({
                'Nguồn': source_name,
                'Số câu': len(clean_sentences),
                'Số từ': word_count,
                'URL': url
            })
            return clean_sentences
        except Exception as e:
            print(f"[!] Lỗi: {e}")
            return []

    def save_report(self):
        df = pd.DataFrame(self.stats)
        df.to_csv(REPORT_PATH, index=False, encoding='utf-8-sig')
        print(f"[*] Đã xuất báo cáo thống kê tại: {REPORT_PATH}")

# --- THỰC THI ---
if __name__ == "__main__":
    scraper = DataScraper()
    
    # Danh sách nguồn thu thập (Bạn có thể thêm nhiều link)
    sources = {
        "Wikipedia": "https://vi.wikipedia.org/wiki/Tr%C3%AD_tu%E1%BB%87_nh%C3%A2n_t%E1%BA%A1o",
        "Báo điện tử": "https://vnexpress.net/chu-tich-fpt-ai-se-thay-doi-moi-nganh-nghe-4724567.html",
        "Báo nhân dân 1": "https://nhandan.vn/trao-huy-hieu-tuoi-tre-dung-cam-tang-sinh-vien-cuu-nguoi-khoi-dam-chay-o-linh-nam-post951116.html",
        "Báo nhân dân 2": "https://nhandan.vn/trung-uong-doan-thanh-nien-cong-san-ho-chi-minh-don-nhan-huan-chuong-lao-dong-hang-nhat-post951100.html",
        "Báo nhân dân 3": "https://nhandan.vn/viet-nam-va-bulgaria-day-manh-quan-he-huu-nghi-hop-tac-giua-cac-dia-phuong-post951154.html"
    }
    
    final_data = []
    for name, url in sources.items():
        data = scraper.process_url(url, name)
        final_data.extend(data)
        
    # Lưu file Gold Dataset
    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        for line in final_data:
            f.write(line + "\n")
            
    scraper.save_report()
    print(f"--- HOÀN THÀNH: Đã lưu {len(final_data)} câu chuẩn ---")