import os
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from underthesea import sent_tokenize

class DataScraper:
    def __init__(self, report_path):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.stats = []
        self.report_path = report_path

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'http\S+', '', text)
        return text

    def process_url(self, url, source_name):
        print(f"[*] Đang xử lý {source_name}: {url}")
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            content = " ".join([p.get_text() for p in soup.find_all('p')])
            sentences = sent_tokenize(content)
            
            clean_sentences = []
            word_count = 0
            for s in sentences:
                s_clean = self.clean_text(s)
                words = s_clean.split()
                if 5 <= len(words) <= 50:
                    clean_sentences.append(s_clean)
                    word_count += len(words)
            
            self.stats.append({
                'Nguồn': source_name, 'Số câu': len(clean_sentences),
                'Số từ': word_count, 'URL': url
            })
            return clean_sentences
        except Exception as e:
            print(f"[!] Lỗi tại {source_name}: {e}")
            return []

    def save_report(self):
        df = pd.DataFrame(self.stats)
        df.to_csv(self.report_path, index=False, encoding='utf-8-sig')
        print(f"[*] Đã xuất báo cáo thống kê tại: {self.report_path}")