import os
import random
import pandas as pd
from tqdm import tqdm

class AdvancedVietnameseNoiser:
    def __init__(self):
        # 1. Quy luật OCR
        self.ocr_rules = {
            'o': ['0', 'ó'], '0': ['o'], 
            'i': ['l', '1'], 'l': ['i', '1'], 
            'g': ['9'], 'a': ['4'], 
            'e': ['3'], 's': ['5'],
            'u': ['v'], 'v': ['u']
        }
        # 2. Quy luật Telex
        self.telex_map = {
            'â': 'aa', 'ă': 'aw', 'ê': 'ee', 'ô': 'oo', 
            'ơ': 'ow', 'ư': 'uw', 'đ': 'dd', 'á': 'as', 'à': 'af',
            'ả': 'ar', 'ã': 'ax', 'ạ': 'aj'
        }
        # 3. Quy luật Bàn phím
        self.kb_neighbors = {
            'h': 'g', 'n': 'm', 'k': 'l', 'u': 'y', 'i': 'o', 
            'p': 'o', 't': 'r', 'g': 'f', 's': 'a', 'd': 's'
        }

    def inject_noise(self, text, method='random', error_rate=0.3):
        words = text.split()
        new_words = []
        
        for word in words:
            should_noise = True if method != 'random' else (random.random() < error_rate)
            
            if should_noise:
                current_method = method if method != 'random' else random.choice(['ocr', 'telex', 'kb', 'swap'])
                word_list = list(word)
                if not word_list: continue

                if current_method == 'telex':
                    for char, telex in self.telex_map.items():
                        if char in word:
                            word = word.replace(char, telex)
                            break
                elif current_method == 'ocr':
                    idx = random.randint(0, len(word_list) - 1)
                    char = word_list[idx].lower()
                    if char in self.ocr_rules:
                        word_list[idx] = random.choice(self.ocr_rules[char])
                    word = "".join(word_list)
                elif current_method == 'kb':
                    idx = random.randint(0, len(word_list) - 1)
                    char = word_list[idx].lower()
                    if char in self.kb_neighbors:
                        word_list[idx] = self.kb_neighbors[char]
                    word = "".join(word_list)
                elif current_method == 'swap' and len(word_list) > 1:
                    i = random.randint(0, len(word_list) - 2)
                    word_list[i], word_list[i+1] = word_list[i+1], word_list[i]
                    word = "".join(word_list)
            
            new_words.append(word)
        return " ".join(new_words)

    def create_dataset(self, gold_path, train_csv_path):
        if not os.path.exists(gold_path):
            print(f"[!] Không tìm thấy file {gold_path}.")
            return

        with open(gold_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        random.shuffle(lines) 
        dataset = []
        print(f"[*] ĐANG CHẠY CHẾ ĐỘ VÉT CẠN (5 loại lỗi cho mỗi câu)...")
        
        for line in tqdm(lines):
            # Sinh đủ 5 trường hợp để AI học đa dạng
            dataset.append({"source": self.inject_noise(line, method='telex'), "target": line})
            dataset.append({"source": self.inject_noise(line, method='ocr'), "target": line})
            dataset.append({"source": self.inject_noise(line, method='kb'), "target": line})
            dataset.append({"source": self.inject_noise(line, method='swap'), "target": line})
            dataset.append({"source": self.inject_noise(line, method='random', error_rate=0.3), "target": line})

        # Lưu dữ liệu
        df = pd.DataFrame(dataset).sample(frac=1).reset_index(drop=True)
        df.to_csv(train_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\n[+] Xong! Đã tạo {len(dataset)} dòng dữ liệu tại {train_csv_path}")