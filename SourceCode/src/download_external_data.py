import pandas as pd
from datasets import load_dataset
import os

OUTPUT_CSV = "data/processed/train_data.csv"

def download_and_merge_datasets():
    print("=== ĐANG TẢI DỮ LIỆU TỪ HUGGING FACE ===")
    
    merged_data = []

    try:
        # Load tập VNTC đã được hiệu chỉnh cho lỗi chính tả
        print("[1] Đang tải tập (bmd1905/error-correction-vi)...")
        vntc_dataset = load_dataset("bmd1905/error-correction-vi", split="train")
        
        # Tập này có khoảng vài triệu câu. Để chạy học tập nhẹ nhàng trên Colab, ta lấy 20.000 câu ngẫu nhiên.
        vntc_dataset = vntc_dataset.shuffle(seed=42).select(range(20000))
        
        count = 0
        for item in vntc_dataset:
            source = item.get('error_text', item.get('input', item.get('wrong', '')))
            target = item.get('correct_text', item.get('output', item.get('text', '')))
            if source and target:
                merged_data.append({
                    "source": source,
                    "target": target
                })
                count += 1
        print(f" [+] Đã lấy {count} câu từ tập vi-error-correction-2.0")
    except Exception as e:
        print(f" [!] Lỗi khi tải tập VNTC: {e}")

    try:
        # Load tập Viwiki/Spelling thông dụng (coung21/vi-spelling-correction)
        print("[2] Đang tải tập Viwiki-Spelling (coung21/vi-spelling-correction)...")
        viwiki_dataset = load_dataset("coung21/vi-spelling-correction", split="train")
        
        # Lấy thêm 20.000 câu ngẫu nhiên nữa
        viwiki_dataset = viwiki_dataset.shuffle(seed=42).select(range(20000))
        
        count_before = len(merged_data)
        for item in viwiki_dataset:
            # Cấu trúc phổ biến: 'text' = source, 'label'/'target' = target
            # Tùy config dataset, ta sẽ cố gắng lấy các trường phổ biến
            source_text = item.get('source', item.get('input', item.get('wrong', '')))
            target_text = item.get('target', item.get('output', item.get('text', '')))
            
            if source_text and target_text:
                merged_data.append({
                    "source": source_text,
                    "target": target_text
                })
        print(f" [+] Đã lấy {len(merged_data) - count_before} câu từ tập ViWiki.")
    except Exception as e:
        print(f" [!] Lỗi khi tải tập ViWiki: {e}")

    # ===== LƯU DỮ LIỆU ĐÈ LÊN FILE CŨ =====
    print(f"\n=== ĐANG LƯU {len(merged_data)} CÂU DỮ LIỆU MỚI VÀO FILE ===")
    
    final_df = pd.DataFrame(merged_data)
    
    # Làm sạch dữ liệu rác
    final_df = final_df.dropna()
    final_df = final_df.drop_duplicates()
    
    # Trộn ngẫu nhiên toàn bộ dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\n[+] HOÀN TẤT! File '{OUTPUT_CSV}' hiện đã bị ghi đè hoàn toàn bởi {len(final_df)} dòng dữ liệu từ VNTC & Viwiki.")

if __name__ == "__main__":
    download_and_merge_datasets()
