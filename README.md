# 📝 Nghiên cứu Hệ thống Sửa lỗi Chính tả Tiếng Việt

Dự án thuộc học phần **Thực tập cơ sở** - Học viện Công nghệ Bưu chính Viễn thông (PTIT). Hệ thống tập trung vào việc thu thập dữ liệu chuẩn, tạo nhiễu và huấn luyện mô hình Transformer để sửa lỗi chính tả tiếng Việt.

## 👥 Thông tin dự án
* **Sinh viên thực hiện:** Phạm Quang Anh
* **Mã sinh viên:** B23DCCN044
* **Giảng viên hướng dẫn:** TS. Kim Ngọc Bách
* **Đơn vị:** Khoa Công nghệ thông tin 1 - PTIT

---

# File Tree: THUC-TAP-CO-SO


```
├── 📁 Documents
│   ├── 📁 FinalReport
│   │   └── ⚙️ .gitkeep
│   ├── 📁 MidtermReport
│   │   └── 📕 B23DCCN044_MidtermReport .pdf
│   ├── 📁 WeeklyReports
│   │   ├── ⚙️ .gitkeep
│   │   ├── 📕 B23DCCN044_04-04-2026.pdf
│   │   ├── 📕 B23DCCN044_07-03-2026.pdf
│   │   ├── 📕 B23DCCN044_14-03-2026.pdf
│   │   ├── 📕 B23DCCN044_21-03-2026.pdf
│   │   └── 📕 B23DCCN044_28-03-2026.pdf
│   └── ⚙️ .gitkeep
├── 📁 SourceCode
│   ├── 📁 data
│   │   ├── 📁 processed
│   │   │   ├── 📄 data_report.csv
│   │   │   └── 📄 train_data.csv
│   │   └── 📁 raw
│   │       └── 📄 gold_data_wikipedia.txt
│   ├── 📁 models
│   │   └── 📁 vit5_checkpoints
│   │       ├── ⚙️ config.json
│   │       ├── ⚙️ generation_config.json
│   │       ├── 📄 model.safetensors
│   │       ├── 🐍 model_init.py
│   │       ├── ⚙️ tokenizer.json
│   │       ├── ⚙️ tokenizer_config.json
│   │       └── ⚙️ training_args.bin
│   ├── 📁 notebooks
│   │   └── 📄 01_eda_tokenizer.ipynb
│   ├── 📁 resources
│   │   └── 📄 dictionary.txt
│   ├── 📁 src
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 build_dict.py
│   │   ├── 🐍 data_processor.py
│   │   ├── 🐍 data_scraper.py
│   │   └── 🐍 noising_system.py
│   ├── ⚙️ .gitkeep
│   ├── 🐍 data_pipeline.py
│   └── 🐍 main.py
├── ⚙️ .dockerignore
├── ⚙️ .gitattributes
├── 🐳 Dockerfile
├── 📝 README.md
└── 📄 requirements.txt
```


## 🛠 Hướng dẫn cài đặt & Chạy (Setup Guide)

1. Sử dụng Docker (Khuyên dùng cho Cloud/DevOps)
Phương pháp này đảm bảo môi trường đồng nhất, sẵn sàng triển khai trên VPS/Cloud.

Bước 1: Build Docker Image
$ docker build -t spelling-app .

Bước 2: Chạy hệ thống & Đồng bộ dữ liệu
$ docker run -v "${PWD}/SourceCode:/app/SourceCode" spelling-app
(Dữ liệu huấn luyện sinh ra sẽ tự động xuất hiện trong thư mục SourceCode/data)

2. Sử dụng Python Local
Bước 1: Cài đặt thư viện: $ pip install -r requirements.txt
Bước 2: Chạy script: $ python SourceCode/main.py