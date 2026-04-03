# 1. Sử dụng Python phiên bản nhẹ
FROM python:3.10-slim

# 2. Cài đặt các công cụ cần thiết cho thư viện NLP (như underthesea)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Thiết lập thư mục làm việc
WORKDIR /app

# 4. Cài đặt thư viện trước để tận dụng cache của Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ dự án vào /app
COPY . .

# 6. QUAN TRỌNG: Thiết lập PYTHONPATH
# Giúp Python hiểu rằng các module nằm trong SourceCode/src
ENV PYTHONPATH=/app/SourceCode

# 7. Lệnh chạy mặc định
# Lưu ý đường dẫn mới theo hình của bạn: SourceCode/main.py
CMD ["python", "SourceCode/main.py"]