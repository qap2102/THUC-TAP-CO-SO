# 📝 Nghiên cứu Hệ thống Sửa lỗi Chính tả Tiếng Việt

Dự án thuộc học phần **Thực tập cơ sở** - Học viện Công nghệ Bưu chính Viễn thông (PTIT). Hệ thống tập trung vào việc thu thập dữ liệu chuẩn, tạo nhiễu và huấn luyện mô hình Transformer để sửa lỗi chính tả tiếng Việt.

## 👥 Thông tin dự án
* **Sinh viên thực hiện:** Phạm Quang Anh
* **Mã sinh viên:** B23DCCN044
* **Giảng viên hướng dẫn:** TS. Kim Ngọc Bách
* **Đơn vị:** Khoa Công nghệ thông tin 1 - PTIT

---

# File Tree: THUC-TAP-CO-SO


```text
THUC-TAP-CO-SO/
├── .idea/                      # Cấu hình IDE (PyCharm/WebStorm)
├── Documents/                  # Tài liệu báo cáo môn học
│   ├── FinalReport/            # Báo cáo cuối kỳ
│   ├── MidtermReport/           # Báo cáo giữa kỳ
│   └── WeeklyReports/          # Các báo cáo tiến độ hàng tuần
├── SourceCode/                 # Mã nguồn triển khai dự án
│   ├── data/                   # Thư mục chứa dữ liệu thô và dữ liệu xử lý
│   ├── notebooks/              # Jupyter Notebooks huấn luyện và thử nghiệm
│   │   ├── 01_train_model.ipynb
│   │   └── 02-output.ipynb
│   ├── src/                    # Mã nguồn ứng dụng giao diện chính
│   │   ├── .gradio/            # File tạm/Cấu hình của giao diện Gradio
│   │   └── app.py              # File chạy ứng dụng web giao diện (Gradio App)
│   └── visuals/                # Thư mục lưu trữ biểu đồ, hình ảnh trực quan
├── .dockerignore               # Các file loại trừ khi build Docker
├── .gitattributes              # Cấu hình thuộc tính Git (LFS, line endings)
├── .gitignore                  # Các file bỏ qua không commit lên GitHub
├── Dockerfile                  # Cấu hình đóng gói ứng dụng với Docker
├── README.md                   # Tài liệu hướng dẫn dự án
└── requirements.txt            # Danh sách các thư viện Python cần thiết
```


---

## Hướng dẫn cài đặt và chạy

### 1. Clone repository
```bash
git clone https://github.com/qap2102/THUC-TAP-CO-SO
cd THUC-TAP-CO-SO
```

### 2. Cài đặt thư viện
```bash
pip install -r SourceCode/requirements.txt
```

### 3. Chạy ứng dụng giao diện chính (Gradio Web App):

```bash
python SourceCode/src/app.py
```
