# TimesNet Spatio - Phân tích & Dự báo Chuỗi Thời Gian Đa Biến

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

> 📌 **Code chính:** [`Code/TimesNet_Spatio.ipynb`](Code/TimesNet_Spatio.ipynb) — Notebook đầy đủ code, giải thích và kết quả thực nghiệm.

## 1. Tổng quan Dự án
Dự án triển khai mô hình **TimesNet** kết hợp module **Spatio Attention** nâng cao, giải quyết các hạn chế của mô hình gốc trong việc nắm bắt tương quan đa biến và thời gian.

**Điểm mới trong triển khai (Key Contributions):**
1.  **Cơ chế Dataset Động (Dynamic Dataset):** Tự động thích ứng với mọi tập dữ liệu CSV thông qua thuật toán tự động nhận diện đặc trưng (Feature Auto-detection) và chia tập dữ liệu theo tỷ lệ linh hoạt.
2.  **Spatio-Temporal Attention:** Tích hợp bộ ba cơ chế Attention: Channel, Cross-Variable và Gated Temporal.
3.  **Hệ thống Huấn luyện Tối ưu:** Tích hợp `Early Stopping` chống Overfitting và cơ chế `Model Checkpointing` lưu trữ phiên bản mô hình tối ưu nhất.

## 2. Cấu trúc Dự án

```
ATS_Nhom13_Final/
│
├── Code/                   # Mã nguồn chính
│   ├── main.py             # Script thực thi chính
│   ├── TimesNet_Spatio.ipynb # Notebook thực nghiệm (Jupyter)
│   ├── src/                # Module lõi
│       ├── model.py        # Kiến trúc TimesNetSpatio
│       ├── dataset.py      # Xử lý dữ liệu (Dataset_Custom)
│       └── utils.py        # Tiện ích huấn luyện (EarlyStopping...)
│
├── Data/                   # Kho dữ liệu
│   ├── ETTh1.csv           # Dữ liệu chuẩn
│   ├── weather/            # Dữ liệu thời tiết
│   ├── exchange_rate/      # Dữ liệu tỷ giá
│   └── ... 
```

## 3. Cài đặt Môi trường
Yêu cầu Python 3.8+ và các thư viện trong `requirements.txt`.

Kích hoạt môi trường ảo:
```powershell
..\Time-Series-Library\venv\Scripts\activate
cd ATS_Nhom13_Final\Code
```

## 4. Hướng dẫn Thực thi (Usage)

Hệ thống hỗ trợ tham số dòng lệnh (CLI) để tùy biến quá trình huấn luyện.

### 4.1. Chạy với dataset mặc định (ETTh1)
```bash
python main.py
```

### 4.2. Chạy với dataset tùy chỉnh (Ví dụ: Weather)
Hệ thống tự động phát hiện số lượng đặc trưng (`enc_in`, `c_out`) từ file dữ liệu.
```bash
python main.py --root_path "../Data/weather/" --data_path "weather.csv"
```

### 4.3. Chạy trên Google Colab
1.  Truy cập [Google Colab](https://colab.research.google.com/).
2.  Upload file `TimesNet_Spatio.ipynb` (trong thư mục `Code/`).
3.  Upload file dataset (ví dụ `weather.csv`) vào mục **Files** (biểu tượng thư mục bên trái).
4.  Tìm cell cấu hình `class Config` trong notebook và sửa đường dẫn:
    ```python
    class Config:
        root_path = './'            # Thư mục hiện tại trên Colab
        data_path = 'weather.csv'   # Tên file dataset bạn vừa upload
    ```
5.  Vào menu **Runtime** > **Change runtime type** > Chọn **T4 GPU** để chạy nhanh hơn.
6.  Bấm **Run All** để chạy toàn bộ code.

### 4.4. Tùy chỉnh tham số huấn luyện
```bash
python main.py --train_epochs 20 --batch_size 16 --learning_rate 0.0005
```

## 5. Các Tính năng Nâng cao

*   **Dynamic Split:** Dữ liệu được chia tự động theo tỷ lệ **70% Train - 10% Validation - 20% Test**, đảm bảo tính tổng quát hóa trên các dataset có độ dài khác nhau.
*   **Feature Auto-detection:** Tự động phân tích header của file CSV để xác định kích thước đầu vào/đầu ra cho mô hình.
*   **Early Stopping:** Tự động dừng huấn luyện khi `Validation Loss` không cải thiện sau số epoch quy định (`patience`), tối ưu hóa thời gian và tài nguyên.
*   **Model Checkpointing:** Tự động lưu trọng số của mô hình tốt nhất vào thư mục `checkpoints/`.

---
**Chúc bạn thực nghiệm thành công!** 
*Nhóm 13*
