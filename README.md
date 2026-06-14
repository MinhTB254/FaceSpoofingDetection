# 3D Depth-Based Face Anti-Spoofing (FAS)

Dự án này triển khai hệ thống chống giả mạo khuôn mặt (Face Anti-Spoofing) dựa trên hồi quy bản đồ độ sâu 3D (3D Depth Regression). Hệ thống sử dụng mạng xương sống **MobileNetV2** kết hợp kiến trúc **U-Net Feature Fusion**, được tối ưu hóa bằng **Spatial Gradient Loss** và thuật toán **Scipy Griddata** để nhận biết cấu trúc 3D thực tế của khuôn mặt và triệt tiêu các cuộc tấn công dạng ảnh in phẳng (2D), màn hình kỹ thuật số hay mặt nạ.

---

## 📌 Các tính năng chính

1. **Nội suy độ sâu dày đặc (Dense Depth Map):** Sử dụng `scipy.interpolate.griddata` để dựng bản đồ độ sâu mịn màng và liên tục từ 468 điểm landmark 3D của MediaPipe Face Mesh.
2. **Spatial Gradient Loss:** Hàm mất mát phạt các độ dốc không gian sai lệch, ép mô hình học đúng cấu trúc địa hình 3D (mũi nhô cao, má dốc xuống) và triệt tiêu độ sâu trên mặt phẳng về 0.
3. **Validation Accuracy Tracking:** Tính toán trực quan độ chính xác phân loại (Acc %) trực tiếp sau mỗi Epoch huấn luyện dựa trên tập kiểm định độc lập không trùng lặp đối tượng (anti-subject-leakage).
4. **Real-time Webcam Test:** Chạy kiểm thử camera thời gian thực với công thức tính điểm tích số ổn định kháng nhiễu `Mean * Std * 400.0`, ngưỡng quyết định `8.0`.
5. **XAI Diagnostics (Grad-CAM & Occlusion):** Chương trình giải thích quyết định mô hình thông qua bản đồ nhiệt Grad-CAM và bản đồ độ nhạy che khuất (Occlusion Sensitivity).

---

## 🛠️ Cài đặt và Yêu cầu hệ thống

### 1. Cài đặt các thư viện cần thiết
Hệ thống yêu cầu các thư viện PyTorch, OpenCV, MediaPipe và Scipy. Cài đặt bằng lệnh:

```bash
pip install torch torchvision numpy opencv-python mediapipe scipy pillow matplotlib
```

### 2. Xác minh môi trường
Chạy file kiểm tra môi trường để đảm bảo CUDA (GPU) và các thư viện đã sẵn sàng:
```bash
python verify_requirements.py
```

---

## 📂 Hướng dẫn chạy Dự án (Quy trình 4 bước)

### Bước 1: Chuẩn bị Dữ liệu và Tiền xử lý
1. Giải nén bộ dữ liệu **CASIA-FASD** vào thư mục gốc của dự án với cấu trúc:
   ```text
   CASIA_faceAntisp/
   ├── train_release/
   │   ├── 1/ (chứa các file .avi)
   │   └── ... (đến 20/)
   └── test_release/
       ├── 1/
       └── ... (đến 30/)
   ```
2. Chạy tập lệnh để cắt khuôn mặt, chuẩn hóa chiều sâu hình học và sinh dữ liệu huấn luyện:
   ```bash
   python run_preprocessing.py
   ```
   *Dữ liệu đầu ra chất lượng cao sẽ được lưu tại `Dataset/Processed/` (gồm thư mục `images/` chứa ảnh khuôn mặt 256x256 có tiền tố nhãn `real_`/`fake_` và thư mục `depths/` chứa file nhãn `.npy` 32x32).*

### Bước 2: Huấn luyện Mô hình
Chạy lệnh huấn luyện mô hình với cấu hình tối ưu hóa tự động (Mixed Precision - AMP):
```bash
python train_depth.py
```
* Tiến trình sẽ chạy trong 15 Epochs. Ở cuối mỗi Epoch, chương trình sẽ in ra chỉ số **Val Loss** và **Val Acc** (Độ chính xác phân loại trên tập Validation độc lập).
* Trọng số mô hình tốt nhất (Val Loss thấp nhất) sẽ được tự động lưu tại: `weights/best_depth_fas.pth`.

### Bước 3: Kiểm thử thời gian thực (Webcam)
Khởi chạy camera để kiểm tra khả năng nhận diện Real/Spoof thời gian thực:
```bash
python test_realtime.py
```
* **Thao tác điều khiển:**
  * Nhấn **`q`** để thoát chương trình.
  * Nhấn **`s`** để chụp lại khuôn mặt hiện tại, lưu thành file ảnh **`debug_face.jpg`** và in chi tiết các thông số đo lường (`Mean`, `Std`, `Liveness Score`) ra Terminal.
* **Ngưỡng quyết định:** Mặc định là `8.0` (Mặt thật dao động từ `15.0 - 25.0`, ảnh in phẳng hoặc màn hình bị triệt tiêu điểm số về dưới `0.1`).

### Bước 4: Chẩn đoán và Giải thích Mô hình (XAI)
Để phân tích xem tại sao mô hình đưa ra quyết định REAL hoặc SPOOF trên một bức ảnh cụ thể (ví dụ: ảnh debug vừa chụp):
```bash
python explain_model.py --image debug_face.jpg
```
* Chương trình sẽ tạo ra ảnh trực quan hóa **`explain_debug_face.png`** chứa bản đồ độ sâu dự đoán, Grad-CAM (vùng kích hoạt chiều sâu) và Occlusion Map (độ nhạy che khuất).
* Nếu chạy không truyền tham số `--image`, chương trình sẽ tự động chọn ngẫu nhiên 1 ảnh Real và 1 ảnh Spoof trong tập Test độc lập để sinh ảnh so sánh `explain_real.png` và `explain_spoof.png`.

---

