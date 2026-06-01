import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from train_depth import DepthFASModel, DEVICE

# Cấu hình ngưỡng quyết định và kích thước đầu vào
FACE_SIZE = 256
THRESHOLD_SCORE = 60   #ngưỡng tổng độ sâu để phân biệt (Cần tinh chỉnh dựa trên thực tế huấn luyện)

# Khởi tạo Transforms (Phải trùng khớp với lúc Train)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    print(f"Đang chạy kiểm thử trên thiết bị: {DEVICE}")
    
    # 1. Khởi tạo mô hình và nạp trọng số đã train
    model = DepthFASModel().to(DEVICE)
    model_path = "weights/best_depth_fas.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Đã nạp trọng số mô hình từ: {model_path}")
    else:
        print(f"CẢNH BÁO: Không tìm thấy file trọng số '{model_path}'. Chạy mô hình với trọng số ngẫu nhiên!")
        
    model.eval()
    
    # 2. Khởi tạo OpenCV Camera & Haar Cascade Face Detector
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("\n=== CHƯƠNG TRÌNH ĐÃ KHỞI ĐỘNG ===")
    print("Nhấn phím 'q' để thoát kiểm thử thời gian thực.\n")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        img_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Thêm biên offset
            offset_w = int(w * 0.1)
            offset_h = int(h * 0.1)
            x1 = max(0, x - offset_w)
            y1 = max(0, y - offset_h)
            x2 = min(frame.shape[1], x + w + offset_w)
            y2 = min(frame.shape[0], y + h + offset_h)
            
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
                
            # Tiền xử lý ảnh khuôn mặt để đưa vào mạng CNN
            face_resized = cv2.resize(face_img, (FACE_SIZE, FACE_SIZE))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
            input_tensor = test_transform(pil_img).unsqueeze(0).to(DEVICE)
            
            # 3. Dự đoán Bản đồ độ sâu bằng mô hình CNN
            with torch.no_grad():
                predicted_depth = model(input_tensor) # Kích thước (1, 1, 32, 32)
                
            # Đưa kết quả về mảng numpy (32x32)
            depth_map = predicted_depth.squeeze().cpu().numpy()
            
            # 4. Tính toán điểm số liveness
            # Điểm liveness bằng tổng tất cả các giá trị pixel trong bản đồ độ sâu dự đoán
            liveness_score = np.sum(depth_map)
            
            # Quyết định Thật hay Giả
            if liveness_score > THRESHOLD_SCORE:
                label = "REAL (LIVE)"
                color = (0, 255, 0) # Xanh lá
            else:
                label = "SPOOF (ATTACK)"
                color = (0, 0, 255) # Đỏ
                
            # 5. Hiển thị thông tin lên camera
            # Vẽ hộp bao quanh mặt
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 3)
            # Viết chữ thông số
            cv2.putText(img_display, f"{label} | Score: {int(liveness_score)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 6. Hiển thị bản đồ độ sâu dự đoán (Được phóng to lên 128x128 để nhìn rõ trực quan)
            depth_map_resized = cv2.resize(depth_map, (128, 128))
            depth_map_colored = cv2.applyColorMap((depth_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Ghép bản đồ độ sâu hiển thị vào góc trên bên phải khung hình camera
            img_display[10:138, 500:628] = depth_map_colored
            cv2.putText(img_display, "Depth Map", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            break # Chỉ xử lý khuôn mặt đầu tiên
            
        cv2.imshow("Anti-Spoofing Real-time Depth", img_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
