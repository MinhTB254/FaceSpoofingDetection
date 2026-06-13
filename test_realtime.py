import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from collections import deque
from train_depth import DepthFASModel, DEVICE

# Cấu hình ngưỡng quyết định và kích thước đầu vào
FACE_SIZE = 256
THRESHOLD_SCORE = 8.0  # Ngưỡng tối ưu dựa trên tích số Mean & Std để chống ảnh thẻ/nhiễu phản xạ

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
    
    # 2. Khởi tạo OpenCV Camera & MediaPipe Face Detector
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    # Khởi tạo cửa sổ trượt lưu trữ 20 điểm số gần nhất để lọc nhiễu
    score_window = deque(maxlen=20)
    
    print("\n=== CHƯƠNG TRÌNH ĐÃ KHỞI ĐỘNG ===")
    print("Nhấn phím 'q' để thoát kiểm thử thời gian thực.\n")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        img_display = frame.copy()
        
        # Chuyển ảnh sang RGB vì MediaPipe yêu cầu RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                # Lấy tọa độ bounding box tương đối (từ 0.0 đến 1.0)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                
                # Chuyển đổi sang tọa độ pixel thực tế
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Giới hạn để tránh cắt tràn viền ngoài ảnh
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                # Thêm biên offset
                offset_w = int(w * 0.1)
                offset_h = int(h * 0.1)
                x1 = max(0, x - offset_w)
                y1 = max(0, y - offset_h)
                x2 = min(iw, x + w + offset_w)
                y2 = min(ih, y + h + offset_h)
                
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
                
                # 4. Tính toán điểm số liveness kháng khoảng cách (Distance-Invariant)
                depth_mean = np.mean(depth_map)
                depth_std = np.std(depth_map)
                
                # Điểm số tích hợp nhân Mean & Std để triệt tiêu các ảnh phẳng có nhiễu đột biến
                raw_score = (depth_mean * depth_std) * 400.0
                
                # Thêm điểm số của khung hình hiện tại vào cửa sổ trượt
                score_window.append(raw_score)
                
                # Tính toán điểm số trung bình trượt đã được làm mịn (Smoothing)
                liveness_score = np.mean(score_window)
                
                # Quyết định Thật hay Giả dựa trên điểm số trung bình trượt
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
                cv2.putText(img_display, f"{label} | Score: {liveness_score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 6. Hiển thị bản đồ độ sâu dự đoán (Được phóng to để nhìn rõ trực quan)
                depth_map_resized = cv2.resize(depth_map, (64, 64))
                depth_map_colored = cv2.applyColorMap((depth_map_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                # Ghép bản đồ độ sâu hiển thị vào góc trên bên phải khung hình camera
                img_display[10:74, 500:564] = depth_map_colored
                cv2.putText(img_display, "Depth Map", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                break # Chỉ xử lý khuôn mặt đầu tiên
        else:
            # Nếu không phát hiện thấy khuôn mặt nào, xóa sạch cửa sổ trượt để tránh trễ điểm số cho người tiếp theo
            score_window.clear()
                
        cv2.imshow("Anti-Spoofing Real-time Depth", img_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if 'liveness_score' in locals():
                cv2.imwrite("debug_face.jpg", frame)
                print("\n[DEBUG] Đã lưu ảnh debug toàn khung hình vào: debug_face.jpg")
                print(f"[DEBUG] Thông số của khung hình vừa lưu:")
                print(f"  - Mean: {depth_mean:.4f}")
                print(f"  - Std: {depth_std:.4f}")
                print(f"  - Raw Score: {raw_score:.2f}")
                print(f"  - Liveness Score (đã mịn): {liveness_score:.2f}")
            else:
                print("\n[DEBUG] Không lưu được ảnh vì chưa phát hiện khuôn mặt nào!")
            
    cap.release()
    face_detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
