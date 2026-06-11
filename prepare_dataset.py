import os
import cv2
import numpy as np
import mediapipe as mp

# Cấu hình các thông số kích thước
FACE_SIZE = 256        # Ảnh khuôn mặt cắt ra để đưa vào mạng CNN
DEPTH_SIZE = 32        # Bản đồ độ sâu 32x32 để làm nhãn giám sát bổ trợ

# Khởi tạo MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def generate_depth_map(image, is_real=True):
    """
    Trích xuất bản đồ độ sâu 32x32 từ ảnh khuôn mặt 2D bằng MediaPipe.
    - is_real=True: Trích xuất độ sâu lồi lõm thực tế và chuẩn hóa về [0, 1].
    - is_real=False: Trả về bản đồ phẳng bằng 0 hoàn toàn (Spoof face).
    """
    if not is_real:
        # Đối với ảnh giả mạo, nhãn độ sâu là phẳng lì toàn số 0
        return np.zeros((DEPTH_SIZE, DEPTH_SIZE), dtype=np.float32)
        
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    # Tạo bản đồ độ sâu mặc định bằng 0 nếu không phát hiện được khuôn mặt
    depth_map = np.zeros((DEPTH_SIZE, DEPTH_SIZE), dtype=np.float32)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # 1. Trích xuất tọa độ X, Y, Z của các landmarks
        coords = []
        for lm in face_landmarks.landmark:
            # lm.z là tọa độ chiều sâu ước lượng bởi MediaPipe (nhỏ hơn là gần camera hơn)
            coords.append([lm.x * w, lm.y * h, lm.z])
            
        coords = np.array(coords)
        
        # 2. Chuẩn hóa chiều sâu Z về khoảng [0, 1]
        # Trong MediaPipe, Z âm có nghĩa là gần camera hơn (mũi nhô ra).
        # Ta đảo ngược Z sao cho phần gần camera nhất (mũi) có giá trị cao nhất (1.0), phần xa nhất có giá trị (0.0)
        z_values = coords[:, 2]
        z_min = np.min(z_values)
        z_max = np.max(z_values)
        
        if z_max != z_min:
            # Đảo ngược và chuẩn hóa về [0, 1]
            normalized_z = 1.0 - (z_values - z_min) / (z_max - z_min)
        else:
            normalized_z = np.zeros_like(z_values)
            
        # 3. Phóng chiếu các landmarks lên lưới 2D kích thước 32x32 để tạo Depth Map
        # Định vị tọa độ X, Y chuẩn hóa trên lưới 32x32
        x_grid = np.clip(np.floor(coords[:, 0] / w * (DEPTH_SIZE - 1)).astype(int), 0, DEPTH_SIZE - 1)
        y_grid = np.clip(np.floor(coords[:, 1] / h * (DEPTH_SIZE - 1)).astype(int), 0, DEPTH_SIZE - 1)
        
        # Điền các giá trị độ sâu vào bản đồ 2D
        for idx in range(len(normalized_z)):
            depth_map[y_grid[idx], x_grid[idx]] = normalized_z[idx]
            
        # Áp dụng bộ lọc Gaussian Blur nhẹ để làm mịn các khoảng trống giữa các điểm landmarks
        depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0)
        
        # Chuẩn hóa lại lần cuối về [0, 1] sau khi lọc blur
        dp_min, dp_max = depth_map.min(), depth_map.max()
        if dp_max != dp_min:
            depth_map = (depth_map - dp_min) / (dp_max - dp_min)
            
    return depth_map

def process_casia_video(video_path, output_dir, is_real=True, frame_interval=10):
    """
    Đọc một video từ CASIA-FASD, cắt khuôn mặt và sinh nhãn Depth Map 32x32 cho các frames.
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.path.join(output_dir, "depths")
    os.makedirs(os.path.join(output_dir, "depths"), exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    # Lấy thông tin subset và subject ID để đặt tên file tránh ghi đè
    normalized_path = os.path.normpath(video_path)
    path_parts = normalized_path.split(os.sep)
    subset = path_parts[-3] if len(path_parts) >= 3 else "subset"
    subj = path_parts[-2] if len(path_parts) >= 2 else "subject"
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Khởi tạo bộ phát hiện khuôn mặt MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Chỉ lấy ảnh ở các khoảng interval để tránh trùng lặp dữ liệu quá nhiều
        if frame_count % frame_interval == 0:
            # Chuyển ảnh sang RGB vì MediaPipe yêu cầu RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)
            
            if results.detections:
                for detection in results.detections:
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
                    
                    # Thêm biên offset xung quanh mặt giống như trong dataCollection
                    offset_w = int(w * 0.1)
                    offset_h = int(h * 0.1)
                    
                    x1 = max(0, x - offset_w)
                    y1 = max(0, y - offset_h)
                    x2 = min(iw, x + w + offset_w)
                    y2 = min(ih, y + h + offset_h)
                    
                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue
                        
                    # Resize khuôn mặt về kích thước chuẩn 256x256
                    face_img_resized = cv2.resize(face_img, (FACE_SIZE, FACE_SIZE))
                    
                    # Sinh Depth Map (32x32)
                    depth_map = generate_depth_map(face_img_resized, is_real=is_real)
                    
                    # NẾU LÀ NGƯỜI THẬT MÀ FACE MESH THẤT BẠI (toàn số 0) -> BỎ QUA KHUNG HÌNH NÀY
                    if is_real and depth_map.max() == 0:
                        continue
                        
                    # Lưu ảnh khuôn mặt và bản đồ độ sâu tương ứng
                    file_id = f"{subset}_{subj}_{video_name}_f{frame_count}_{saved_count}"
                    img_path = os.path.join(output_dir, "images", f"{file_id}.jpg")
                    depth_path = os.path.join(output_dir, "depths", f"{file_id}.npy")
                    
                    cv2.imwrite(img_path, face_img_resized)
                    np.save(depth_path, depth_map)
                    saved_count += 1
                    break # Chỉ lấy khuôn mặt đầu tiên phát hiện được
                    
        frame_count += 1
        
    cap.release()
    face_detector.close()
    print(f"Finished processing {video_path}: Extracted {saved_count} quality frames.")

# --- FOR TESTING ---
if __name__ == "__main__":
    print("Environment ready! You can import `process_casia_video` to preprocess CASIA-FASD.")
    print("Depth Labels: Real = Normalized 3D [0,1] | Spoof = Flat Zeros Map.")

