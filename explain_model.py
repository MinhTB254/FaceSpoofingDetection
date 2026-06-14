import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from train_depth import DepthFASModel, DEVICE

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình kích thước đầu vào
FACE_SIZE = 256

# Bộ biến đổi ảnh đầu vào (phải đồng bộ với lúc train)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class GradCAM:
    """
    Lớp hỗ trợ tính toán Grad-CAM cho mô hình hồi quy độ sâu.
    Mục tiêu: Tìm các điểm ảnh đầu vào tác động nhiều nhất đến tổng độ sâu dự đoán.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook lưu gradients và activations
        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def save_activation(module, input, output):
            self.activations = output
            
        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)
        
    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor) # Kích thước (1, 1, 32, 32)
        
        # Tính Loss là tổng các pixel trên bản đồ độ sâu dự đoán
        # Mục tiêu: Xem những vùng nào trên ảnh gốc làm tăng tổng giá trị độ sâu này lên
        loss = output.sum()
        loss.backward()
        
        # Lấy gradients và activations của layer đích
        gradients = self.gradients.cpu().data.numpy()[0]   # (C, H, W)
        activations = self.activations.cpu().data.numpy()[0] # (C, H, W)
        
        # Tính trọng số bằng cách lấy trung bình cộng gradient theo không gian (Global Average Pooling)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Tính tổ hợp tuyến tính các activations theo trọng số
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Áp dụng hàm ReLU (chỉ giữ lại các giá trị có tác động tích cực)
        cam = np.maximum(cam, 0)
        
        # Chuẩn hóa bản đồ nhiệt về dải [0, 1]
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        # Phóng to bản đồ nhiệt về kích thước ảnh gốc 256x256
        cam = cv2.resize(cam, (FACE_SIZE, FACE_SIZE))
        
        return cam, output.squeeze().cpu().data.numpy()

def run_occlusion_sensitivity(model, input_tensor, patch_size=32, stride=16):
    """
    Chạy thử nghiệm che khuất (Occlusion Test) trên ảnh đầu vào.
    Di chuyển một ô vuông xám qua ảnh, đo sự thay đổi của độ sâu trung bình.
    """
    model.eval()
    with torch.no_grad():
        baseline_output = model(input_tensor)
        baseline_score = baseline_output.mean().item()
        
    # Tạo bản đồ độ nhạy (sensitivity map)
    sensitivity_map = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.float32)
    counts = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.float32)
    
    # Chuyển tensor về numpy để tiện thao tác che khuất
    img_np = input_tensor.squeeze().cpu().numpy() # (3, 256, 256)
    
    # Lặp qua các tọa độ trên ảnh để che khuất
    for y in range(0, FACE_SIZE - patch_size + 1, stride):
        for x in range(0, FACE_SIZE - patch_size + 1, stride):
            # Tạo bản sao của ảnh
            occluded_img = img_np.copy()
            
            # Che khuất bằng giá trị xám trung bình (0.0 sau khi chuẩn hóa)
            occluded_img[:, y:y+patch_size, x:x+patch_size] = 0.0
            
            # Chuyển lại về tensor và đưa vào mô hình
            occluded_tensor = torch.tensor(occluded_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(occluded_tensor)
                score = output.mean().item()
                
            # Độ sụt giảm điểm số: điểm càng giảm nhiều chứng tỏ vùng bị che càng quan trọng
            diff = baseline_score - score
            
            sensitivity_map[y:y+patch_size, x:x+patch_size] += diff
            counts[y:y+patch_size, x:x+patch_size] += 1
            
    # Lấy trung bình cộng tại những chỗ bị đè lặp
    sensitivity_map = np.divide(sensitivity_map, counts, out=np.zeros_like(sensitivity_map), where=counts!=0)
    
    # Phân biệt Real và Spoof dựa trên baseline score (độ sâu trung bình dự đoán)
    # Ảnh Real có baseline_score > 0.25, ảnh Spoof < 0.01
    if baseline_score > 0.05:
        # Đối với ảnh Real, ta chuẩn hóa bản đồ độ nhạy về dải [0, 1]
        s_min, s_max = sensitivity_map.min(), sensitivity_map.max()
        diff_val = s_max - s_min
        if diff_val > 1e-8:
            sensitivity_map = (sensitivity_map - s_min) / diff_val
        else:
            sensitivity_map = np.zeros_like(sensitivity_map)
    else:
        # Đối với ảnh Spoof (độ sâu phẳng gần bằng 0), sự biến đổi chỉ là nhiễu số học
        # Ta triệt tiêu hoàn toàn để hiển thị màu xanh lam đồng nhất
        sensitivity_map = np.zeros_like(sensitivity_map)
        
    return sensitivity_map

def process_and_save(model, image_path, output_filename):
    """
    Tiền xử lý ảnh (tự động crop khuôn mặt bằng MediaPipe), 
    chạy Grad-CAM, Occlusion Test và lưu kết quả trực quan hóa.
    """
    import mediapipe as mp
    
    # Đọc ảnh gốc
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        print(f"Error: Could not read image at '{image_path}'")
        return
        
    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
    # 1. Tự động phát hiện khuôn mặt và crop sát khuôn mặt (giống tập Train)
    mp_face_detection = mp.solutions.face_detection
    face_cropped_rgb = None
    
    # model_selection=0 cho khoảng cách gần (webcam) để đồng nhất với test_realtime.py
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
        results = face_detector.process(orig_image_rgb)
        
        if results.detections:
            # Lấy khuôn mặt đầu tiên phát hiện được
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = orig_image.shape
            
            # Chuyển đổi sang tọa độ pixel thực tế
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            # Giới hạn
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)
            
            # Thêm biên offset 10% để giữ đúng tỉ lệ lúc train
            offset_w = int(w * 0.1)
            offset_h = int(h * 0.1)
            
            x1 = max(0, x - offset_w)
            y1 = max(0, y - offset_h)
            x2 = min(iw, x + w + offset_w)
            y2 = min(ih, y + h + offset_h)
            
            face_img = orig_image_rgb[y1:y2, x1:x2]
            if face_img.size > 0:
                face_cropped_rgb = cv2.resize(face_img, (FACE_SIZE, FACE_SIZE))
                print(f"Successfully detected and cropped face for: {os.path.basename(image_path)}")
                
        if face_cropped_rgb is None:
            # Fallback nếu không phát hiện được mặt
            print(f"Warning: Face detection failed for {os.path.basename(image_path)}. Using raw resize.")
            face_cropped_rgb = cv2.resize(orig_image_rgb, (FACE_SIZE, FACE_SIZE))
            
    pil_img = Image.fromarray(face_cropped_rgb)
    input_tensor = test_transform(pil_img).unsqueeze(0).to(DEVICE)
    
    # 2. Chạy Grad-CAM trên layer tích chập cuối cùng của Decoder (final_conv[0])
    target_layer = model.final_conv[0]
    grad_cam_extractor = GradCAM(model, target_layer)
    cam, predicted_depth = grad_cam_extractor.generate(input_tensor)
    
    # 3. Chạy Thử nghiệm che khuất (Occlusion Test)
    print(f"Running Occlusion Test for {os.path.basename(image_path)}...")
    occlusion_map = run_occlusion_sensitivity(model, input_tensor, patch_size=32, stride=16)
    
    # 4. Vẽ đồ thị so sánh kết quả
    plt.figure(figsize=(15, 10))
    
    # Ảnh gốc RGB sau khi crop
    plt.subplot(2, 2, 1)
    plt.imshow(face_cropped_rgb)
    plt.title(f"Input Face Image ({os.path.basename(image_path)})")
    plt.axis("off")
    
    # Bản đồ độ sâu ước lượng
    plt.subplot(2, 2, 2)
    plt.imshow(predicted_depth, cmap="jet", vmin=0, vmax=1)
    plt.title(f"Predicted Depth Map (Mean: {predicted_depth.mean():.4f}, Std: {predicted_depth.std():.4f})")
    plt.colorbar()
    plt.axis("off")
    
    # Grad-CAM Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    grad_cam_overlay = cv2.addWeighted(face_cropped_rgb, 0.6, heatmap_rgb, 0.4, 0)
    
    plt.subplot(2, 2, 3)
    plt.imshow(grad_cam_overlay)
    plt.title("Grad-CAM Overlay (Depth-activating regions)")
    plt.axis("off")
    
    # Bản đồ độ nhạy che khuất (Occlusion Map)
    heatmap_occ = cv2.applyColorMap(np.uint8(255 * occlusion_map), cv2.COLORMAP_JET)
    heatmap_occ_rgb = cv2.cvtColor(heatmap_occ, cv2.COLOR_BGR2RGB)
    occlusion_overlay = cv2.addWeighted(face_cropped_rgb, 0.6, heatmap_occ_rgb, 0.4, 0)
    
    plt.subplot(2, 2, 4)
    plt.imshow(occlusion_overlay)
    plt.title("Occlusion Map Overlay (Occlusion-sensitive regions)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150)
    plt.close() # Giải phóng RAM
    print(f">>> Completed! Visualization results saved to: {output_filename}")

def main():
    import argparse
    import glob
    import random
    
    parser = argparse.ArgumentParser(description="Chương trình trực quan hóa giải thích mô hình (XAI).")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn đến ảnh khuôn mặt đầu vào. Nếu bỏ trống sẽ tự chọn 1 Real & 1 Spoof.")
    parser.add_argument("--weights", type=str, default="weights/best_depth_fas.pth", help="Đường dẫn đến file trọng số.")
    args = parser.parse_args()
    
    # 1. Khởi tạo mô hình và nạp trọng số
    model = DepthFASModel().to(DEVICE)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        print(f"Loaded model weights from: {args.weights}")
    else:
        print(f"WARNING: Weights file '{args.weights}' not found. Running with random weights!")
    
    model.eval()
    
    # 2. Xử lý ảnh đầu vào
    if args.image is not None:
        # Chạy trên ảnh cụ thể do người dùng truyền vào
        if not os.path.exists(args.image):
            print(f"Error: Image not found at '{args.image}'")
            return
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_filename = f"explain_{base_name}.png"
        process_and_save(model, args.image, output_filename)
        print(f"Please check '{output_filename}' to get the visual evidence for your presentation.")
    else:
        # Chế độ tự động: Chọn 1 ảnh Real và 1 ảnh Spoof trong tập Test (Subject 36-50 từ test_release)
        images_dir = "Dataset/Processed/images"
        image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
        
        if not image_paths:
            print(f"Error: No images found in '{images_dir}'. Please run preprocessing first.")
            return
            
        real_paths = []
        spoof_paths = []
        for path in image_paths:
            filename = os.path.basename(path)
            parts = filename.split("_")
            if len(parts) < 8 or parts[2] != "test":
                continue
            try:
                subj_id = int(parts[4])
            except ValueError:
                continue
                
            # Chỉ chọn từ tập Test độc lập (Subject 16 đến 30)
            if 16 <= subj_id <= 30:
                if parts[0] == "real":
                    real_paths.append(path)
                elif parts[0] == "fake":
                    spoof_paths.append(path)
                
        # Xử lý và lưu ảnh thật (Real)
        if real_paths:
            selected_real = random.choice(real_paths)
            print(f"\n[AUTO-MODE] Selected Real image: {selected_real}")
            process_and_save(model, selected_real, "explain_real.png")
        else:
            print("Warning: No Real images found in the dataset folder.")
            
        # Xử lý và lưu ảnh giả (Spoof)
        if spoof_paths:
            selected_spoof = random.choice(spoof_paths)
            print(f"\n[AUTO-MODE] Selected Spoof image: {selected_spoof}")
            process_and_save(model, selected_spoof, "explain_spoof.png")
        else:
            print("Warning: No Spoof images found in the dataset folder.")
            
        print("\n>>> Auto-mode finished! Please check 'explain_real.png' and 'explain_spoof.png' in your project directory.")

if __name__ == "__main__":
    main()
