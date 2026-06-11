import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình siêu tham số
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. Định nghĩa Dataset Loader cho PyTorch
# =====================================================================
class RandomGaussianNoise(object):
    def __init__(self, mean=0., std=0.02, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p
    def __call__(self, tensor):
        if random.random() < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

# =====================================================================
class FASDepthDataset(Dataset):
    """
    Dataset loader tải ảnh khuôn mặt 256x256 và bản đồ độ sâu 32x32 tương ứng.
    Tự động chia tách tập Train, Val, Test dựa trên Subject ID từ tên tệp tin để tránh rò rỉ.
    """
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        all_image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.jpg")))
        self.image_paths = []
        
        for path in all_image_paths:
            filename = os.path.basename(path)
            parts = filename.split("_")
            if len(parts) < 3:
                continue
                
            subset_type = parts[0] # 'train' hoặc 'test'
            try:
                subj_id = int(parts[2])
            except ValueError:
                continue
                
            if split == "train":
                # Tập Train chỉ lấy từ train_release (Subject 1 đến 20)
                if subset_type == "train":
                    self.image_paths.append(path)
            elif split == "val":
                # Tập Val lấy từ test_release (Subject 1 đến 15)
                if subset_type == "test" and 1 <= subj_id <= 15:
                    self.image_paths.append(path)
            elif split == "test":
                # Tập Test lấy từ test_release (Subject 16 đến 30)
                if subset_type == "test" and 16 <= subj_id <= 30:
                    self.image_paths.append(path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Đọc ảnh khuôn mặt
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # 2. Tìm file npy bản đồ độ sâu tương ứng
        file_id = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(self.data_dir, "depths", f"{file_id}.npy")
        depth_map = np.load(depth_path) # Kích thước 32x32
        
        # Đưa depth_map về tensor PyTorch dạng (1, 32, 32)
        depth_tensor = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
            
        # Áp dụng Cải tiến số 2: Lật ngang ngẫu nhiên (Random Horizontal Flip) đồng thời cả ảnh và depth map
        if self.transform is not None:
            if random.random() > 0.5:
                image = TF.hflip(image)
                depth_tensor = TF.hflip(depth_tensor)
            
        return image, depth_tensor

# =====================================================================
# 2. Định nghĩa Mô hình ước lượng Depth Map (MobileNetV2 làm Backbone + U-Net Feature Fusion)
# =====================================================================
class DepthFASModel(nn.Module):
    """
    Mô hình CNN nhận đầu vào (3, 256, 256) và hồi quy ra bản đồ độ sâu (1, 32, 32).
    Sử dụng MobileNetV2 pretrained làm backbone kết hợp kiến trúc U-Net (Multi-scale Feature Fusion)
    để dung hợp đặc trưng đa độ phân giải, giúp khôi phục bản đồ độ sâu sắc nét hơn.
    """
    def __init__(self):
        super(DepthFASModel, self).__init__()
        # Load MobileNetV2 pretrained
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Chia nhỏ các nhóm tầng của MobileNetV2
        self.stage1 = mobilenet.features[:7]   # Đầu ra: (B, 32, 32, 32)
        self.stage2 = mobilenet.features[7:14]  # Đầu ra: (B, 96, 16, 16)
        self.stage3 = mobilenet.features[14:]  # Đầu ra: (B, 1280, 8, 8)
        
        # Khối nén kênh của stage3
        self.conv_latent = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Bộ giải mã (Decoder) tích hợp Multi-scale Fusion
        # Bước 1: Upsample từ 8x8 lên 16x16
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Dung hợp với stage2 (đầu ra stage2 có 96 kênh). Tổng kênh: 256 + 96 = 352
        self.merge1 = nn.Sequential(
            nn.Conv2d(352, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Bước 2: Upsample từ 16x16 lên 32x32
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Dung hợp với stage1 (đầu ra stage1 có 32 kênh). Tổng kênh: 128 + 32 = 160
        self.merge2 = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Khối tích chập cuối cùng để đưa về 1 kênh bản đồ độ sâu
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Trích xuất đặc trưng đa quy mô
        feat_32x32 = self.stage1(x)            # (B, 32, 32, 32)
        feat_16x16 = self.stage2(feat_32x32)    # (B, 96, 16, 16)
        feat_8x8 = self.stage3(feat_16x16)      # (B, 1280, 8, 8)
        
        # Nén kênh từ 1280 xuống 512
        latent = self.conv_latent(feat_8x8)     # (B, 512, 8, 8)
        
        # Upsample + Dung hợp mức 1 (16x16)
        up_16x16 = self.up1(latent)             # (B, 256, 16, 16)
        concat1 = torch.cat([up_16x16, feat_16x16], dim=1) # (B, 352, 16, 16)
        merged1 = self.merge1(concat1)          # (B, 256, 16, 16)
        
        # Upsample + Dung hợp mức 2 (32x32)
        up_32x32 = self.up2(merged1)            # (B, 128, 32, 32)
        concat2 = torch.cat([up_32x32, feat_32x32], dim=1) # (B, 160, 32, 32)
        merged2 = self.merge2(concat2)          # (B, 64, 32, 32)
        
        # Bản đồ độ sâu cuối cùng
        depth = self.final_conv(merged2)        # (B, 1, 32, 32)
        return depth

# =====================================================================
# 3. Tiến trình Huấn Luyện (Training Pipeline)
# =====================================================================
def main():
    print(f"Đang chạy huấn luyện trên thiết bị: {DEVICE}")
    
    # 1. Khởi tạo Transforms riêng biệt cho Train và Val
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Thay đổi độ sáng ngẫu nhiên
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.4), # 40% cơ hội ảnh bị làm mờ ngẫu nhiên khi huấn luyện
        transforms.RandomAdjustSharpness(sharpness_factor=0.2, p=0.5), # 50% cơ hội giảm độ sắc nét (chống màn hình nét)
        transforms.ToTensor(),
        RandomGaussianNoise(std=0.02, p=0.3), # 30% cơ hội thêm nhiễu hạt Gaussian ngẫu nhiên
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Khởi tạo Dataset & Dataloader
    data_dir = "Dataset/Processed" # Thư mục chứa dữ liệu đầu ra của prepare_dataset.py
    if not os.path.exists(os.path.join(data_dir, "images")):
        print(f"Lỗi: Không tìm thấy thư mục ảnh tại {data_dir}/images. Vui lòng tiền xử lý trước!")
        return
        
    # Tạo các dataset riêng biệt dựa trên Subject ID để tránh trùng lặp thông tin người
    train_dataset = FASDepthDataset(data_dir, split="train", transform=train_transform)
    val_dataset = FASDepthDataset(data_dir, split="val", transform=val_transform)
    
    # Ở Windows sử dụng num_workers=0 để tránh lỗi Multiprocessing
    # Thêm pin_memory=True khi sử dụng CUDA để chuyển dữ liệu từ RAM lên GPU nhanh hơn
    use_pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=use_pin)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=use_pin)
    
    print(f"Tổng số ảnh huấn luyện: {len(train_dataset)} | Số ảnh đánh giá: {len(val_dataset)}")
    
    # 3. Khởi tạo Model, Loss, Optimizer và AMP GradScaler
    model = DepthFASModel().to(DEVICE)
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Khởi tạo bộ quản lý tăng tốc AMP (Mixed Precision) trên GPU
    scaler = torch.cuda.amp.GradScaler(enabled=use_pin)
    
    best_val_loss = float("inf")
    
    # 4. Vòng lặp huấn luyện chính (Epochs)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, depths in train_loader:
            images, depths = images.to(DEVICE), depths.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Sử dụng Autocast để tự động tính toán với độ chính xác hỗn hợp (float16 & float32)
            with torch.cuda.amp.autocast(enabled=use_pin):
                outputs = model(images)
                loss = criterion(outputs, depths)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_dataset)
        
        # Giai đoạn Đánh giá (Validation)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, depths in val_loader:
                images, depths = images.to(DEVICE), depths.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=use_pin):
                    outputs = model(images)
                    loss = criterion(outputs, depths)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_dataset)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Lưu trữ mô hình có Val Loss thấp nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/best_depth_fas.pth")
            print("--> Đã lưu mô hình tối ưu nhất tại: weights/best_depth_fas.pth")

if __name__ == "__main__":
    main()
