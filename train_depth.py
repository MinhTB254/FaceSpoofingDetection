import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Cấu hình siêu tham số
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. Định nghĩa Dataset Loader cho PyTorch
# =====================================================================
class FASDepthDataset(Dataset):
    """
    Dataset loader tải ảnh khuôn mặt 256x256 và bản đồ độ sâu 32x32 tương ứng.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.jpg")))
        self.transform = transform

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
            
        return image, depth_tensor

# =====================================================================
# 2. Định nghĩa Mô hình ước lượng Depth Map (MobileNetV2 làm Backbone)
# =====================================================================
class DepthFASModel(nn.Module):
    """
    Mô hình CNN nhận đầu vào (3, 256, 256) và hồi quy ra bản đồ độ sâu (1, 32, 32).
    Sử dụng MobileNetV2 pre-trained để học kết cấu bề mặt cực mạnh và nhẹ.
    """
    def __init__(self):
        super(DepthFASModel, self).__init__()
        # Load MobileNetV2 pretrained
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Giữ lại phần trích xuất đặc trưng (Features Extractor)
        self.features = mobilenet.features
        
        # Nhánh hồi quy bản đồ độ sâu (Depth Regression Head)
        # MobileNetV2 feature maps có số kênh là 1280 ở tầng cuối cùng trước classifier.
        # Đầu vào của tầng này là feature map kích thước (1280, 8, 8)
        # Ta dùng Transposed Convolution để Upsample từ 8x8 lên 32x32
        self.depth_head = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Upsample lần 1: 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Upsample lần 2: 16x16 -> 32x32
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Khối tích chập cuối cùng để đưa về 1 kênh độ sâu (Depth Map)
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Giới hạn giá trị đầu ra trong khoảng [0, 1] chuẩn hóa
        )

    def forward(self, x):
        # Trích xuất đặc trưng qua MobileNetV2: (B, 3, 256, 256) -> (B, 1280, 8, 8)
        x = self.features(x)
        # Hồi quy ra bản đồ độ sâu: (B, 1280, 8, 8) -> (B, 1, 32, 32)
        depth = self.depth_head(x)
        return depth

# =====================================================================
# 3. Tiến trình Huấn Luyện (Training Pipeline)
# =====================================================================
def main():
    print(f"Đang chạy huấn luyện trên thiết bị: {DEVICE}")
    
    # 1. Khởi tạo Transforms cho Ảnh khuôn mặt
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Thay đổi độ sáng ngẫu nhiên
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Khởi tạo Dataset & Dataloader
    data_dir = "Dataset/Processed" # Thư mục chứa dữ liệu đầu ra của prepare_dataset.py
    if not os.path.exists(os.path.join(data_dir, "images")):
        print(f"Lỗi: Không tìm thấy thư mục ảnh tại {data_dir}/images. Vui lòng tiền xử lý trước!")
        return
        
    dataset = FASDepthDataset(data_dir, transform=train_transform)
    
    # Chia Train/Val tỷ lệ 85/15
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Tổng số ảnh huấn luyện: {len(train_dataset)} | Số ảnh đánh giá: {len(val_dataset)}")
    
    # 3. Khởi tạo Model, Loss và Optimizer
    model = DepthFASModel().to(DEVICE)
    criterion = nn.MSELoss() # Đo sai số bình phương trung bình giữa bản đồ độ sâu dự đoán và nhãn Ground Truth
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    best_val_loss = float("inf")
    
    # 4. Vòng lặp huấn luyện chính (Epochs)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, depths in train_loader:
            images, depths = images.to(DEVICE), depths.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_dataset)
        
        # Giai đoạn Đánh giá (Validation)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, depths in val_loader:
                images, depths = images.to(DEVICE), depths.to(DEVICE)
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
