import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_curve, auc

# Thiết lập encoding UTF-8 để hiển thị tiếng Việt trên Terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import mô hình và Dataset từ train_depth
from train_depth import DepthFASModel, FASDepthDataset, DEVICE

def main():
    print("=====================================================================")
    print("   HỆ THỐNG ĐÁNH GIÁ CHỈ SỐ SINH TRẮC HỌC (ISO/IEC 30107-3)   ")
    print("=====================================================================")
    
    # 1. Khởi tạo mô hình và nạp trọng số tối ưu
    model = DepthFASModel().to(DEVICE)
    model_path = "weights/best_depth_fas.pth"
    
    if not os.path.exists(model_path):
        print(f"LỖI: Không tìm thấy tệp trọng số tại '{model_path}'.")
        print("Vui lòng huấn luyện mô hình bằng cách chạy `python train_depth.py` trước.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"[*] Đã nạp thành công trọng số mô hình từ: {model_path}")
    model.eval()
    
    # 2. Định nghĩa transforms kiểm thử (đồng bộ với train_depth.py)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Nạp tập Test độc lập (Subject 16 đến 30)
    print("[*] Đang nạp tập dữ liệu Test (Subjects 16-30)...")
    dataset = FASDepthDataset(data_dir="Dataset/Processed", split="test", transform=test_transform)
    
    if len(dataset) == 0:
        print("LỖI: Tập Test trống! Hãy đảm bảo đã chạy tiền xử lý thành công.")
        return
        
    print(f"--> Tổng số mẫu ảnh trong tập Test: {len(dataset)}")
    
    # Dùng DataLoader để chạy suy luận nhanh hơn
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    scores = []
    labels = []
    attack_types = []
    
    print("[*] Đang chạy suy luận trên tập Test để thu thập điểm số liveness...")
    with torch.no_grad():
        for idx, (images, _, batch_labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            preds = model(images)  # (B, 1, 32, 32)
            
            # Tính điểm liveness cho từng mẫu trong batch
            preds_np = preds.squeeze(1).cpu().numpy()  # (B, 32, 32)
            for i in range(len(preds_np)):
                depth_map = preds_np[i]
                depth_mean = np.mean(depth_map)
                depth_std = np.std(depth_map)
                
                # Công thức tính điểm số liveness kháng khoảng cách (Mean * Std * 400)
                score = (depth_mean * depth_std) * 400.0
                scores.append(score)
                labels.append(batch_labels[i].item())
                
                # Lấy attack_type từ đường dẫn file tương ứng
                dataset_idx = idx * 16 + i
                img_path = dataset.image_paths[dataset_idx]
                filename = os.path.basename(img_path)
                parts = filename.split("_")
                attack_type = parts[1] if len(parts) >= 2 else "unknown"
                attack_types.append(attack_type)
                
    scores = np.array(scores)
    labels = np.array(labels)
    attack_types = np.array(attack_types)
    
    # Tách riêng tập real (bonafide) và các tập attack (print, cut, replay)
    real_indices = (labels == 1.0)
    fake_indices = (labels == 0.0)
    
    real_scores = scores[real_indices]
    fake_scores = scores[fake_indices]
    fake_attacks = attack_types[fake_indices]
    
    print(f"    + Số lượng mẫu thật (Bona Fide): {len(real_scores)}")
    print(f"    + Số lượng mẫu giả mạo (Attack): {len(fake_scores)}")
    print(f"      - Warped Photo (Print): {np.sum(fake_attacks == 'print')}")
    print(f"      - Cut-eye Photo (Cut):   {np.sum(fake_attacks == 'cut')}")
    print(f"      - Video Replay (Replay): {np.sum(fake_attacks == 'replay')}")
    
    # 4. Quét ngưỡng quyết định để tìm FAR, FRR, EER
    # Tạo dải ngưỡng từ 0 đến max_score + 1 với bước nhảy nhỏ
    max_score = np.max(scores) if len(scores) > 0 else 30.0
    thresholds = np.linspace(0.0, max(30.0, max_score), 1000)
    
    far_list = []
    frr_list = []
    apcer_print_list = []
    apcer_cut_list = []
    apcer_replay_list = []
    
    min_diff = float('inf')
    optimal_threshold = 8.0
    eer = 0.0
    
    for t in thresholds:
        # FRR (BPCER): tỉ lệ bonafide bị phân loại nhầm thành spoof (score <= t)
        frr = np.sum(real_scores <= t) / len(real_scores) if len(real_scores) > 0 else 0.0
        
        # FAR (APCER overall): tỉ lệ spoof bị phân loại nhầm thành real (score > t)
        far = np.sum(fake_scores > t) / len(fake_scores) if len(fake_scores) > 0 else 0.0
        
        # APCER riêng biệt cho từng dạng tấn công
        print_fakes = fake_scores[fake_attacks == 'print']
        cut_fakes = fake_scores[fake_attacks == 'cut']
        replay_fakes = fake_scores[fake_attacks == 'replay']
        
        apcer_print = np.sum(print_fakes > t) / len(print_fakes) if len(print_fakes) > 0 else 0.0
        apcer_cut = np.sum(cut_fakes > t) / len(cut_fakes) if len(cut_fakes) > 0 else 0.0
        apcer_replay = np.sum(replay_fakes > t) / len(replay_fakes) if len(replay_fakes) > 0 else 0.0
        
        far_list.append(far)
        frr_list.append(frr)
        apcer_print_list.append(apcer_print)
        apcer_cut_list.append(apcer_cut)
        apcer_replay_list.append(apcer_replay)
        
        # Tìm EER (nơi FAR và FRR gần nhau nhất)
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0
            optimal_threshold = t
            
    # Tính toán các chỉ số tại ngưỡng tối ưu EER
    idx_opt = np.argmin(np.abs(thresholds - optimal_threshold))
    opt_far = far_list[idx_opt]
    opt_frr = frr_list[idx_opt]
    opt_apcer_print = apcer_print_list[idx_opt]
    opt_apcer_cut = apcer_cut_list[idx_opt]
    opt_apcer_replay = apcer_replay_list[idx_opt]
    
    # APCER theo chuẩn ISO/IEC 30107-3 là giá trị lớn nhất trong các dạng tấn công
    opt_apcer_iso = max(opt_apcer_print, opt_apcer_cut, opt_apcer_replay)
    opt_bpcer_iso = opt_frr # BPCER chính là FRR
    opt_acer_iso = (opt_apcer_iso + opt_bpcer_iso) / 2.0
    
    # Tính ROC AUC
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    print("\n=====================================================================")
    print("   KẾT QUẢ ĐÁNH GIÁ CHỈ SỐ SINH TRẮC HỌC TRÊN TẬP TEST ĐỘC LẬP   ")
    print("=====================================================================")
    print(f"[-] Ngưỡng tối ưu EER (Optimal Threshold): {optimal_threshold:.4f}")
    print(f"[-] Tỉ lệ lỗi bằng nhau (EER):           {eer * 100:.2f}%")
    print(f"[-] Diện tích dưới đường cong (ROC AUC):   {roc_auc:.4f}")
    print(f"[-] Chỉ số tại Ngưỡng tối ưu:")
    print(f"    + FAR (Tỉ lệ nhận sai chung):         {opt_far * 100:.2f}%")
    print(f"    + FRR (Tỉ lệ từ chối sai chung):      {opt_frr * 100:.2f}%")
    print(f"[-] Chỉ số theo chuẩn ISO/IEC 30107-3:")
    print(f"    + APCER (Print - Warped Photo):       {opt_apcer_print * 100:.2f}%")
    print(f"    + APCER (Cut - Cut-eye Photo):        {opt_apcer_cut * 100:.2f}%")
    print(f"    + APCER (Replay - Video Screen):      {opt_apcer_replay * 100:.2f}%")
    print(f"    + APCER (ISO - Max APCER):            {opt_apcer_iso * 100:.2f}%")
    print(f"    + BPCER (ISO - Bona Fide Error):      {opt_bpcer_iso * 100:.2f}%")
    print(f"    + ACER (Average Classification Error): {opt_acer_iso * 100:.2f}%")
    print("=====================================================================")
    
    # 5. Vẽ đồ thị trực quan cao cấp (Vibrant Dark Mode Theme)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    
    # Tạo figure lớn với 2 đồ thị
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0f172a')
    
    # -------------------------------------------------------------
    # Đồ thị 1: ROC Curve
    # -------------------------------------------------------------
    ax1.set_facecolor('#1e293b')
    ax1.plot(fpr, tpr, color='#06b6d4', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='#64748b', linestyle='--', linewidth=1.5)
    
    # Vẽ điểm EER trên ROC Curve
    ax1.scatter([eer], [1 - eer], color='#f43f5e', s=100, zorder=5, 
                label=f'Điểm EER ({eer*100:.1f}%)')
    
    ax1.set_title("Đường cong ROC (Receiver Operating Characteristic)", color='#f8fafc', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("Tỉ lệ nhận sai (False Positive Rate / FAR)", color='#94a3b8', fontsize=11, labelpad=10)
    ax1.set_ylabel("Tỉ lệ nhận đúng (True Positive Rate / 1-FRR)", color='#94a3b8', fontsize=11, labelpad=10)
    ax1.tick_params(colors='#94a3b8')
    ax1.grid(color='#334155', linestyle=':', linewidth=1)
    ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    # -------------------------------------------------------------
    # Đồ thị 2: FAR / FRR vs Threshold
    # -------------------------------------------------------------
    ax2.set_facecolor('#1e293b')
    ax2.plot(thresholds, far_list, color='#f43f5e', linewidth=2.5, label='FAR (APCER - Tỉ lệ nhận sai)')
    ax2.plot(thresholds, frr_list, color='#10b981', linewidth=2.5, label='FRR (BPCER - Tỉ lệ từ chối sai)')
    
    # Vẽ các đường APCER chi tiết mờ hơn
    ax2.plot(thresholds, apcer_print_list, color='#fb923c', linestyle=':', alpha=0.7, label='APCER - Print')
    ax2.plot(thresholds, apcer_cut_list, color='#a855f7', linestyle=':', alpha=0.7, label='APCER - Cut')
    ax2.plot(thresholds, apcer_replay_list, color='#38bdf8', linestyle=':', alpha=0.7, label='APCER - Replay')
    
    # Vẽ điểm giao nhau EER
    ax2.axvline(x=optimal_threshold, color='#e2e8f0', linestyle='--', alpha=0.5)
    ax2.axhline(y=eer, color='#e2e8f0', linestyle='--', alpha=0.5)
    ax2.scatter([optimal_threshold], [eer], color='#f59e0b', s=100, zorder=5, 
                label=f'EER: {eer*100:.2f}% tại T={optimal_threshold:.2f}')
    
    ax2.set_title("Biến thiên FAR và FRR theo Ngưỡng quyết định", color='#f8fafc', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel("Ngưỡng điểm số Liveness (Threshold)", color='#94a3b8', fontsize=11, labelpad=10)
    ax2.set_ylabel("Tỉ lệ lỗi (Error Rate)", color='#94a3b8', fontsize=11, labelpad=10)
    ax2.tick_params(colors='#94a3b8')
    ax2.grid(color='#334155', linestyle=':', linewidth=1)
    ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#f8fafc', loc='upper right')
    ax2.set_xlim([0.0, max(30.0, max_score)])
    ax2.set_ylim([-0.02, 1.02])
    
    plt.suptitle("BÁO CÁO ĐÁNH GIÁ CHỈ SỐ SINH TRẮC HỌC (CASIA-FASD TEST SPLIT)", 
                 color='#f8fafc', fontsize=16, fontweight='bold', y=0.98)
    
    # Lưu biểu đồ sắc nét
    output_img = "evaluation_results.png"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_img, dpi=200, facecolor='#0f172a')
    plt.close()
    
    print(f"\n[+] Đồ thị đánh giá trực quan đã được lưu vào: {output_img}")
    print("=====================================================================")

if __name__ == "__main__":
    main()
