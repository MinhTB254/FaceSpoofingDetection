import os
import shutil
from prepare_dataset import process_casia_video

def main():
    # 1. DATASET PATH AUTOMATICALLY ALIGNED TO YOUR DIRECTORY
    raw_dataset_dir = "CASIA_faceAntisp" 
    output_dir = "Dataset/Processed"
    
    print("=== AUTOMATIC PREPROCESSING SYSTEM FOR CASIA-FASD ===")
    print(f"Raw dataset directory: {raw_dataset_dir}")
    print(f"Output folder: {output_dir}\n")
    
    if not os.path.exists(raw_dataset_dir):
        print(f"ERROR: Dataset directory not found at '{raw_dataset_dir}'.")
        return

    # Clear old processed dataset to avoid stale/incorrect depth maps mixing up
    if os.path.exists(output_dir):
        print("Clearing old incorrect processed files...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Scan release train and test directories of CASIA
    subsets = ["train_release", "test_release"]
    
    for subset in subsets:
        subset_path = os.path.join(raw_dataset_dir, subset)
        if not os.path.exists(subset_path):
            continue
            
        print(f"--- Processing partition: {subset.upper()} ---")
        
        # Scan subject directories (e.g., '1', '2', ..., '50')
        subjects = sorted(os.listdir(subset_path))
        for subj in subjects:
            subj_path = os.path.join(subset_path, subj)
            if not os.path.isdir(subj_path):
                continue
                
            print(f"  -> Processing Subject: {subj}")
            
            # Scan video files inside subject directory
            video_files = sorted(os.listdir(subj_path))
            for video_file in video_files:
                if not video_file.endswith(".avi"):
                    continue
                    
                video_path = os.path.join(subj_path, video_file)
                
                # CLASSIFICATION RULES FOR ATTACK TYPES:
                if video_file in ["1.avi", "2.avi", "HR_1.avi"]:
                    attack_type = "bonafide"
                elif video_file in ["3.avi", "4.avi", "HR_2.avi"]:
                    attack_type = "print"
                elif video_file in ["5.avi", "6.avi", "HR_3.avi"]:
                    attack_type = "cut"
                elif video_file in ["7.avi", "8.avi", "HR_4.avi"]:
                    attack_type = "replay"
                else:
                    attack_type = "print"
                
                # Extract frames and generate labels
                # frame_interval = 10 (Take 1 frame every 10 frames)
                process_casia_video(
                    video_path=video_path,
                    output_dir=output_dir,
                    attack_type=attack_type,
                    frame_interval=10
                )
                
    print("\n=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
    print(f"Normalized 256x256 face images saved at: {output_dir}/images/")
    print(f"32x32 depth maps (.npy) saved at: {output_dir}/depths/")
    print("Now you are ready to run `python train_depth.py` to start training!")

if __name__ == "__main__":
    main()


