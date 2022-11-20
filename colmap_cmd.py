import os
import argparse

data_dir = "data/yangang_Recon/bottle"

print("———————特征提取———————")
cmd1 = f"colmap feature_extractor \
    --database_path {data_dir}/database.db \
    --image_path {data_dir}/images/ \
    --ImageReader.camera_model PINHOLE \
    --SiftExtraction.gpu_index 0"
os.system(cmd1)

print("———————特征匹配———————")
cmd2 = f"colmap exhaustive_matcher \
    --database_path {data_dir}/database.db \
    --SiftMatching.gpu_index 0"
os.system(cmd2)

print("———————位姿求解———————")
os.makedirs(f"{data_dir}/sparse/0", exist_ok=True)
cmd3=f"colmap mapper \
    --database_path {data_dir}/database.db \
    --image_path {data_dir}/images \
    --output_path {data_dir}/sparse "
os.system(cmd3)
