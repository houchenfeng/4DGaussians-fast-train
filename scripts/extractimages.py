import os
import sys
import shutil

# Accept absolute or relative dataset directory path
input_path = sys.argv[1]
print(f"[extract] 输入目录: {input_path}")
if not os.path.isabs(input_path):
    input_path = os.path.abspath(input_path)
print(f"[extract] 绝对路径: {input_path}")

colmap_path = "./colmap_tmp"
images_path = os.path.join(colmap_path, "images")
os.makedirs(images_path, exist_ok=True)
print(f"[extract] 输出目录(首帧复制至): {images_path}")

image_index = 0

# The input directory should contain camera subfolders (e.g., cam01, cam02, ...)
dataset_dir = input_path
camera_folders = [d for d in sorted(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir, d))]
print(f"[extract] 发现相机文件夹数量: {len(camera_folders)}")
for camera_folder_name in camera_folders:
    camera_dir = os.path.join(dataset_dir, camera_folder_name)
    for file_name in sorted(os.listdir(camera_dir)):
        if file_name.startswith("frame_00001"):
            image_index += 1
            src_path = os.path.join(camera_dir, file_name)
            dst_path = os.path.join(images_path, f"image{image_index}.jpg")
            shutil.copyfile(src_path, dst_path)
            print(f"[extract] 复制 {src_path} -> {dst_path}")

print(f"[extract] 完成，复制首帧数量: {image_index}")
