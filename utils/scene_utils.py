import torch
import os
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

import numpy as np

import copy
@torch.no_grad()
def render_training_image(scene, gaussians, viewpoints, render_func, pipe, background, stage, iteration, time_now, dataset_type):
    def render(gaussians, viewpoint, path, scaling, cam_type):
        # scaling_copy = gaussians._scaling
        render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=cam_type)
        label1 = f"stage:{stage},iter:{iteration}"
        times =  time_now/60
        if times < 1:
            end = "min"
        else:
            end = "mins"
        label2 = "time:%.2f" % times + end
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        if dataset_type == "PanopticSports":
            gt_np = viewpoint['image'].permute(1,2,0).cpu().numpy()
        else:
            gt_np = viewpoint.original_image.permute(1,2,0).cpu().numpy()
        image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        depth_np = depth.permute(1, 2, 0).cpu().numpy()
        depth_np /= depth_np.max()
        depth_np = np.repeat(depth_np, 3, axis=2)
        image_np = np.concatenate((gt_np, image_np, depth_np), axis=1)
        image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8'))  
        draw1 = ImageDraw.Draw(image_with_labels)
        font = ImageFont.truetype('./utils/TIMES.TTF', size=40) 
        text_color = (255, 0, 0)  
        label1_position = (10, 10)
        label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10) 
        draw1.text(label1_position, label1, fill=text_color, font=font)
        draw1.text(label2_position, label2, fill=text_color, font=font)
        
        image_with_labels.save(path)
    render_base_path = os.path.join(scene.model_path, f"{stage}_render")
    point_cloud_path = os.path.join(render_base_path,"pointclouds")
    image_path = os.path.join(render_base_path,"images")
    if not os.path.exists(os.path.join(scene.model_path, f"{stage}_render")):
        os.makedirs(render_base_path)
    if not os.path.exists(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    
    for idx in range(len(viewpoints)):
        image_save_path = os.path.join(image_path,f"{iteration}_{idx}.jpg")
        render(gaussians,viewpoints[idx],image_save_path,scaling = 1,cam_type=dataset_type)
    pc_mask = gaussians.get_opacity
    pc_mask = pc_mask > 0.1

@torch.no_grad()
def save_debug_image(scene, gaussians, viewpoint, render_func, pipe, background, stage, iteration, dataset_type):
    """保存debug图像，左右分别是GT和render"""
    # 创建debug文件夹
    debug_path = os.path.join(scene.model_path, "debug")
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)
    
    # 渲染图像
    render_pkg = render_func(viewpoint, gaussians, pipe, background, stage=stage, cam_type=dataset_type)
    render_image = render_pkg["render"]
    
    # 获取GT图像
    if dataset_type == "PanopticSports":
        gt_image = viewpoint['image']
    else:
        gt_image = viewpoint.original_image
    
    # 转换为numpy数组并调整维度
    render_np = render_image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    gt_np = gt_image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    
    # 左右拼接图像
    combined_image = np.concatenate((gt_np, render_np), axis=1)  # 水平拼接
    
    # 转换为PIL图像并保存
    combined_image_pil = Image.fromarray((np.clip(combined_image, 0, 1) * 255).astype('uint8'))
    
    # 生成文件名
    if hasattr(viewpoint, 'image_name'):
        image_name = viewpoint.image_name
    else:
        image_name = "unknown"
    
    if hasattr(viewpoint, 'time'):
        time_str = f"{viewpoint.time:.4f}"
    else:
        time_str = "0.0000"
    
    filename = f"{stage}_{iteration}_{image_name}_{time_str}.jpg"
    filepath = os.path.join(debug_path, filename)
    
    combined_image_pil.save(filepath)
    print(f"Debug image saved: {filepath}")

def visualize_and_save_point_cloud(point_cloud, R, T, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    R = R.T
    T = -R.dot(T)
    transformed_point_cloud = np.dot(R, point_cloud) + T.reshape(-1, 1)
    ax.scatter(transformed_point_cloud[0], transformed_point_cloud[1], transformed_point_cloud[2], c='g', marker='o')
    ax.axis("off")
    plt.savefig(filename)

