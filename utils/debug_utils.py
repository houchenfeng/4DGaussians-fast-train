import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def save_debug_image(rendered_image, gt_image, stage, iteration, cam, output_dir, maxtime):
    """
    保存debug图像：将训练图像和GT图像拼接在一起
    
    Args:
        rendered_image: 渲染的图像 (torch.Tensor)
        gt_image: 真实图像 (torch.Tensor) 
        stage: 训练阶段 ("coarse" 或 "fine")
        iteration: 当前迭代次数
        cam: 相机对象，包含time和image_name属性
        output_dir: 输出目录
        maxtime: 最大时间值，用于计算帧数
    """
    # 创建debug目录
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # 将tensor转换为numpy数组并确保在[0,1]范围内
    if isinstance(rendered_image, torch.Tensor):
        rendered_np = torch.clamp(rendered_image.detach().cpu(), 0, 1).numpy()
    else:
        rendered_np = np.clip(rendered_image, 0, 1)
        
    if isinstance(gt_image, torch.Tensor):
        gt_np = torch.clamp(gt_image.detach().cpu(), 0, 1).numpy()
    else:
        gt_np = np.clip(gt_image, 0, 1)
    
    # 如果是CHW格式，转换为HWC格式
    if rendered_np.shape[0] == 3:
        rendered_np = rendered_np.transpose(1, 2, 0)
    if gt_np.shape[0] == 3:
        gt_np = gt_np.transpose(1, 2, 0)
    
    # 转换为0-255范围的uint8
    rendered_np = (rendered_np * 255).astype(np.uint8)
    gt_np = (gt_np * 255).astype(np.uint8)
    
    # 获取相机信息
    if hasattr(cam, 'time') and hasattr(cam, 'image_name'):
        time_val = cam.time
        image_name = cam.image_name
    else:
        # 兼容PanopticSports数据集格式
        time_val = cam.get('time', 0.0)
        image_name = cam.get('image_name', f'frame_{iteration}')

    # camera_id = int(idx / 300) + 1
    # camid = int(cam.uid / maxtime)+1 if maxtime > 0 else int(cam.uid / 300)+1
    camid = cam.camid
    frameid = cam.frameid
    # 计算帧数：将0-1的time值乘以maxtime转换为具体帧数
    # frame_number = int(time_val * maxtime) if maxtime > 0 else int(time_val * 300)

    # 生成文件名：{stage}_{iter}_{camid}_{int(cam.time*maxtime)}.png
    # filename = f"{stage}_{iteration}_cam{camid}_frame{frame_number}.png"
    filename = f"{stage}_{iteration}_cam{camid}_frame{frameid}.png"
    # 水平拼接图像
    combined_image = np.concatenate([rendered_np, gt_np], axis=1)
    
    # # 添加分隔线
    # separator = np.zeros((combined_image.shape[0], 5, 3), dtype=np.uint8)
    # separator.fill(255)  # 白色分隔线
    # combined_image = np.concatenate([rendered_np, separator, gt_np], axis=1)
    
    # 保存图像
    combined_pil = Image.fromarray(combined_image)
    filepath = os.path.join(debug_dir, filename)
    combined_pil.save(filepath)
    
    return filepath

def create_debug_summary(debug_dir, stage, iteration):
    """
    创建debug图像的摘要信息
    """
    summary_file = os.path.join(debug_dir, f"{stage}_debug_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Debug图像摘要 - {stage}阶段, 迭代 {iteration}\n")
        f.write("=" * 50 + "\n")
        f.write("文件名格式: {stage}_{iter}_{cam.image_name}_{frame_number}.png\n")
        f.write("图像布局: [渲染图像] | [真实图像]\n")
        f.write("帧数计算: int(cam.time * maxtime)\n")

