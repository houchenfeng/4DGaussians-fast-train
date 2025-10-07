from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
                
                # 对于 try 块，从 dataset 的 image_paths 中提取 camid 和 frameid
                if hasattr(self.dataset, 'image_paths') and index < len(self.dataset.image_paths):
                    from scene.dataset_readers import extract_camid_frameid_from_path
                    camid, frameid = extract_camid_frameid_from_path(self.dataset.image_paths[index])
                else:
                    # 如果无法获取路径，使用 uid 计算 camid 和 frameid
                    # 对于 MultipleView 数据集：camid = int(uid / 300) + 1, frameid = (uid % 300) + 1
                    camid = int(index / 300) + 1
                    frameid = (index % 300) + 1
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                mask = caminfo.mask
                # 从 caminfo 获取 camid 和 frameid
                camid = getattr(caminfo, 'camid', None)
                frameid = getattr(caminfo, 'frameid', None)
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask,camid=camid,frameid=frameid)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
