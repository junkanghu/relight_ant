import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import pickle
from . import get_dir_video

permute = lambda x: x.permute(*torch.arange(x.ndim-1, -1, -1))

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
normalize = lambda x: x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-5)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, all_image_dir, all_mask_dir, all_corre):
        self.opt = opt
        self.image_dir = all_image_dir
        self.mask_dir = all_mask_dir
        self.corre_dir = all_corre
        self.num = len(self.image_dir)
        
        self.transform_img = transforms.ToTensor()
        self.transform_scale = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_dirs = self.image_dir[idx % self.num]
        mask_dirs = self.mask_dir[idx % self.num]
        corre_dir = self.corre_dir[idx % self.num]
        imgs = self.load_sequence(img_dirs)
        masks = self.load_sequence(mask_dirs, ismask=True)
        corre = self.load_correspondences(corre_dir)
        
        return_dict = {
            'input': imgs,
            'mask': masks,
            'corre': corre,
            'name': str(idx)
        }
        return return_dict

    def load_light(self, path):
        with open(path, 'rb') as f:
            light = pickle.load(f)
        return light['l']
    
    def load_sequence(self, dirs, ismask=False):
        frames = []
        for dir in dirs:
            frame = Image.open(dir)
            frame = self.transform_img(frame)
            if not ismask:
                frame = self.transform_scale(frame)
            frames.append(frame)
        frames = torch.stack(frames, dim=0)
        return frames
    
    def load_correspondences(self, dir):
        all_corre = np.load(dir)
        corrs = []
        for key in list(all_corre.keys()):
            corrs.append(all_corre[key])
        corrs = np.stack(corrs, axis=0).astype(np.float32)
        corrs = torch.from_numpy(corrs)
        return corrs

def create_dataset(opt):
    all_image_dir, all_mask_dir, all_corre = get_dir_video(opt)
    dataset_train = Dataset(opt, all_image_dir, all_mask_dir, all_corre)
    dataset_val = Dataset(opt, valid_dir, light_info)
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=opt.batch_size,
                                                num_workers=opt.data_load_works,
                                                sampler=train_sampler if opt.distributed else None)
    print(f"Dataloader Train created, length {len(dataset_train)}")
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, num_workers=opt.data_load_works,
                                                sampler=val_sampler if opt.distributed else None)
    print(f"Dataloader Validation created, length {len(dataset_val)}")
    return dataloader_train#, dataloader_val