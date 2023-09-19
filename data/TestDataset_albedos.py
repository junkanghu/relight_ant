import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import pickle
import yaml
from tqdm import tqdm
permute = lambda x: x.permute(*torch.arange(x.ndim-1, -1, -1))

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
normalize = lambda x: x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-5)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, dir=None, stage='train'):
        self.opt = opt
        self.stage = stage
        self.input_dir, self.mask_dir, self.path = dir
        self.num = len(self.input_dir)
        
        # rgb -> bgr
        self.transform_img = transforms.ToTensor()
        self.transform_scale = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        idx1 = idx % self.num
        mask = Image.open(self.mask_dir[idx1]).convert('L')
        input = Image.open(self.input_dir[idx1])
        input = self.transform_scale(self.transform_img(input))
        mask = self.transform_img(mask) > 0.4
        return_dict = {
            'input': input,
            'mask': mask,
            'name': self.input_dir[idx1]
        }
        return return_dict

    def load_light(self, path):
        with open(path, 'rb') as f:
            light = pickle.load(f)
        return light['l']

def get_dir(opt):
    mask_dir = []
    input_dir = []
    names = []
    video_name = sorted(os.listdir(opt.test_dir))
    for name in tqdm(video_name):
        video_image_dir = os.path.join(opt.test_dir, name, "images")
        seq_names = sorted(os.listdir(video_image_dir))
        for n in seq_names:
            if not n[0] == '.':
                input_dir.append(os.path.join(video_image_dir, n))
                mask_dir.append(os.path.join(video_image_dir.replace('images', 'masks'), n))
                names.append(n.split('.')[0])
    return (input_dir, mask_dir, names)

def create_dataset(opt):
    dir = get_dir(opt)
    dataset_test = Dataset(opt, dir, stage='test')
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=opt.batch_size,
                                                num_workers=opt.data_load_works,
                                                sampler=train_sampler if opt.distributed else None)
    print(f"Dataloader {dataset_test.stage} created, length {len(dataset_test)}")
    
    return dataloader_test