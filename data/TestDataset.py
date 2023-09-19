import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import pickle
import yaml
permute = lambda x: x.permute(*torch.arange(x.ndim-1, -1, -1))

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
normalize = lambda x: x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-5)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, dir=None, stage='train'):
        self.opt = opt
        self.stage = stage
        self.input_dir, self.mask_dir, self.name = dir
        self.test_light_dir = opt.test_light_dir
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
            'name': self.name[idx1]
        }
        if self.test_light_dir is not None:
            light = self.load_light(self.test_light_dir)[:, ::-1] # bgr -> rgb
            light = torch.FloatTensor(light.copy())
            return_dict.update({'light': light})
        return return_dict

    def load_light(self, path):
        with open(path, 'rb') as f:
            light = pickle.load(f)
        return light['l']

def get_dir(opt):
    img_name = sorted(os.listdir(opt.test_dir))
    mask_dir = []
    input_dir = []
    name = []
    for p in img_name:
        if not p[0] == '.':
            input_dir.append(os.path.join(opt.test_dir, p))
            mask_dir.append(os.path.join(opt.test_dir.replace('images', 'masks'), p))
            name.append(p.split('.')[0])
    return (input_dir, mask_dir, name)

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