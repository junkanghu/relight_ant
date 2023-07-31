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
    def __init__(self, opt, dir=None, yaml_info=None, stage='train'):
        self.opt = opt
        self.stage = stage
        if not stage == "test":
            self.prt_dir = dir
            self.num = len(dir)
            self.light_yaml_info = yaml_info

            self.prt_d_dir = []
            self.prt_s_dir = []
            self.shading_dir = []
            self.albedo_dir = []
            self.mask_dir = []
            self.transport_d_dir = []
            self.transport_s_dir = []
            self.light_dir = []
            self.name = []
            
            for prt_dir in self.prt_dir:
                dir_name = os.path.dirname(prt_dir)
                scan_name = os.path.basename(os.path.dirname(dir_name))
                change_dir = lambda x: dir_name.replace('prt', x)
                img_name = os.path.basename(prt_dir)
                pose_name = img_name.split('_')[0]
                
                self.prt_d_dir.append(os.path.join(change_dir('prt_d'), img_name))
                self.prt_s_dir.append(os.path.join(change_dir('prt_s'), img_name))
                self.shading_dir.append(os.path.join(change_dir('shading'), img_name))
                self.albedo_dir.append(os.path.join(change_dir('albedo'), pose_name + '.png'))
                self.mask_dir.append(os.path.join(change_dir('mask'), pose_name + '.png'))
                self.transport_d_dir.append(os.path.join(change_dir('transport_d'), pose_name + '.npy'))
                self.transport_s_dir.append(os.path.join(change_dir('transport_s'), pose_name + '.npy'))
                self.light_dir.append(self.light_yaml_info[scan_name][img_name])
                self.name.append(os.path.basename(scan_name + '_' + img_name))
        else:
            self.input_dir, self.mask_dir, self.name = dir
            self.num = len(self.input_dir)
        
        # rgb -> bgr
        self.transform_img = transforms.ToTensor()
        self.transform_scale = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        idx1 = idx % self.num
        mask = Image.open(self.mask_dir[idx1])
        if not self.stage == "test": 
            albedo = Image.open(self.albedo_dir[idx1])
            prt = Image.open(self.prt_dir[idx1])
            prt_d = Image.open(self.prt_d_dir[idx1])
            prt_s = Image.open(self.prt_s_dir[idx1])
            shading = Image.open(self.shading_dir[idx1])
            transport_d = np.load(self.transport_d_dir[idx1])
            transport_s = np.load(self.transport_s_dir[idx1])
            light = self.load_light(self.light_dir[idx1])[:, ::-1] # bgr -> rgb

            mask = self.transform_img(mask)
            albedo = self.transform_img(albedo)
            prt = self.transform_img(prt)
            prt_d = self.transform_img(prt_d)
            prt_s = self.transform_img(prt_s)
            shading = self.transform_img(shading)
            _, h, w = albedo.shape
            transport_d = permute(torch.FloatTensor(transport_d)).reshape(-1, h, w)
            transport_s = permute(torch.FloatTensor(transport_s)).reshape(-1, h, w)
            input = self.transform_scale(prt)
            light = torch.FloatTensor(light.copy())
        
            return_dict = {
                'input': input,
                'prt_d': prt_d,
                'prt_s': prt_s,
                'shading': shading,
                'albedo': albedo,
                'mask': mask,
                'transport_d': transport_d,
                'transport_s': transport_s,
                'light': light,
            }
        else:
            input = Image.open(self.input_dir[idx1])
            input = self.transform_scale(self.transform_img(input))
            return_dict = {
                'input': input,
                'mask': mask,
            }
        if not self.stage == "train":
            return_dict['name'] = self.name[idx1]
        return return_dict

    def load_light(self, path):
        with open(path, 'rb') as f:
            light = pickle.load(f)
        return light['l']

def get_dir(opt):
    if not opt.test:
        with open(opt.yaml_dir) as f:
            data_info = yaml.safe_load(f)
        scan_path = []

        for idx, scan in enumerate(data_info):
            scan_dir = scan[0]['albedo'].split('/albedo')[0] # xxx/../0000
            scan_path.append(scan_dir)
        # get scan dirs
        test_scan_dir = scan_path[::50]
        train_scan_dir = [p for p in scan_path if p not in test_scan_dir]
        
        # get rendered image dirs
        prt_all_path_train = []
        prt_all_path_test = []
        light_yaml_info = {}
        for idx, scan_dir in enumerate(scan_path):
            prt_dir = os.path.join(scan_dir, 'prt')
            prt_path = sorted(os.listdir(prt_dir))
            prt_path = [os.path.join(prt_dir, p) for p in prt_path if not p[0] == '.']
            if scan_dir in train_scan_dir:
                prt_all_path_train += prt_path
            else:
                prt_all_path_test += prt_path
            
            l_yaml = os.path.join(scan_dir, 'yaml/data.yaml')
            with open(l_yaml) as f:
                yaml_info = yaml.safe_load(f)
            light_yaml_info[os.path.basename(scan_dir)] = yaml_info

        return prt_all_path_train, prt_all_path_test, light_yaml_info
    else:
        img_name = sorted(os.listdir(opt.test_dir))
        mask_dir = []
        input_dir = []
        name = []
        for p in img_name:
            if not p[0] == '.':
                mask_dir.append(os.path.join(opt.test_dir, p))
                input_dir.append(os.path.join(opt.test_dir.replace('img', 'mask'), p[:-3] + 'png'))
                name.append(p.split('.')[0])
        return (input_dir, mask_dir, name)

def create_dataset(opt):
    train_dir, valid_dir, light_info = get_dir(opt)
    dataset_train = Dataset(opt, train_dir, light_info, stage='train')
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=opt.batch_size,
                                                num_workers=opt.data_load_works,
                                                sampler=train_sampler if opt.distributed else None)
    print(f"Dataloader {dataset_train.stage} created, length {len(dataset_train)}")
    
    dataset_val = Dataset(opt, valid_dir, light_info, stage='val')
    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, num_workers=opt.data_load_works,
                                                sampler=val_sampler if opt.distributed else None)
    print(f"Dataloader {dataset_val.stage} created, length {len(dataset_val)}")
    
    return dataloader_train, dataloader_val