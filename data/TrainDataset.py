import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import pickle
from . import get_dir, get_dir_video
import sys
sys.path.append(os.getenv('PWD'))
from video_utils import ImageSequenceReader

permute = lambda x: x.permute(*torch.arange(x.ndim-1, -1, -1))

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
normalize = lambda x: x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-5)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, dir=None, yaml_info=None):
        self.opt = opt
        self.prt_dir = dir
        self.num = len(dir)
        self.light_yaml_info = yaml_info

        self.prt_d_dir = []
        self.bprt_d_dir = []
        self.prt_s_dir = []
        self.shading_dir = []
        self.bshading_dir = []
        self.albedo_dir = []
        self.mask_dir = []
        self.skin_mask_dir = []
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
            self.skin_mask_dir.append(os.path.join(change_dir('skin_mask'), pose_name + '.png'))
            self.transport_d_dir.append(os.path.join(change_dir('transport_d'), pose_name + '.npy'))
            self.transport_s_dir.append(os.path.join(change_dir('transport_s'), pose_name + '.npy'))
            self.light_dir.append(self.light_yaml_info[scan_name][img_name])
            self.name.append(os.path.basename(scan_name + '_' + img_name))
            if opt.use_res:
                self.bprt_d_dir.append(os.path.join(change_dir('bprt_d'), img_name))
                self.bshading_dir.append(os.path.join(change_dir('bshading'), img_name))

        self.transform_img = transforms.ToTensor()
        self.transform_scale = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        idx1 = idx % self.num
        mask = Image.open(self.mask_dir[idx1])
        skin_mask = Image.open(self.skin_mask_dir[idx1])
        albedo = Image.open(self.albedo_dir[idx1])
        prt = Image.open(self.prt_dir[idx1])
        prt_d = Image.open(self.prt_d_dir[idx1])
        prt_s = Image.open(self.prt_s_dir[idx1])
        shading = Image.open(self.shading_dir[idx1])
        transport_d = np.load(self.transport_d_dir[idx1])
        transport_s = np.load(self.transport_s_dir[idx1])
        light = self.load_light(self.light_dir[idx1])[:, ::-1] # bgr -> rgb

        mask = self.transform_img(mask)
        skin_mask = self.transform_img(skin_mask)
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
            'parsing': skin_mask,
            'transport_d': transport_d,
            'transport_s': transport_s,
            'light': light,
            'name': self.name[idx1]
        }
        
        if self.opt.use_res:
            bprt_d = Image.open(self.bprt_d_dir[idx1])
            bprt_d = self.transform_img(bprt_d)
            bshading = Image.open(self.bshading_dir[idx1]) # srgb
            bshading = self.transform_img(bshading)
            return_dict.update({
                'bprt_d': bprt_d,
                'bshading': bshading
            })
        return return_dict

    def load_light(self, path):
        with open(path, 'rb') as f:
            light = pickle.load(f)
        return light['l']
    
class Dataset_Video(torch.utils.data.Dataset):
    def __init__(self, opt, dirs, val=False):
        self.opt = opt
        self.image_dir, self.mask_dir, self.names = dirs[:3]
        if not val:
            self.albedo_dir, self.light_dir, self.parsing_dir, self.prtd_dir, self.prts_dir, self.shading_dir, self.transportd_dir, self.transports_dir, self.corre_dir = dirs[3:]
        self.num = len(self.image_dir)
        self.val = val
        
        self.transform_img = transforms.ToTensor()
        self.transform_scale = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        idx_cut = idx % self.num
        img_dirs = self.image_dir[idx_cut]
        mask_dirs = self.mask_dir[idx_cut]
        imgs = self.load_img_sequence(img_dirs, itype='image')
        masks = self.load_img_sequence(mask_dirs, itype='mask')
        return_dict = {
            'input': imgs,
            'mask': masks,
            'name': self.names[idx_cut]
        }
        if not self.val:
            albedo_dirs = self.albedo_dir[idx_cut]
            corre_dirs = self.corre_dir[idx_cut]
            light_dirs = self.light_dir[idx_cut]
            parsing_dirs = self.parsing_dir[idx_cut]
            prtd_dirs = self.prtd_dir[idx_cut]
            prts_dirs = self.prts_dir[idx_cut]
            shading_dirs = self.shading_dir[idx_cut]
            transportd_dirs = self.transportd_dir[idx_cut]
            transports_dirs = self.transports_dir[idx_cut]

            albedos = self.load_img_sequence(albedo_dirs, itype='albedo')
            corre = self.load_correspondences(corre_dirs)
            light = self.load_np_sequence(light_dirs)
            prt_d = self.load_img_sequence(prtd_dirs, itype='prt_d')
            prt_s = self.load_img_sequence(prts_dirs, itype='prt_s')
            shading = self.load_img_sequence(shading_dirs, itype='shading')
            parsing = self.load_img_sequence(parsing_dirs, itype='parsing')
            transport_d = self.load_np_sequence(transportd_dirs)
            transport_s = self.load_np_sequence(transports_dirs)
            return_dict.update({'albedo': albedos,
                                'corre': corre,
                                'prt_d': prt_d,
                                'prt_s': prt_s,
                                'shading': shading,
                                'parsing': parsing,
                                'transport_d': transport_d,
                                'transport_s': transport_s,
                                'light': light,
                                })
        return return_dict

    def load_light(self, path):
        with open(path, 'rb') as f:
            light = pickle.load(f)
        return light['l']
    
    def load_np_sequence(self, dirs):
        frames = []
        for dir in dirs:
            frame = np.load(dir).astype(np.float32)
            frame = torch.from_numpy(frame)
            frames.append(frame)
        frames = torch.stack(frames, dim=0)
        return frames
    
    def load_img_sequence(self, dirs, itype="image"):
        frames = []
        for dir in dirs:
            frame = Image.open(dir)
            if itype == 'mask':
                frame = frame.convert('L')
            frame = self.transform_img(frame)
            if itype == "image":
                frame = self.transform_scale(frame)
            elif itype == 'mask':
                frame = frame > 0.4
            frames.append(frame)
        frames = torch.stack(frames, dim=0)
        return frames
    
    def load_correspondences(self, dir):
        all_corre = np.load(dir)
        corrs = []
        for key in list(all_corre.keys()):
            corrs.append(torch.from_numpy(all_corre[key].astype(np.float32)))
        return corrs
    
# class Dataset_Video(torch.utils.data.Dataset):
#     def __init__(self, opt, all_image_dir, all_mask_dir, all_albedo_dir=None, names=None, all_corre=None):
#         self.opt = opt
#         self.image_dir = all_image_dir
#         self.mask_dir = all_mask_dir
#         self.albedo_dir = all_albedo_dir
#         self.corre_dir = all_corre
#         self.names = names
#         self.num = len(self.image_dir)
        
#         self.transform_img = transforms.ToTensor()
#         self.transform_scale = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

#     def __len__(self):
#         return self.num

#     def __getitem__(self, idx):
#         idx_cut = idx % self.num
#         img_dirs = self.image_dir[idx_cut]
#         mask_dirs = self.mask_dir[idx_cut]
#         imgs = self.load_sequence(img_dirs, itype='image')
#         masks = self.load_sequence(mask_dirs, itype='mask')
#         return_dict = {
#             'input': imgs,
#             'mask': masks,
#             'name': self.names[idx_cut]
#         }
#         if self.albedo_dir is not None:
#             albedo_dirs = self.albedo_dir[idx_cut]
#             albedos = self.load_sequence(albedo_dirs, itype='albedo')
#             return_dict.update({'albedo': albedos})
        
#         if self.corre_dir is not None:
#             corre_dir = self.corre_dir[idx % self.num]
#             corre = self.load_correspondences(corre_dir)
#             return_dict.update({'corre': corre})
#         return return_dict

#     def load_light(self, path):
#         with open(path, 'rb') as f:
#             light = pickle.load(f)
#         return light['l']
    
#     def load_sequence(self, dirs, itype="image"):
#         frames = []
#         for dir in dirs:
#             frame = Image.open(dir)
#             if itype == 'mask':
#                 frame = frame.convert('L')
#             frame = self.transform_img(frame)
#             if itype == "image":
#                 frame = self.transform_scale(frame)
#             elif itype == 'mask':
#                 frame = frame > 0.4
#             frames.append(frame)
#         frames = torch.stack(frames, dim=0)
#         return frames
    
#     def load_correspondences(self, dir):
#         all_corre = np.load(dir)
#         corrs = []
#         for key in list(all_corre.keys()):
#             corrs.append(torch.from_numpy(all_corre[key].astype(np.float32)))
#         return corrs

def create_dataset(opt):
    # if opt.video:
    #     all_image_dir, all_mask_dir, all_albedo_dir, all_corre, train_names, val_image_dir, val_mask_dir, val_names = get_dir_video(opt)
    #     dataset_train = Dataset_Video(opt, all_image_dir, all_mask_dir, all_albedo_dir, train_names, all_corre)
    #     dataset_val = Dataset_Video(opt, val_image_dir, val_mask_dir, names=val_names)
    if opt.video:
        # train_dirs, val_image_dir, val_mask_dir, val_names = get_dir_video(opt)
        train_dirs, val_dirs = get_dir_video(opt)
        dataset_train = Dataset_Video(opt, train_dirs)
        dataset_val = Dataset_Video(opt, val_dirs, val=True)
    else:
        train_dir, valid_dir, light_info = get_dir(opt)
        dataset_train = Dataset(opt, train_dir, light_info)
        dataset_val = Dataset(opt, valid_dir, light_info)
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=opt.batch_size,
                                                num_workers=opt.data_load_works,
                                                sampler=train_sampler if opt.distributed else None)
    print(f"Dataloader Train created, length {len(dataset_train)}")

    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                batch_size=opt.batch_size if not opt.video else 1,
                                                shuffle=False,
                                                num_workers=opt.data_load_works,
                                                sampler=val_sampler if opt.distributed else None)
    print(f"Dataloader Validation created, length {len(dataset_val)}")
    
    return dataloader_train, dataloader_val