import torch
import models.networks as networks
from .BaseModel import BaseModel
import os
import numpy as np
eps = 1e-8
l2srgb = lambda x: torch.pow(x.clamp_min_(0.) + eps, 1/2.2)

class lumos(BaseModel):
    def __init__(self, opt):
        super(lumos, self).__init__(opt)
        self.einsum_format = 'bchw,bcd->bdhw' if not opt.video else 'bfchw,bfcd->bfdhw'
        self.net_name = ['main']
        self.net_main = networks.define_G(opt.sh_num, is_video=opt.video, use_tfa=opt.use_tfa)
        if opt.use_res:
            self.net_name.append('residual')
            self.net_residual = networks.define_G(opt.sh_num, is_res=True)

    def set_input(self, data):
        self.mask = data['mask'].to(self.device)
        if 'light' in list(data.keys()):
            self.light = data['light'].to(self.device)
        self.input = data['input'].to(self.device) * self.mask
        self.name = data['name']

    def forward(self):
        transport_d_hat, transport_s_hat, albedo_hat, light_hat, *res_groups = self.net_main(self.input)
        self.albedo_hat = self.mask * albedo_hat[..., :3, :, :]
        self.parsing_hat = self.mask * albedo_hat[..., 3:4, :, :]
        self.transport_d_hat = self.mask * transport_d_hat
        self.transport_s_hat = self.mask * transport_s_hat
        self.light_hat = light_hat
        if self.light is not None:
            self.light_hat = self.light
        
        self.shading_all_hat_linear = torch.einsum(self.einsum_format, self.transport_d_hat, self.light_hat)
        self.shading_all_hat = l2srgb(self.shading_all_hat_linear) * self.mask
        self.sepc_all_hat = l2srgb(torch.einsum(self.einsum_format, self.transport_s_hat, self.light_hat)) * self.mask
        self.diffuse = self.albedo_hat * self.shading_all_hat
        self.rendering = self.diffuse + self.albedo_hat * self.sepc_all_hat
        
        if self.opt.use_res:
            self.shading_res = self.net_residual(torch.cat([self.shading_all_hat.detach(), 
                                self.transport_d_hat.detach()], 1), self.light_hat.detach()) * self.mask
            self.shading_final = l2srgb(self.shading_all_hat_linear + self.shading_res) * self.mask
            self.diffuse_final = self.shading_final * self.albedo_hat
            self.rendering_final = self.diffuse_final + self.albedo_hat * self.sepc_all_hat
    
    def concat_img(self, id):
        if self.opt.use_res:
            output_vs_gt_plot = torch.cat([
                                self.albedo_hat[id:id+1].detach(), 
                                self.shading_all_hat[id:id+1].detach(),
                                l2srgb(self.shading_res[id:id+1].detach()),
                                self.shading_final[id:id+1].detach(),
                                self.sepc_all_hat[id:id+1].detach(),
                                self.diffuse[id:id+1].detach(),
                                self.diffuse_final[id:id+1].detach(),
                                self.rendering[id:id+1].detach(),
                                self.rendering_final[id:id+1].detach(),
                                ((self.input[id:id+1] * 0.5 + 0.5) * self.mask[id:id+1]).detach(),
                                ], 0)
        else:
            output_vs_gt_plot = torch.cat([
                                self.albedo_hat[id:id+1].detach(), 
                                self.shading_all_hat[id:id+1].detach(),
                                self.sepc_all_hat[id:id+1].detach(),
                                self.diffuse[id:id+1].detach(),
                                self.rendering[id:id+1].detach(),
                                ((self.input[id:id+1] * 0.5 + 0.5) * self.mask[id:id+1]).detach(),
                                ], 0)
        return output_vs_gt_plot

    def save_albedos(self,):
        for id in range(self.albedo_hat.shape[0]):
            albedo = self.albedo_hat[id]
            img = self.tensor2image(albedo)
            img_path = self.name[id].replace('images', 'albedos')
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            print('saving rendered img to {}'.format(img_path))
            img.save(img_path)
            
    def saving_img(self, img, path=None, name=None):
        img = self.tensor2image(img)
        img_path = path.replace('images', name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        print('saving rendered img to {}'.format(img_path))
        img.save(img_path)
    
    def saving_np(self, array, path=None, name=None):
        array = array.detach().cpu().numpy()
        array_path = path.replace('images', name).split('.')[0] + '.npy'
        os.makedirs(os.path.dirname(array_path), exist_ok=True)
        print('saving {} to {}'.format(name, array_path))
        np.save(array_path, array)
        
    def save_inferred(self):
        for id in range(self.albedo_hat.shape[0]):
            self.saving_img(self.albedo_hat[id], self.name[id], 'albedo')
            self.saving_img(self.parsing_hat[id] > 0.45, self.name[id], 'parsing')
            self.saving_img(self.shading_all_hat[id], self.name[id], 'shading')
            self.saving_img(self.sepc_all_hat[id], self.name[id], 'prt_s')
            self.saving_img(self.diffuse[id], self.name[id], 'prt_d')
            self.saving_img(self.rendering[id], self.name[id], 'prt')
            self.saving_np(self.transport_d_hat[id], self.name[id], 'transport_d')
            self.saving_np(self.transport_s_hat[id], self.name[id], 'transport_s')
            self.saving_np(self.light_hat[id], self.name[id], 'light')

    # def save_inferred(self):
    #     torch2np = lambda x: x.detach().cpu().numpy()
    #     for id in range(self.albedo_hat.shape[0]):
    #         return_dict = {
    #             'transport_d': torch2np(self.transport_d_hat[0]),
    #             'transport_s': torch2np(self.transport_s_hat[0]),
    #             'albedo': torch2np(self.albedo_hat[0]),
    #             'light': torch2np(self.light_hat[0]),
    #             'parsing': torch2np(self.parsing_hat[0]),
    #             'shading': torch2np(self.shading_all_hat[0]),
    #             'spec': torch2np(self.sepc_all_hat[0]),
    #             'diffuse': torch2np(self.diffuse[0]),
    #             'rendering': torch2np(self.rendering[0])
    #         }
    #         np_path = self.name[id].replace('images', 'inferred')
    #         os.makedirs(os.path.dirname(np_path), exist_ok=True)
    #         print('saving inferred components to {}'.format(np_path))
    #         np.savez(np_path, **return_dict)