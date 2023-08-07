import torch.nn as nn
import torch
import numpy as np
import models.networks as networks
import torch.nn.functional as F
from .BaseModel import BaseModel
from torch.cuda.amp import autocast
DEBUG = False
eps = 1e-8
l2png = lambda x: torch.pow(x.clamp_min_(0), 1/2.2).clip(0, 1)
l2srgb = lambda x: torch.pow(x.clamp_min_(0.) + eps, 1/2.2)
mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

def get_img(img, name):
    img = img[0].detach().cpu().numpy()
    img = (np.clip(img.transpose(1, 2, 0), 0.01, 0.99) * 255.).astype(np.uint8)
    import imageio.v2 as imageio
    imageio.imwrite("{}.png".format(name), img)

class lumos(BaseModel):
    def __init__(self, opt):
        super(lumos, self).__init__(opt)
        self.loss_l1 = nn.L1Loss().to(self.device)
        self.sf_loss = networks.SpatialFrequencyLoss(device=self.device)
        self.einsum_format = 'bchw,bcd->bdhw' if not opt.video else 'bfchw,bfcd->bfdhw'

        self.optimizer_name = ['optim_main']
        self.net_name = ['main']
        
        self.net_main = networks.define_G(opt.sh_num, is_video=opt.video)
        self.optim_main = torch.optim.Adam(self.net_main.parameters(),lr=opt.lr, betas=(0.5, 0.999))

        if opt.use_res:
            self.optimizer_name.append('optim_residual')
            self.net_name.append('residual')
            self.net_residual = networks.define_G(opt.sh_num, is_res=True, is_video=opt.video)
            self.optim_residual = torch.optim.Adam(self.net_residual.parameters(),lr=opt.lr, betas=(0.5, 0.999))

        self.loss_name = self.get_loss_name()
        self.initialize_loss()

    def set_input(self, data):
        self.mask = data['mask'].to(self.device)
        self.albedo = data['albedo'].to(self.device) * self.mask
        self.shading = data['shading'].to(self.device) * self.mask
        self.transport_d = data['transport_d'].to(self.device)
        self.transport_s = data['transport_s'].to(self.device)
        self.prt_d = data['prt_d'].to(self.device) * self.mask
        self.prt_s = data['prt_s'].to(self.device) * self.mask
        self.light = data['light'].to(self.device)
        self.input = data['input'].to(self.device) * self.mask
        self.name = data['name']
        
        if self.opt.use_res:
            self.bshading = data['bshading'].to(self.device) * self.mask
            self.bprt_d = data['bprt_d'].to(self.device) * self.mask

    def get_matching_loss(self):
        loss = 0.
        for res in self.res_groups:
            if res is not None:
                loss += self.loss_l1(res.flatten(0, 1), torch.empty_like(res).fill_(0.).flatten(0, 1))
        return loss
    
    def get_res_loss(self):
        loss = 0.
        L_shading_final = self.loss_l1(self.shading_final, self.bshading)
        L_shading_sf = self.sf_loss(self.shading_final, self.bshading)
        L_rendering_diff = self.loss_l1(self.diffuse_final, self.bprt_d)
        loss = L_shading_final + L_shading_sf + L_rendering_diff
        return loss

    def forward(self):
        transport_d_hat, transport_s_hat, albedo_hat, light_hat, *res_groups = self.net_main(self.input)
        self.albedo_hat = self.mask * albedo_hat
        self.transport_d_hat = self.mask * transport_d_hat
        self.transport_s_hat = self.mask * transport_s_hat
        self.light_hat = light_hat
        if self.opt.video:
            self.res_groups = res_groups[0]
        
        self.shading_all_hat_linear = torch.einsum(self.einsum_format, self.transport_d_hat, self.light_hat)
        self.shading_all_hat = l2srgb(self.shading_all_hat_linear) * self.mask
        self.sepc_all_hat = l2srgb(torch.einsum(self.einsum_format, self.transport_s_hat, self.light_hat)) * self.mask
        self.diffuse = self.albedo_hat * self.shading_all_hat
        self.rendering = self.diffuse + self.albedo_hat * self.sepc_all_hat
        
        if self.opt.use_res and self.epoch >= self.opt.res_epoch:
            self.shading_res = self.net_residual(torch.cat([self.shading_all_hat.detach(), 
                                self.transport_d_hat.detach()], -3), self.light_hat.detach()) * self.mask
            self.shading_final = l2srgb(self.shading_all_hat_linear + self.shading_res) * self.mask
            self.diffuse_final = self.shading_final * self.albedo_hat
            self.rendering_final = self.diffuse_final + self.albedo_hat * self.sepc_all_hat

    def backward_G(self):
        L_transport_d = self.loss_l1(self.transport_d_hat, self.transport_d)
        L_transport_s = self.loss_l1(self.transport_s_hat, self.transport_s)
        L_albedo = self.loss_l1(self.albedo_hat, self.albedo)
        L_albedo_sf = self.sf_loss(self.albedo_hat, self.albedo)
        L_light = self.loss_l1(self.light_hat, self.light)

        shading_transport_hat = l2srgb(torch.einsum(self.einsum_format, self.transport_d_hat, self.light)) * self.mask
        L_shading_transport = self.loss_l1(shading_transport_hat, self.shading)

        spec_transport_hat = l2srgb(torch.einsum(self.einsum_format, self.transport_s_hat, self.light)) * self.mask
        L_spec_transport = self.loss_l1(spec_transport_hat * self.albedo, self.prt_s)
        L_spec_transport1 = self.loss_l1(spec_transport_hat * self.albedo_hat, self.prt_s)

        shading_light_hat = l2srgb(torch.einsum(self.einsum_format, self.transport_d, self.light_hat)) * self.mask
        L_shading_light = self.loss_l1(shading_light_hat, self.shading)

        spec_light_hat = l2srgb(torch.einsum(self.einsum_format, self.transport_s, self.light_hat)) * self.mask
        L_spec_light = self.loss_l1(spec_light_hat * self.albedo, self.prt_s)
        L_spec_light1 = self.loss_l1(spec_light_hat * self.albedo_hat, self.prt_s)

        L_shading_all = self.loss_l1(self.shading_all_hat, self.shading)
        L_spec_all = self.loss_l1(self.sepc_all_hat * self.albedo, self.prt_s)
        L_spec_all1 = self.loss_l1(self.sepc_all_hat * self.albedo_hat, self.prt_s)
        L_shading_all_sf = self.sf_loss(self.shading_all_hat, self.shading)

        rendering_albedo_hat = (self.albedo_hat * self.shading)
        L_rendering_albedo = self.loss_l1(rendering_albedo_hat, self.prt_d)
                
        rendering_transport_hat = self.albedo * shading_transport_hat
        L_rendering_transport = self.loss_l1(rendering_transport_hat, self.prt_d)
                
        rendering_light_hat = self.albedo * shading_light_hat
        L_rendering_light = self.loss_l1(rendering_light_hat, self.prt_d)
                
        rendering_albedo_transport_hat = self.albedo_hat * shading_transport_hat
        L_rendering_albedo_transport = self.loss_l1(rendering_albedo_transport_hat, self.prt_d)
                
        rendering_transport_light_hat = self.albedo * self.shading_all_hat
        L_rendering_transport_light = self.loss_l1(rendering_transport_light_hat, self.prt_d)

        rendering_albedo_light_hat = self.albedo_hat * shading_light_hat
        L_rendering_albedo_light = self.loss_l1(rendering_albedo_light_hat, self.prt_d)
                
        rendering_all_hat1 = self.diffuse + self.albedo_hat * spec_transport_hat
        rendering_all_hat2 = self.diffuse + self.albedo_hat * spec_light_hat
        L_rendering_all = self.loss_l1(self.diffuse, self.prt_d)
        L_rendering_all1 = self.loss_l1(rendering_all_hat1, self.input * 0.5 + 0.5)
        L_rendering_all2 = self.loss_l1(rendering_all_hat2, self.input * 0.5 + 0.5)
        L_rendering_all3 = self.loss_l1(self.rendering, self.input * 0.5 + 0.5)
        
        self.loss_total = self.opt.w_transport * L_transport_d + self.opt.w_albedo * L_albedo + self.opt.w_light * L_light +\
            self.opt.w_shading_transport * L_shading_transport + self.opt.w_shading_light * L_shading_light + self.opt.w_shading_all * L_shading_all +\
            self.opt.w_rendering_albedo * L_rendering_albedo + self.opt.w_rendering_transport * L_rendering_transport + self.opt.w_rendering_light * L_rendering_light + \
            self.opt.w_rendering_albedo_transport * L_rendering_albedo_transport + self.opt.w_rendering_transport_light * L_rendering_transport_light + self.opt.w_rendering_albedo_light * L_rendering_albedo_light +\
            self.opt.w_rendering_all * L_rendering_all + \
            self.opt.w_albedo_sf * L_albedo_sf + self.opt.w_shading_sf * L_shading_all_sf +\
            self.opt.w_transport * L_transport_s * 0.5 +\
            self.opt.w_shading_transport * (L_spec_transport + L_spec_transport1) * 0.5 +\
            self.opt.w_shading_light * (L_spec_light + L_spec_light1) * 0.5 +\
            self.opt.w_shading_all * (L_spec_all + L_spec_all1) * 0.5 +\
            self.opt.w_rendering_all * (L_rendering_all1 + L_rendering_all2 + L_rendering_all3) * 0.5
            
        if self.opt.use_res and self.epoch >= self.opt.res_epoch:
            self.loss_res = self.get_res_loss()
            self.loss_total += self.loss_res
        if self.opt.video:
            self.loss_matching = self.get_matching_loss()
            self.loss_total += self.loss_matching

    def optimize_parameters(self, epoch):
        self.epoch = epoch
        self.optim_main.zero_grad()  # set G_A and G_B's gradients to zero
        if self.opt.use_res and self.epoch >= self.opt.res_epoch:
            self.optim_residual.zero_grad()
        if self.opt.amp:
            with autocast():
                self.forward()
                self.backward_G()
            self.scaler.scale(self.loss_total).backward()
            self.scaler.step(self.optim_main)
            if self.opt.use_res and self.epoch >= self.opt.res_epoch:
                self.scaler.step(self.optim_residual)
            self.scaler.update()
        else:
            self.forward()
            self.backward_G()
            self.loss_total.backward()
            self.optim_main.step()
            if self.opt.use_res and self.epoch >= self.opt.res_epoch:
                self.optim_residual.step()
    
    def concat_img(self, id):
        if self.opt.use_res and self.epoch >= self.opt.res_epoch:
            output_vs_gt_plot = torch.cat([
                                self.albedo_hat[id:id+1].detach(), 
                                self.albedo[id:id+1].detach(), 
                                self.shading_all_hat[id:id+1].detach(),
                                self.shading[id:id+1].detach(),
                                l2srgb(self.shading_res[id:id+1].detach()),
                                self.shading_final[id:id+1].detach(),
                                self.bshading[id:id+1].detach(),
                                self.sepc_all_hat[id:id+1].detach(),
                                self.prt_s[id:id+1].detach(),
                                self.diffuse[id:id+1].detach(),
                                self.prt_d[id:id+1].detach(),
                                self.diffuse_final[id:id+1].detach(),
                                self.bprt_d[id:id+1].detach(),
                                self.rendering[id:id+1].detach(),
                                self.rendering_final[id:id+1].detach(),
                                ((self.input[id:id+1] * 0.5 + 0.5) * self.mask[id:id+1]).detach(),
                                ], 0)
        else:
            output_vs_gt_plot = torch.cat([
                                self.albedo_hat[id:id+1].detach(), 
                                self.albedo[id:id+1].detach(), 
                                self.shading_all_hat[id:id+1].detach(),
                                self.shading[id:id+1].detach(),
                                self.sepc_all_hat[id:id+1].detach(),
                                self.prt_s[id:id+1].detach(),
                                self.diffuse[id:id+1].detach(),
                                self.prt_d[id:id+1].detach(),
                                self.rendering[id:id+1].detach(),
                                ((self.input[id:id+1] * 0.5 + 0.5) * self.mask[id:id+1]).detach(),
                                ], 0)
        return output_vs_gt_plot

    def get_loss_name(self):
        name = ['total']
        if self.opt.use_res:
            name.append('res')
        if self.opt.video:
            name.append('matching')
        return name