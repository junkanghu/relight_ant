import torch.nn as nn
import torch
import numpy as np
import models.networks as networks
import torch.nn.functional as F
from .BaseModel import BaseModel
from torch.cuda.amp import autocast
import lpips
from .ssim import SSIM_Loss
DEBUG = False
eps = 1e-7
l2png = lambda x: torch.pow(x.clamp_min_(0), 1/2.2).clip(0, 1)
l2srgb = lambda x: torch.pow(x.clamp_min_(0.) + eps, 1/2.2)
mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)
scale = lambda x: x * 2. - 1.

def get_img(img, name):
    img = img[0].detach().cpu().numpy()
    img = (np.clip(img.transpose(1, 2, 0), 0.01, 0.99) * 255.).astype(np.uint8)
    import imageio.v2 as imageio
    imageio.imwrite("{}.png".format(name), img)

class lumos(BaseModel):
    def __init__(self, opt):
        super(lumos, self).__init__(opt)
        if opt.sil1:
            self.loss_l1 = self.scale_invariant_l1loss
        else:
            self.loss_l1 = nn.L1Loss().to(self.device)
        self.criterion_ssim = SSIM_Loss(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.criterion_ssim_transport = SSIM_Loss(data_range=1.0, size_average=True, channel=opt.sh_num).to(self.device)
        self.sf_loss = networks.SpatialFrequencyLoss(device=self.device)
        self.sf_loss_transport = networks.SpatialFrequencyLoss(num_channels=opt.sh_num, device=self.device)
        if opt.albedo_vgg or opt.shading_vgg:
            self.loss_vgg_api = lpips.LPIPS(net='vgg').to(self.device)
        self.einsum_format = 'bchw,bcd->bdhw' if not opt.video else 'bfchw,bfcd->bfdhw'

        self.optimizer_name = ['optim_main']
        self.net_name = ['main']
        
        self.net_main = networks.define_G(opt.sh_num, is_video=opt.video, use_tfa=opt.use_tfa)
        self.optim_main = torch.optim.Adam(self.net_main.parameters(),lr=opt.lr, betas=(0.5, 0.999))

        if opt.use_res:
            self.optimizer_name.append('optim_residual')
            self.net_name.append('residual')
            self.net_residual = networks.define_G(opt.sh_num, is_res=True, is_video=opt.video)
            self.optim_residual = torch.optim.Adam(self.net_residual.parameters(),lr=opt.lr, betas=(0.5, 0.999))

        self.loss_name = self.get_loss_name()
        self.initialize_loss()

    def set_input_image(self, data):
        self.mask = data['mask'].to(self.device)
        self.parsing = data['parsing'].to(self.device) * self.mask
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
            
    def set_input_video(self, data, val=False):
        self.mask = data['mask'].to(self.device)
        self.input = data['input'].to(self.device) * self.mask
        self.name = data['name']
        if not val:
            self.albedo = data['albedo'].to(self.device) * self.mask
            self.corre = [c.to(self.device) for c in data['corre']]
            self.parsing = data['parsing'].to(self.device) * self.mask
            self.shading = data['shading'].to(self.device) * self.mask
            self.transport_d = data['transport_d'].to(self.device)
            self.transport_s = data['transport_s'].to(self.device)
            self.prt_d = data['prt_d'].to(self.device) * self.mask
            self.prt_s = data['prt_s'].to(self.device) * self.mask
            self.light = data['light'].to(self.device)
        
    def set_input(self, data, val=False):
        if self.opt.video:
            self.set_input_video(data, val=val)
        else:
            self.set_input_image(data)
            
    def get_transport_sf_loss(self):
        loss = self.sf_loss_transport(self.transport_d, self.transport_d_hat)
        return loss
    
    def get_light_loss(self):
        loss = 0.
        b, f, *_ = self.light_hat.shape
        for i in range(b):
            light = self.light_hat[i]
            for j in range(f):
                for k in range(j, f):
                    loss += self.loss_l1(light[j], light[k])
        return loss

    def scale_invariant_l2loss(outs: torch.Tensor, targets: torch.Tensor, reduction="mean"):
        """
        outs: N ( x C) x H x W
        targets: N ( x C) x H x W
        reduction: ...
        """
        outs = outs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)
        alpha = (targets - outs).mean(dim=1, keepdim=True)
        return F.mse_loss(outs + alpha, targets, reduction=reduction)
    
    def scale_invariant_l1loss(self, outs: torch.Tensor, targets: torch.Tensor, reduction="mean"):
        """
        outs: N ( x C) x H x W
        targets: N ( x C) x H x W
        reduction: ...
        """
        outs = outs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)
        alpha = (targets - outs).mean(dim=1, keepdim=True)
        return F.l1_loss(outs + alpha, targets, reduction=reduction)

    def get_chromatic_loss(self, input, target):
        input_sum = torch.sum(input, dim=-3, keepdim=True)
        input_chromatic_map = input / (input_sum + 1e-6)
        target_sum = torch.sum(target, dim=-3, keepdim=True)
        target_chromatic_map = target / (target_sum + 1e-6)
        loss = self.loss_l1(input_chromatic_map, target_chromatic_map)
        return loss
    
    def get_regular_loss(self):
        averaged = torch.mean(self.albedo_hat, dim=-3, keepdim=True)
        loss = self.loss_l1(averaged.expand_as(self.albedo_hat), self.albedo_hat)
        return loss
    
    def get_tv_loss(self, x):
        b, c, h, w = x.size()
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        loss = (tv_h + tv_w) / (b * c * h * w)
        return loss
    
    def get_focal_loss(self, input, target):
        input = input.clamp(eps, 1.-eps)
        target = target.clamp(eps, 1.-eps)
        loss = ((-(1. - input) ** 2 * target * torch.log(input) \
            - input ** 2 * (1. - target) * torch.log(1. -input)).sum()) / (input.shape[-1] * input.shape[-2])
        return loss

    def get_matching_loss(self):
        loss = 0.
        for res in self.res_groups:
            if res is not None:
                loss += self.loss_l1(res.flatten(0, 1), torch.empty_like(res).fill_(0.).flatten(0, 1))
        return loss
    
    def get_albedo_matching_loss(self):
        loss = 0.
        for idx, corr in enumerate(self.corre):
            albedos = self.albedo_hat
            b, f, *_ = albedos.shape
            seq = corr[..., :2].to(torch.int64)
            mid = corr[..., 2:].to(torch.int64)
            albedo_seq = albedos[:, idx if idx < f//2 else idx + 1, ...]
            albedo_mid = albedos[:, f//2, ...]
            corre_seq = albedo_seq[torch.arange(b).unsqueeze(1), :, seq[:, :, 0], seq[:, :, 1]]
            corre_mid = albedo_mid[torch.arange(b).unsqueeze(1), :, mid[:, :, 0], mid[:, :, 1]]
            loss += self.loss_l1(corre_seq, corre_mid)
            
        return loss
    
    def get_ssim_loss(self):
        loss = 0.
        if self.opt.ssim_albedo:
            loss += self.criterion_ssim(self.albedo, self.albedo_hat)
        if self.opt.ssim_transport:
            loss += self.criterion_ssim_transport(self.transport_d, self.transport_d_hat)
        if self.opt.ssim_shading:
            loss += self.criterion_ssim(self.shading_all_hat, self.shading)
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
        self.albedo_hat = self.mask * albedo_hat[..., :3, :, :]
        self.parsing_hat = self.mask * albedo_hat[..., 3:4, :, :]
        self.transport_d_hat = self.mask * transport_d_hat
        self.transport_s_hat = self.mask * transport_s_hat
        self.light_hat = light_hat
        if self.opt.video and self.opt.use_tfa:
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
            
    def backward_video(self):
        self.loss_total = 0.
        self.backward_image()
        self.loss_matching = self.get_albedo_matching_loss()
        self.loss_total += self.loss_matching

        self.loss_light = self.get_light_loss()
        self.loss_total += self.loss_light
        # self.loss_regular = self.loss_l1(self.rendering, self.input * 0.5 + 0.5) * 0.1

        # self.loss_albedo = self.loss_l1(self.albedo, self.albedo_hat) * 10.
        # self.loss_total = self.loss_total + (self.loss_regular + self.loss_albedo)
        return self.loss_total

    def backward_image(self):
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
            self.opt.w_albedo_sf * L_albedo_sf +\
            self.opt.w_transport * L_transport_s * 0.5 +\
            self.opt.w_shading_transport * (L_spec_transport + L_spec_transport1) * 0.5 +\
            self.opt.w_shading_light * (L_spec_light + L_spec_light1) * 0.5 +\
            self.opt.w_shading_all * (L_spec_all + L_spec_all1) * 0.5 +\
            self.opt.w_rendering_all * (L_rendering_all1 + L_rendering_all2 + L_rendering_all3) * 0.5 + self.opt.w_shading_sf * L_shading_all_sf
            
        self.loss_parsing = self.get_focal_loss(self.parsing_hat, self.parsing)
        self.loss_total += self.loss_parsing
        
        if self.opt.ssim_albedo or self.opt.ssim_transport or self.opt.ssim_shading:
            self.loss_ssim = self.get_ssim_loss()
            self.loss_total += self.loss_ssim
            
        if self.opt.regular:
            self.loss_regular = self.get_regular_loss()
            self.loss_total += self.loss_regular

        if self.opt.albedo_vgg:
            self.loss_albedo_vgg = torch.mean(self.loss_vgg_api(scale(self.albedo), scale(self.albedo_hat)))
            self.loss_total += self.loss_albedo_vgg

        if self.opt.shading_vgg:
            self.loss_shading_vgg = torch.mean(self.loss_vgg_api(scale(self.shading), scale(self.shading_all_hat)))
            self.loss_total += 0.1 * self.loss_shading_vgg
            
        if self.opt.transport_sf:
            self.loss_transport_sf = self.get_transport_sf_loss()
            self.loss_total += self.loss_transport_sf

        if self.opt.tv:
            self.loss_tv = self.get_tv_loss(self.shading_all_hat) + self.get_tv_loss(self.transport_d_hat)
            self.loss_total += self.loss_tv
            
        if self.opt.use_res and self.epoch >= self.opt.res_epoch:
            self.loss_res = self.get_res_loss()
            self.loss_total += self.loss_res
        # if self.opt.video:
        #     self.loss_matching = self.get_matching_loss()
        #     self.loss_total += self.loss_matching

    def optimize_parameters(self, epoch):
        self.epoch = epoch
        self.optim_main.zero_grad()  # set G_A and G_B's gradients to zero
        if self.opt.use_res and self.epoch >= self.opt.res_epoch:
            self.optim_residual.zero_grad()
        with autocast(enabled=self.opt.amp):
            self.forward()
            if not self.opt.video:
                self.backward_image()
            else:
                self.backward_video()
        self.scaler.scale(self.loss_total).backward()
        self.scaler.step(self.optim_main)
        if self.opt.use_res and self.epoch >= self.opt.res_epoch:
            self.scaler.step(self.optim_residual)
        self.scaler.update()
    
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
        if self.opt.tv:
            name.append('tv')
        if self.opt.regular:
            name.append('regular')
        if self.opt.shading_vgg:
            name.append('shading_vgg')
        if self.opt.albedo_vgg:
            name.append('albedo_vgg')
        if self.opt.use_res:
            name.append('res')
        # if self.opt.video:
        #     name.append('matching')
        return name