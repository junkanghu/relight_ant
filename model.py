import torch.nn as nn
import torchvision
import torch
import os
import numpy as np
from PIL import Image
from networks import CNNAE2ResNet
from utils import ssim, psnr
import sfloss as sfl
import torch.nn.functional as F
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

class lumos(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.optimizers = []
        self.device = torch.device("cuda", opt.local_rank)
        self.name = None

        self.loss_l1 = nn.L1Loss().to(self.device)
        self.sf_loss = sfl.SpatialFrequencyLoss(num_channels=3, device=self.device)

        self.optimizer_name = ['optimizer_G']
        self.net_name = ['model']
        self.net_model = CNNAE2ResNet(albedo_decoder_channels=3).to(self.device)

        self.optimizer_G = torch.optim.Adam(self.net_model.parameters(),lr=opt.lr, betas=(0.5, 0.999))
        self.loss_all = {}
        self.loss_name = self.get_loss_name()

        self.initialize_loss()

    def set_input(self, data, val=False):
        self.mask = data['mask'].to(self.device)
        self.albedo = data['albedo'].to(self.device) * self.mask
        self.shading = data['shading'].to(self.device) * self.mask
        self.transport_d = data['transport_d'].to(self.device)
        self.transport_s = data['transport_s'].to(self.device)
        self.prt_d = data['prt_d'].to(self.device) * self.mask
        self.prt_s = data['prt_s'].to(self.device) * self.mask
        self.light = data['light'].to(self.device)
        self.input = data['input'].to(self.device) * self.mask
        if val:
            self.name = data['name']

    def forward(self, val=False):
        transport_d_hat, transport_s_hat, albedo_hat, light_hat = self.net_model(self.input)
        self.albedo_hat = self.mask * albedo_hat
        self.transport_d_hat = self.mask * transport_d_hat
        self.transport_s_hat = self.mask * transport_s_hat
        self.light_hat = light_hat
        
        self.shading_all_hat = l2srgb(torch.einsum('bchw,bcd->bdhw', self.transport_d_hat, self.light_hat)) * self.mask
        self.sepc_all_hat = 0.8 * l2srgb(torch.einsum('bchw,bcd->bdhw', self.transport_s_hat, self.light_hat)) * self.mask
        self.rendering_all_hat = self.albedo_hat * self.shading_all_hat
        self.rendering_all_hat3 = self.rendering_all_hat + self.albedo_hat * self.sepc_all_hat

    def backward_G(self):
        L_transport_d = self.loss_l1(self.transport_d_hat, self.transport_d)
        L_transport_s = self.loss_l1(self.transport_s_hat, self.transport_s)
        L_albedo = self.loss_l1(self.albedo_hat, self.albedo)
        L_albedo_sf = self.sf_loss(self.albedo_hat, self.albedo)
        L_light = self.loss_l1(self.light_hat, self.light)

        shading_transport_hat = l2srgb(torch.einsum('bchw,bcd->bdhw', self.transport_d_hat, self.light)) * self.mask
        L_shading_transport = self.loss_l1(shading_transport_hat, self.shading)

        spec_transport_hat = 0.8 * l2srgb(torch.einsum('bchw,bcd->bdhw', self.transport_s_hat, self.light)) * self.mask
        # L_spec_transport = self.loss_l1((spec_transport_hat * self.albedo).clamp_(0, 1), self.prt_s)
        # L_spec_transport1 = self.loss_l1((spec_transport_hat * self.albedo_hat).clamp_(0, 1), self.prt_s)
        L_spec_transport = self.loss_l1(spec_transport_hat * self.albedo, self.prt_s)
        L_spec_transport1 = self.loss_l1(spec_transport_hat * self.albedo_hat, self.prt_s)

        shading_light_hat = l2srgb(torch.einsum('bchw,bcd->bdhw', self.transport_d, self.light_hat)) * self.mask
        L_shading_light = self.loss_l1(shading_light_hat, self.shading)

        spec_light_hat = 0.8 * l2srgb(torch.einsum('bchw,bcd->bdhw', self.transport_s, self.light_hat)) * self.mask
        # L_spec_light = self.loss_l1((spec_light_hat * self.albedo).clamp_(0, 1), self.prt_s)
        # L_spec_light1 = self.loss_l1((spec_light_hat * self.albedo_hat).clamp_(0, 1), self.prt_s)
        L_spec_light = self.loss_l1(spec_light_hat * self.albedo, self.prt_s)
        L_spec_light1 = self.loss_l1(spec_light_hat * self.albedo_hat, self.prt_s)

        L_shading_all = self.loss_l1(self.shading_all_hat, self.shading)
        # L_spec_all = self.loss_l1((self.sepc_all_hat * self.albedo).clamp_(0, 1), self.prt_s)
        # L_spec_all1 = self.loss_l1((self.sepc_all_hat * self.albedo_hat).clamp_(0, 1), self.prt_s)
        L_spec_all = self.loss_l1(self.sepc_all_hat * self.albedo, self.prt_s)
        L_spec_all1 = self.loss_l1(self.sepc_all_hat * self.albedo_hat, self.prt_s)
        L_shading_all_sf = self.sf_loss(self.shading_all_hat, self.shading)

        rendering_albedo_hat = (self.albedo_hat * self.shading)
        # L_rendering_albedo = self.loss_l1(rendering_albedo_hat.clamp_(0, 1), self.prt_d)
        L_rendering_albedo = self.loss_l1(rendering_albedo_hat, self.prt_d)
                
        # rendering_transport_hat = (self.albedo * shading_transport_hat).clamp_(0, 1)
        rendering_transport_hat = self.albedo * shading_transport_hat
        L_rendering_transport = self.loss_l1(rendering_transport_hat, self.prt_d)
                
        # rendering_light_hat = (self.albedo * shading_light_hat).clamp_(0, 1)
        rendering_light_hat = self.albedo * shading_light_hat
        L_rendering_light = self.loss_l1(rendering_light_hat, self.prt_d)
                
        # rendering_albedo_transport_hat = (self.albedo_hat * shading_transport_hat).clamp_(0, 1)
        rendering_albedo_transport_hat = self.albedo_hat * shading_transport_hat
        L_rendering_albedo_transport = self.loss_l1(rendering_albedo_transport_hat, self.prt_d)
                
        # rendering_transport_light_hat = (self.albedo * self.shading_all_hat).clamp_(0, 1)
        rendering_transport_light_hat = self.albedo * self.shading_all_hat
        L_rendering_transport_light = self.loss_l1(rendering_transport_light_hat, self.prt_d)

        # rendering_albedo_light_hat = (self.albedo_hat * shading_light_hat).clamp_(0, 1)
        rendering_albedo_light_hat = self.albedo_hat * shading_light_hat
        L_rendering_albedo_light = self.loss_l1(rendering_albedo_light_hat, self.prt_d)
                
        rendering_all_hat1 = self.rendering_all_hat + self.albedo_hat * spec_transport_hat
        rendering_all_hat2 = self.rendering_all_hat + self.albedo_hat * spec_light_hat
        # L_rendering_all = self.loss_l1(self.rendering_all_hat.clamp_(0, 1), self.prt_d)
        # L_rendering_all1 = self.loss_l1(rendering_all_hat1.clamp_(0, 1), self.input * 0.5 + 0.5)
        # L_rendering_all2 = self.loss_l1(rendering_all_hat2.clamp_(0, 1), self.input * 0.5 + 0.5)
        # L_rendering_all3 = self.loss_l1(self.rendering_all_hat3.clamp_(0, 1), self.input * 0.5 + 0.5)
        L_rendering_all = self.loss_l1(self.rendering_all_hat, self.prt_d)
        L_rendering_all1 = self.loss_l1(rendering_all_hat1, self.input * 0.5 + 0.5)
        L_rendering_all2 = self.loss_l1(rendering_all_hat2, self.input * 0.5 + 0.5)
        L_rendering_all3 = self.loss_l1(self.rendering_all_hat3, self.input * 0.5 + 0.5)
        
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
        self.loss_total.backward()

    def optimize_parameters(self):
        self.forward()
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

    def update_lr(self, epoch):
        decay_epoch = self.opt.epochs // 2
        lr = self.opt.lr * (1. - 1. * max(epoch - decay_epoch, 0) / (decay_epoch + 10))
        print("learning rate:", lr)
        for name in self.optimizer_name:
            optimizer = getattr(self, name)
            for param in optimizer.param_groups:
                param['lr'] = lr

    def load_ckpt(self, val=False):
        os.makedirs(self.opt.out_dir, exist_ok=True)
        start_epoch = 1
        ckpt_dir = os.path.join(self.opt.out_dir, "latest.pth")
        if not os.path.exists(ckpt_dir) or self.opt.discard_ckpt:
            print('No checkpoints!')
            return start_epoch

        print("Loading checkpoint:", ckpt_dir)
        ckpt = torch.load(ckpt_dir, map_location=self.device)
        
        for name in self.net_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.load_state_dict(ckpt[name])

        start_epoch = ckpt["epoch"] + 1
        for name in self.optimizer_name:
            if isinstance(name, str):
                optimizer = getattr(self, name)
                optimizer.load_state_dict(ckpt[name])
        return start_epoch

    def save_ckpt(self, epoch):
        save_dir_epoch = os.path.join(self.opt.out_dir, "%d.pth" % epoch)
        save_dir_latest = os.path.join(self.opt.out_dir, "latest.pth")
        results = {"epoch": epoch}
        for name in self.net_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                results[name] = net.cpu().state_dict()
                net = net.cuda(self.device)
        for name in self.optimizer_name:
            optimizer = getattr(self, name)
            results[name] = optimizer.state_dict()

        torch.save(results, save_dir_epoch)
        torch.save(results, save_dir_latest)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def plot_val(self, epoch=1, test=None):
        # shape of output_vs_gt_plot [B, C, H, W]
        psnr_group = []
        for id in range(self.albedo_hat.shape[0]):
            if test is None:
                output_vs_gt_plot = torch.cat([
                                    self.albedo_hat[id:id+1].detach(), 
                                    self.albedo[id:id+1].detach(), 
                                    self.shading_all_hat[id:id+1].detach(),
                                    self.shading[id:id+1].detach(),
                                    self.sepc_all_hat[id:id+1].detach(),
                                    self.prt_s[id:id+1].detach(),
                                    self.rendering_all_hat[id:id+1].detach(),
                                    self.prt_d[id:id+1].detach(),
                                    self.rendering_all_hat3[id:id+1].detach(),
                                    ((self.input[id:id+1] * 0.5 + 0.5) * self.mask).detach(),
                                    ], 0)
            else:    
                output_vs_gt_plot = torch.cat([
                                    self.albedo_hat[id:id+1].detach(), 
                                    self.shading_all_hat[id:id+1].detach(),
                                    self.sepc_all_hat[id:id+1].detach(),
                                    self.rendering_all_hat[id:id+1].detach(),
                                    self.rendering_all_hat3[id:id+1].detach(),
                                    ((self.input[id:id+1] * 0.5 + 0.5) * self.mask).detach(),
                                    ], 0)
                
            out = torchvision.utils.make_grid(output_vs_gt_plot,
                                            scale_each=False,
                                            normalize=False,
                                            nrow=2 if test is not None else 5).detach().cpu().numpy()
            out = out.transpose(1, 2, 0)
            out = np.clip(out, 0.01, 0.99)
            scale_factor = 255
            tensor = (out * scale_factor).astype(np.uint8)
            img = Image.fromarray(tensor)
            img_path = os.path.join(self.opt.out_dir, "val_imgs")
            if test is not None:
                img_path = os.path.join(self.opt.out_dir, test + "_imgs") 
            os.makedirs(img_path, exist_ok=True)
            img_dir = os.path.join(img_path, "{0}_{1}".format(epoch, self.name[id][:-3] + 'png'))
            if test is not None:
                img_dir = os.path.join(img_path, "{0}_{1}".format(test, self.name[id][:-3] + 'png'))
            print('saving rendered img to {}'.format(img_dir))
            img.save(img_dir)

            # calculate ssim and psnr
            psnr_group.append(psnr(((self.input[id] * 0.5 + 0.5) * self.mask[id]).detach(), self.rendering_all_hat3[id].detach()))
        ssim_batch = [ssim(((self.input * 0.5 + 0.5) * self.mask).detach(), self.rendering_all_hat3.detach())]
        return ssim_batch, psnr_group

    def get_loss_name(self):
        name = ['total']
        return name
    
    def initialize_loss(self):
        for idx, name in enumerate(self.loss_name):
            self.loss_all[name] = 0.

    def gather_loss(self, end_of_epoch=False):
        if end_of_epoch:
            loss_all = self.loss_all.copy()
            self.initialize_loss()
            return loss_all

        for idx, name in enumerate(self.loss_name):
            value = getattr(self, 'loss_' + name).item()
            self.loss_all[name] += value
