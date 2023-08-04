import torch
import models.networks as networks
from .BaseModel import BaseModel
eps = 1e-8
l2srgb = lambda x: torch.pow(x.clamp_min_(0.) + eps, 1/2.2)

class lumos(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.net_main = networks.define_G(opt.sh_num)
        if opt.use_res:
            self.net_residual = networks.define_G(opt.sh_num, is_res=True)

    def set_input(self, data):
        self.mask = data['mask'].to(self.device)
        # self.light = data['light'].to(self.device)
        self.input = data['input'].to(self.device) * self.mask
        self.name = data['name']

    def forward(self):
        transport_d_hat, transport_s_hat, albedo_hat, light_hat = self.net_main(self.input)
        self.albedo_hat = self.mask * albedo_hat
        self.transport_d_hat = self.mask * transport_d_hat
        self.transport_s_hat = self.mask * transport_s_hat
        self.light_hat = light_hat
        
        self.shading_all_hat_linear = torch.einsum('bchw,bcd->bdhw', self.transport_d_hat, self.light_hat)
        self.shading_all_hat = l2srgb(self.shading_all_hat_linear) * self.mask
        self.sepc_all_hat = l2srgb(torch.einsum('bchw,bcd->bdhw', self.transport_s_hat, self.light_hat)) * self.mask
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
