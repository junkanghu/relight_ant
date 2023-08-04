import torch
import models.networks as networks
from .BaseModel import BaseModel
eps = 1e-8
l2srgb = lambda x: torch.pow(x.clamp_min_(0.) + eps, 1/2.2)

class lumos(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.net_main = self.net_main = networks.VideoRelight(sh_num=opt.sh_num, use_res=opt.use_res).to(self.device)

    def set_input(self, data):
        self.mask = data['mask'].to(self.device)
        self.albedo = data['albedo'].to(self.device) * self.mask
        self.shading = data['shading'].to(self.device) * self.mask
        self.bshading = data['bshading'].to(self.device) * self.mask
        self.transport_d = data['transport_d'].to(self.device)
        self.transport_s = data['transport_s'].to(self.device)
        self.prt_d = data['prt_d'].to(self.device) * self.mask
        self.bprt_d = data['bprt_d'].to(self.device) * self.mask
        self.prt_s = data['prt_s'].to(self.device) * self.mask
        self.light = data['light'].to(self.device)
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
        self.rendering_all_hat = self.albedo_hat * self.shading_all_hat
        self.rendering_all_hat3 = self.rendering_all_hat + self.albedo_hat * self.sepc_all_hat
    
    def concat_img(self, id):
        output_vs_gt_plot = torch.cat([
                            self.albedo_hat[id:id+1].detach(), 
                            self.shading_all_hat[id:id+1].detach(),
                            self.shading_final[id:id+1].detach(),
                            self.sepc_all_hat[id:id+1].detach(),
                            self.rendering_all_hat[id:id+1].detach(),
                            self.rendering_d.detach(),
                            self.rendering_all_hat3[id:id+1].detach(),
                            self.rendering_final.detach(),
                            ((self.input[id:id+1] * 0.5 + 0.5) * self.mask[id:id+1]).detach(),
                            ], 0)
        return output_vs_gt_plot
