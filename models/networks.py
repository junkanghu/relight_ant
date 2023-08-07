import torch
import torch.nn as nn
import functools
from .attend import Attend, TFA_Attention
import numpy as np
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as transforms
from antialiased_cnns.blurpool import BlurPool
import math
from .deform_conv import ModulatedDeformConvPack as DCN
from einops import rearrange
ac_type = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh
}

norm_type = {
    "batch": nn.BatchNorm2d,
    "instance": nn.InstanceNorm2d
}

class VideoRelight(nn.Module):
    def __init__(self, encoder_dims=[3, 64, 128, 256, 512, 512],
                decoder_dims=[512, 512, 256, 128, 64, 32],
                skip_dims=[512, 256, 128, 64, 0],
                light_dims=[512*3, 512, 256, 128], sh_num=25, is_video=False):
        super().__init__()
        self.encoder = Encoder(encoder_dims)
        self.decoder = Decoder(decoder_dims, skip_dims, light_dims, sh_num, is_video=is_video)
        
    def forward(self, x):
        x = self.encoder(x)
        t_d, t_s, albedo, light, *res_groups = self.decoder(x)
        return t_d, t_s, albedo, light, res_groups[0] if len(res_groups) > 0 else None

class ConvBlock(nn.Module):
    def __init__(self, in_nc, out_nc,
                kernel_size, stride, padding, bias=False,
                norm=None, ac=None, lr_slope=0.01):
        """
        Args:
            in_nc (int): input channels
            out_nc (int): output channels
            kernel_size (int): kernel_size for conv
            stride (int): stride for conv
            padding (int): padding for conv
            bias (bool, optional): bias for conv
            norm (string, optional): choice ["batch", "instance"]
            ac (string, optional): choice ["relu", "leakyrelu", "sigmoid", "tanh"]
            lr_slope (float, optional): negtive slope for LeakyRelu. Defaults to 0.01.
        """   
        super().__init__()
        model = [nn.Conv2d(in_nc, out_nc, kernel_size, stride, padding, bias=bias)]
        if norm is not None:
            model.append(norm_type[norm](out_nc))
        if ac is not None:
            if ac == "relu":
                ac_layer = ac_type[ac](True)
            elif ac == "leakyrelu":
                ac_layer = ac_type[ac](lr_slope, True)
            else:
                ac_layer = ac_type[ac]()
            model.append(ac_layer)
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, dims:list, lr_slope=0.01):
        """
        Args:
            dims (list): Features dimensions. The first is the input one.
            lr_slope (float, optional): Used for LeakyRelu
        """        
        super().__init__()
        self.n_layer = len(dims) - 1
        for i in range(self.n_layer):
            setattr(self, "e{}".format(i),
                    ConvBlock(dims[i], dims[i+1], 4, 2, 1,
                                norm="batch" if i > 0 else None, ac="leakyrelu", lr_slope=lr_slope))
        
    def forward_single_frame(self, x):
        out = []
        for i in range(self.n_layer):
            block = getattr(self, "e{}".format(i))
            x = block(x)
            out.append(x)
        return out

    def forward_time_series(self, x):
        B, F = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, F)) for f in features]
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

class UpsamplingBlock(nn.Module):
    def __init__(self, in_nc, skin_nc, out_nc,
                kernel_size, stride, padding, bias=False,
                norm=None, ac=None, dropout=False, drop_p=0.5, lr_slope=0.01, is_video=False, is_albedo=False, is_last=False):
        """Upsampling block for decoder.

        Args:
            in_nc (int): input channels
            out_nc (int): output channels
            kernel_size (int): kernel_size for conv
            stride (int): stride for conv
            padding (int): padding for conv
            bias (bool, optional): bias for conv
            norm (string, optional): choice ["batch", "instance"]
            ac (string, optional): choice ["relu", "leakyrelu", "sigmoid", "tanh"]
            dropout (bool, optional): Whether use dropout. Defaults to False.
            drop_p (float, optional): Drop out parameter. Defaults to 0.5.
            lr_slope (float, optional): negtive slope for LeakyRelu. Defaults to 0.01.
            is_video (bool, optional): Whether train with video.
            is_albedo (bool, optional): Whether albedo decoder block.
            is_last (bool, optional): Whether the last albedo decoder block which does't use temporal block.
        """        
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        model = [nn.Conv2d(in_nc + skin_nc, out_nc, kernel_size, stride, padding, bias=bias)]
        if norm is not None:
            model.append(norm_type[norm](out_nc))
        if dropout:
            model.append(nn.Dropout(drop_p))
        if ac is not None:
            if ac == "relu":
                ac_layer = ac_type[ac](True)
            elif ac == "leakyrelu":
                ac_layer = ac_type[ac](lr_slope, True)
            else:
                ac_layer = ac_type[ac]()
            model.append(ac_layer)
        self.model = nn.Sequential(*model)
        self.is_albedo = is_albedo
        self.temporal = None
        if is_video and not is_last:
            self.temporal = TemporalBlock(out_nc)
            if is_albedo:
                self.tfa = TFA_Block(out_nc)
    
    def forward_single_frame(self, x1, x2=None):
        """

        Args:
            x1: Features from the last block of the decoder.
            x2: Features from the encoder.

        Returns:
            Features
        """        
        x1 = self.upsample(x1)
        if x2 is not None:
            x1 = torch.cat([x1, x2], 1)
        x1 = self.model(x1)
        return x1, None

    def forward_time_series(self, x1, x2=None):
        B, F, *_ = x1.shape
        process = lambda x1, x2, func: func(x1.flatten(0, 1), x2.flatten(0, 1) if x2 is not None else None)[0].unflatten(0, x1.shape[:2])
        x1 = process(x1, x2, self.forward_single_frame)
        res = None
        if self.temporal is not None:
            x1 = self.temporal(x1)
            if self.is_albedo:
                res = self.tfa(x1)
        return x1, res
    
    def forward(self, x1, x2=None):
        if x1.ndim == 5:
            return self.forward_time_series(x1, x2)
        else:
            return self.forward_single_frame(x1, x2)

class Decoder_Albedo(nn.Module):
    def __init__(self, decoder_dims, skip_dims, is_video=False):
        """Branch for albedo.

        Args:
            decoder_dims (list): Dimensions of features in the decoder.
            skip_dims (list): Dimensions of features from encoder.
        """        
        super().__init__()
        self.bottlneck = ResnetBlock(decoder_dims[0], 'zero', "batch", False, True)
        self.n_layer = len(decoder_dims) - 1
        for i in range(self.n_layer):
            setattr(self, "up{}".format(i),
                UpsamplingBlock(decoder_dims[i], skip_dims[i], decoder_dims[i+1], 3, 1, 1,
                norm="batch", ac="relu", dropout=True if i < 2 else False, is_video=is_video, is_albedo=True, is_last=(i == self.n_layer - 1)))
        self.out = nn.Sequential(nn.Conv2d(decoder_dims[-1], 3, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.Sigmoid())
        
    def block(self, x, bottle):
        res_groups = []
        for i in range(self.n_layer):
            layer = getattr(self, "up{}".format(i))
            inter, *res = layer(bottle if i == 0 else inter, x[-(i + 2)] if not i == self.n_layer - 1 else None)
            res_groups += res
        return inter, res_groups
    
    def forward_single_frame(self, x):
        bottle = self.bottlneck(x[-1])
        inter, _ = self.block(x, bottle)
        inter = self.out(inter)
        return inter, bottle

    def forward_time_series(self, x):
        process = lambda x, module: module(x.flatten(0, 1)).unflatten(0, x.shape[:2])
        bottle = process(x[-1], self.bottlneck)
        inter, res_groups = self.block(x, bottle)
        inter = process(inter, self.out)
        return inter, bottle, res_groups
    
    def forward(self, x):
        if x[0].ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

class Decoder_Transport(nn.Module):
    def __init__(self, decoder_dims, skip_dims, sh_num, is_video=False):
        """Branch for transport.

        Args:
            decoder_dims (list): Dimensions of features in the decoder.
            skip_dims (list): Dimensions of features from encoder.
            sh_num (int): Number of SH parameters which is determined by the SH order.
        """        
        super().__init__()
        self.bottlneck = ResnetBlock(decoder_dims[0], 'zero', "batch", False, True)
        self.n_layer = len(decoder_dims) - 1
        for i in range(self.n_layer):
            setattr(self, "upd{}".format(i),
                UpsamplingBlock(decoder_dims[i], skip_dims[i], decoder_dims[i+1], 3, 1, 1,
                    norm="batch", ac="relu", dropout=True if i < 2 else False, is_video=is_video))
            if i > 2:
                setattr(self, "ups{}".format(i),
                    UpsamplingBlock(decoder_dims[i], skip_dims[i], decoder_dims[i+1], 3, 1, 1,
                        norm="batch", ac="relu", dropout=True if i < 2 else False, is_video=is_video))
        self.out_d = nn.Conv2d(decoder_dims[-1], sh_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.out_s = nn.Conv2d(decoder_dims[-1], sh_num, kernel_size=3, stride=1, padding=1, bias=False)
        
    def block(self, x, bottle):
        for i in range(self.n_layer):
            if i > 2:
                layer_s = getattr(self, "ups{}".format(i))
                inter1, _ = layer_s(inter if i == 3 else inter1, x[-(i + 2)] if not i == self.n_layer - 1 else None)
            layer_d = getattr(self, "upd{}".format(i))
            inter, _ = layer_d(bottle if i == 0 else inter, x[-(i + 2)] if not i == self.n_layer - 1 else None)
        return inter, inter1
        
    def forward_single_frame(self, x):
        bottle = self.bottlneck(x[-1])
        inter, inter1 = self.block(x, bottle)
        inter = self.out_d(inter)
        inter1 = self.out_s(inter1)
        return inter, inter1, bottle

    def forward_time_series(self, x):
        process = lambda x, module: module(x.flatten(0, 1)).unflatten(0, x.shape[:2])
        bottle = process(x[-1], self.bottlneck)
        inter, inter1 = self.block(x, bottle)
        inter = process(inter, self.out_d)
        inter1 = process(inter1, self.out_s)
        return inter, inter1, bottle
        
    def forward(self, x):
        if x[0].ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
    
class Decoder_Light(nn.Module):
    def __init__(self, dims, sh_num, is_video=False):
        """Branch for transport.

        Args:
            dims (list): Dimensions of features in the decoder.
            sh_num (int): Number of SH parameters which is determined by the SH order.
        """        
        super().__init__()
        self.n_layer = len(dims) - 1
        self.sh_num = sh_num
        self.is_video = is_video
        for i in range(self.n_layer):
            setattr(self, "down{}".format(i),
                ConvBlock(dims[i], dims[i+1], 4, 2, 1, norm="batch", ac="leakyrelu"))
            if is_video:
                setattr(self, "temp{}".format(i), TemporalBlock(dims[i+1]))
        self.out = nn.Conv2d(dims[-1], sh_num * 3, 4, stride=2, padding=1) # 2 -> 1
        
    def block(self, x, shape=None):
        for i in range(self.n_layer):
            layer = getattr(self, "down{}".format(i))
            if self.is_video:
                x = layer(x.flatten(0, 1)).unflatten(0, shape)
                layer_t = getattr(self, "temp{}".format(i))
                x = layer_t(x)
            else:
                x = layer(x)
        return x
        
    def forward_single_frame(self, e, bt, ba):
        """
        Args:
            e: The features of the last layer of the encoder.
            ba: BottleNeck features of albedo Decoder.
            bt: BottleNeck features of transport Decoder.
        """
        x = torch.cat([e, bt, ba], 1)
        x = self.block(x)
        x = self.out(x)
        x = x.reshape(-1, self.sh_num, 3)
        return x

    def forward_time_series(self, e, bt, ba):
        x = torch.cat([e, bt, ba], 2)
        x = self.block(x, e.shape[:2])
        x = self.out(x.flatten(0, 1))
        x = x.reshape(*e.shape[:2], self.sh_num, 3)
        return x
        
    def forward(self, e, bt, ba):
        if e.ndim == 5:
            return self.forward_time_series(e, bt, ba)
        else:
            return self.forward_single_frame(e, bt, ba)

class Decoder(nn.Module):
    def __init__(self, decoder_dims=[512, 512, 256, 128, 64, 32],
                skip_dims=[512, 256, 128, 64, 0],
                light_dims=[512*3, 512, 256, 128], sh_num=25, is_video=False):
        super().__init__()
        self.decoder_albedo = Decoder_Albedo(decoder_dims, skip_dims, is_video=is_video)
        self.decoder_transport = Decoder_Transport(decoder_dims, skip_dims, sh_num, is_video=is_video)
        self.decoder_light = Decoder_Light(light_dims, sh_num, is_video=is_video)
        
    def forward(self, x):
        albedo, bottle_a, *res_groups = self.decoder_albedo(x)
        t_d, t_s, bottle_t = self.decoder_transport(x)
        light = self.decoder_light(x[-1], bottle_a, bottle_t)
        return t_d, t_s, albedo, light, res_groups[0] if len(res_groups) > 0 else None
    
class Decoder_Res(nn.Module):
    def __init__(self, decoder_dims, skip_dims, lr_slope=0.01, is_video=False):
        """Branch for shading residual.

        Args:
            decoder_dims (list): Dimensions of features in the decoder.
            skip_dims (list): Dimensions of features from encoder.
            lr_slope (float): Used for LeakyRelu.
        """        
        super().__init__()
        self.is_video = is_video
        self.bottlneck = ResnetBlock(decoder_dims[0], 'zero', "batch", False, True)
        self.n_layer = len(decoder_dims) - 1
        for i in range(self.n_layer):
            setattr(self, "up{}".format(i),
                UpsamplingBlock(decoder_dims[i], skip_dims[i], decoder_dims[i+1], 3, 1, 1,
                    norm="batch", ac="relu", dropout=True if i < 2 else False, lr_slope=lr_slope, is_video=is_video))
        self.fc_light = nn.Linear(25*3, decoder_dims[0])
        self.up_light = nn.Upsample(scale_factor=16)
        self.fusion = ConvBlock(decoder_dims[0]*2, decoder_dims[0], 3, 1, 1, ac="leakyrelu", lr_slope=0.2)
        self.out = nn.Sequential(ConvBlock(decoder_dims[-1], 8, 3, 1, 1),
                                ConvBlock(8, 3, 3, 1, 1))
        
    def block(self, x, inter):
        for i in range(self.n_layer):
            layer = getattr(self, "up{}".format(i))
            inter, _ = layer(inter, x[-(i + 2)] if not i == self.n_layer - 1 else None)
        return inter
        
    def forward_single_frame(self, x, light):
        """Use transport_d & shading & light to predict the residual.

        Args:
            x: Features from encoder.
            light: Light parameters. Shape: [B, 25, 3]

        Returns:
            Shading residual which is to be added to the original shading.
        """        
        inter = self.bottlneck(x[-1])
        light = torch.flatten(light, 1)
        light = self.up_light(self.fc_light(light).unsqueeze(-1).unsqueeze(-1))
        inter = self.fusion(torch.cat([inter, light], 1))
        inter = self.block(x, inter)
        inter = self.out(inter)
        return inter

    def forward_time_series(self, x, light):
        inter = self.bottlneck(x[-1].flatten(0, 1))
        light = rearrange(light, 'b f c d -> (b f) (c d)')
        light = self.up_light(self.fc_light(light).unsqueeze(-1).unsqueeze(-1))
        inter = self.fusion(torch.cat([inter, light], 1)).unflatten(0, x[0].shape[:2])
        inter = self.block(x, inter)
        inter = self.out(inter.flatten(0, 1)).unflatten(0, x[0].shape[:2])
        return inter
        
    def forward(self, x, light):
        if x[0].ndim == 5:
            return self.forward_time_series(x, light)
        else:
            return self.forward_single_frame(x, light)

class ShadingRes(nn.Module):
    def __init__(self, encoder_dims=[28, 32, 64, 128, 256, 256],
                decoder_dims=[256, 256, 128, 64, 32, 16],
                skip_dims=[256, 128, 64, 32, 0], is_video=False):
        super().__init__()
        self.encoder = Encoder(encoder_dims, lr_slope=0.2)
        self.decoder = Decoder_Res(decoder_dims, skip_dims, lr_slope=0.2, is_video=is_video)
        
    def forward(self, x, light):
        x = self.encoder(x)
        res = self.decoder(x, light)
        return res

class DeformConv(nn.Module):
    def __init__(self, nf, groups=8):
        super(DeformConv, self).__init__()
        self.off_conv1 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.off_conv2 = nn.Conv2d(nf  , nf, 3, 1, 1, bias=True)
        self.dcnpack = DCN(nf, nf, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr, ref):
        """Temporal Feature Alignment block.
        We take the middle feature as the reference one. By default, the frame number is odd.

        Args:
            nbr (tensor): Neighboring features. Shape: [B, F, C, H, W]
            ref (tensor): Reference features which is the expanded middle one of nbr. Shape: [B, F, C, H, W]

        Returns:
            tensor: aligned features.
        """        
        b, f, *_ = nbr.shape
        nbr = nbr.flatten(0, 1)
        off = self.lrelu(self.off_conv1(torch.cat([nbr, ref.flatten(0, 1)], dim=1)))
        off = self.lrelu(self.off_conv2(off))
        fea = self.dcnpack([nbr, off]).unflatten(0, (b, f)) # Shape: [B, F, C, H, W]
        return fea
    
class SpatialWeight(nn.Module):
    def __init__(self):
        """Get per-pixel feature weights to calculate the feature loss.
        """        
        super().__init__()
        self.num = int(5000)
        
    @torch.no_grad()
    def chunk(self, q, k, b, f):
        vec = torch.ones([b, f, k.shape[-1], 1], device=q.device)
        sim = torch.einsum(f"b f c d, b f c e -> b f d e", q, k)
        weight = torch.einsum(f"b f d e, b f e g -> b f g d", sim, vec)
        return weight
        
    @torch.no_grad()
    def forward(self, nbr, ref):
        b, f, c, h, w = nbr.shape
        nbr = nbr.flatten(-2, -1) # b, f, c, h*w
        ref = ref.flatten(-2, -1)
        q, k = ref, nbr
        
        weight = torch.ones([b, f, 1, h*w], device=nbr.device)
        for i in range(0, q.shape[-1], self.num):
            sum = 0
            for j in range(0, q.shape[-1], self.num):
                q_chunk = q[..., i:i+self.num]
                k_chunk = k[..., j:j+self.num]
                sum += self.chunk(q_chunk, k_chunk, b, f)
            weight[..., i:i+q_chunk.shape[-1]] = sum
        weight = torch.sigmoid(weight).unflatten(-1, (h, w))
        return weight
    
class TFA_Block(nn.Module):
    def __init__(self, in_nc, is_att=False):
        """Align all the temporal features according to the reference feature map.

        Args:
            in_nc (int): Input and output feature dimensions.
            is_att (bool, optional): When set true, use cross attention for TFA, otherwise deform conv. Defaults to False.
        """        
        super().__init__()
        self.model = TFA_Attention(in_nc) if is_att else DeformConv(in_nc)
        self.weight = SpatialWeight()
        
    def forward(self, nbr):
        """_summary_

        Args:
            nbr (tensor): Input temporal features. The middle one is the reference frame. Shape: [B, F, C, H, W]

        Returns:
            Weighted residual between aligned features and reference features. The shape is the same as the input.
        """        
        b, f, c, h, w = nbr.shape
        ref = nbr[:, f//2:f//2 + 1].expand_as(nbr)
        aligned = self.model(nbr, ref)
        weight = self.weight(nbr.detach(), ref.detach())
        res = (aligned - ref) * weight # Feature loss can be obtained by calculating the l1 loss between res and zeros.
        return res
    
class TemporalBlock(nn.Module):
    def __init__(self, in_nc, middle_nc=256, groups=8):        
        super().__init__()
        model = [
            nn.Conv1d(in_nc, middle_nc, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, middle_nc),
            nn.SiLU(True),
            nn.Conv1d(middle_nc, in_nc, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, in_nc),
            nn.SiLU(True)
        ]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        """
        Args:
            x (tensor): Features from decoder.
        """        
        b, f, c, h, w = x.shape
        x = rearrange(x, 'b f c h w -> (b f) c (h w)')
        x = self.model(x)
        x = rearrange(x, '(b f) c (h w) -> b f c h w', b=b, h=h)
        return x

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm (str)          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        norm_layer = norm_type[norm]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


def get_LoG_kernel(size, sigma, device):
    lin = torch.linspace(-(size - 1) // 2, size // 2, size, device=device)
    [x, y] = torch.meshgrid(lin, lin, indexing='ij')
    ss = sigma ** 2
    xx = x * x
    yy = y * y
    g_div_ss = torch.exp(-(xx + yy) / (2. * ss)) / (2. * torch.pi * (ss ** 2))
    a = (xx + yy - 2. * ss) * g_div_ss

    # Normalize.
    a = a - a.sum() / size ** 2
    return a

def get_LoG_filter(num_channels, sigma, device='cuda'):
    kernel_size = int(torch.ceil(8. * sigma))
    # Make into odd number.
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = get_LoG_kernel(kernel_size, sigma, device)
    # [kH, kW] => [OutChannels, (InChannels / groups) => 1, kH, kW].
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(num_channels, 1, 1, 1)

    # Create filter.
    filter = torch.nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                            kernel_size=kernel_size, groups=num_channels, bias=False,
                             padding=kernel_size // 2, padding_mode='reflect')
    filter.weight.data = kernel
    filter.weight.requires_grad = False

    return filter

class SpatialFrequencyLoss(nn.Module):
    def __init__(self, num_channels=3, device='cuda', debug = False):
        super(SpatialFrequencyLoss, self).__init__()
        self.debug = debug

        self.sigmas = torch.tensor([0.6, 1.2, 2.4, 4.8, 9.6, 19.2]).to(device)
        self.w_sfl = torch.tensor([600, 500, 400, 20, 10, 10]).to(device)
        self.num_filters = len(self.sigmas)

        self.filters = []
        for x in range(self.num_filters):
            filter = get_LoG_filter(num_channels, self.sigmas[x], device)
            self.filters.append(filter)
    def forward(self, input, target):
        if input.ndim == 5:
            input = input.flatten(0, 1)
            target = target.flatten(0, 1)
        loss = 0.
        for x in range(self.num_filters):
            input_LoG = self.filters[x](input)
            target_LoG = self.filters[x](target)
            loss += self.w_sfl[x] * F.mse_loss(input_LoG, target_LoG)
        return loss

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_G(sh_num, is_res=False, is_video=False, init_type='normal', init_gain=0.02, Net="Unet"):
    net = VideoRelight(sh_num=sh_num, is_video=is_video) if not is_res else ShadingRes(is_video=is_video)
    return init_net(net, init_type, init_gain)

if __name__ == "__main__":
    # a = torch.randn([1, 3, 3, 512, 512]).cuda()
    a = torch.randn([4, 3, 512, 512]).cuda()
    model = VideoRelight(sh_num=25, is_video=False).cuda()
    model_res = ShadingRes(is_video=False).cuda()
    optim = torch.optim.Adam(model.parameters(),lr=0.0001, betas=(0.5, 0.999))
    optim_res = torch.optim.Adam(model_res.parameters(),lr=0.0001, betas=(0.5, 0.999))
    t_d, t_s, albedo, light, res_groups = model(a)
    shading_all_hat_linear = torch.einsum('bchw,bcd->bdhw', t_d, light)
    l2srgb = lambda x: torch.pow(x.clamp_min_(0.) + 1e-8, 1/2.2)
    shading_all_hat = l2srgb(shading_all_hat_linear)
    shading_res = model_res(torch.cat([shading_all_hat.detach(), 
                                t_d.detach()], 1), light.detach())
    # sepc_all_hat = l2srgb(torch.einsum('bfchw,bfcd->bfdhw', t_s, light))
    # diffuse = albedo * shading_all_hat
    # rendering = diffuse + albedo * sepc_all_hat
    # shading_final = l2srgb(shading_all_hat_linear + shading_res)
    # diffuse_final = shading_final * albedo
    # rendering_final = diffuse_final + albedo * sepc_all_hat
    # loss = 0.
    # loss_l1 = nn.L1Loss().cuda()
    # for res in res_groups:
    #     if res is not None:
    #         loss += loss_l1(res, torch.empty_like(res).fill_(0.))
    # loss += loss_l1(shading_final, shading_all_hat)
    # loss += loss_l1(diffuse, diffuse_final)
    # loss += loss_l1(rendering, rendering_final)
    # optim.zero_grad()  # set G_A and G_B's gradients to zero
    # optim_res.zero_grad()
    # loss.backward()             # calculate gradients for G_A and G_B
    # optim_res.step()
    # optim.step()
    print(1)