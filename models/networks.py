import torch
import torch.nn as nn
import functools
import numpy as np
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as transforms
from antialiased_cnns.blurpool import BlurPool
import math
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
                light_dims=[512*3, 512, 256, 128], sh_num=25, use_res=False):
        super().__init__()
        self.use_res = use_res
        self.encoder = Encoder(encoder_dims)
        self.decoder = Decoder(decoder_dims, skip_dims, light_dims, sh_num)
        if use_res:
            self.shading_res = S_Res()
        
    def forward(self, x):
        x = self.encoder(x)
        t_d, t_s, albedo, light = self.decoder(x)
        if not self.use_res:
            return t_d, t_s, albedo, light
        res = self.shading_res(t_d, light)
        return t_d, t_s, albedo, light, res
    
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
    def __init__(self, dims:list):
        """
        Args:
            dims (list): Features dimensions. The first is the input one.
        """        
        super().__init__()
        self.n_layer = len(dims) - 1
        for i in range(self.n_layer):
            setattr(self, "e{}".format(i),
                    ConvBlock(dims[i], dims[i+1], 4, 2, 1,
                                norm="batch" if i > 0 else None, ac="leakyrelu"))
        
    def forward(self, x):
        out = []
        for i in range(self.n_layer):
            block = getattr(self, "e{}".format(i))
            x = block(x)
            out.append(x)
        return out

class UpsamplingBlock(nn.Module):
    def __init__(self, in_nc, skin_nc, out_nc,
                kernel_size, stride, padding, bias=False,
                norm=None, ac=None, dropout=False, drop_p=0.5, lr_slope=0.01):
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
    
    def forward(self, x1, x2=None):
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
        return x1

class Decoder_Albedo(nn.Module):
    def __init__(self, decoder_dims, skip_dims):
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
                    norm="batch", ac="relu", dropout=True if i < 2 else False))
        self.out = nn.Sequential(nn.Conv2d(decoder_dims[-1], 3, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.Sigmoid())
        
    def forward(self, x):
        bottle = self.bottlneck(x[-1])
        for i in range(self.n_layer):
            layer = getattr(self, "up{}".format(i))
            inter = layer(bottle if i == 0 else inter, x[-(i + 2)] if not i == self.n_layer - 1 else None)
        inter = self.out(inter)
        return inter, bottle

class Decoder_Transport(nn.Module):
    def __init__(self, decoder_dims, skip_dims, sh_num):
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
                    norm="batch", ac="relu", dropout=True if i < 2 else False))
            if i > 2:
                setattr(self, "ups{}".format(i),
                    UpsamplingBlock(decoder_dims[i], skip_dims[i], decoder_dims[i+1], 3, 1, 1,
                        norm="batch", ac="relu", dropout=True if i < 2 else False))
        self.out_d = nn.Conv2d(decoder_dims[-1], sh_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.out_s = nn.Conv2d(decoder_dims[-1], sh_num, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        bottle = self.bottlneck(x[-1])
        for i in range(self.n_layer):
            if i > 2:
                layer_s = getattr(self, "ups{}".format(i))
                inter1 = layer_s(inter if i == 3 else inter1, x[-(i + 2)] if not i == self.n_layer - 1 else None)
            layer_d = getattr(self, "upd{}".format(i))
            inter = layer_d(bottle if i == 0 else inter, x[-(i + 2)] if not i == self.n_layer - 1 else None)
        inter = self.out_d(inter)
        inter1 = self.out_s(inter1)
        return inter, inter1, bottle
    
class Decoder_Light(nn.Module):
    def __init__(self, dims, sh_num):
        """Branch for transport.

        Args:
            dims (list): Dimensions of features in the decoder.
            sh_num (int): Number of SH parameters which is determined by the SH order.
        """        
        super().__init__()
        self.n_layer = len(dims) - 1
        self.sh_num = sh_num
        model = []
        for i in range(self.n_layer):
            model.append(ConvBlock(dims[i], dims[i+1], 4, 2, 1, norm="batch", ac="leakyrelu"))
        model.append(nn.Conv2d(dims[-1], sh_num * 3, 4, stride=2, padding=1)) # 2 -> 1
        self.model = nn.Sequential(*model)
        
    def forward(self, e, bt, ba):
        """
        Args:
            e: The features of the last layer of the encoder.
            ba: BottleNeck features of albedo Decoder.
            bt: BottleNeck features of transport Decoder.
        """
        x = torch.cat([e, bt, ba], 1)
        x = self.model(x)
        x = x.reshape(-1, self.sh_num, 3)
        return x

class Decoder(nn.Module):
    def __init__(self, decoder_dims=[512, 512, 256, 128, 64, 32],
                skip_dims=[512, 256, 128, 64, 0],
                light_dims=[512*3, 512, 256, 128], sh_num=25):
        super().__init__()
        self.decoder_albedo = Decoder_Albedo(decoder_dims, skip_dims)
        self.decoder_transport = Decoder_Transport(decoder_dims, skip_dims, sh_num)
        self.decoder_light = Decoder_Light(light_dims, sh_num)
        
    def forward(self, x):
        albedo, bottle_a = self.decoder_albedo(x)
        t_d, t_s, bottle_t = self.decoder_transport(x)
        light = self.decoder_light(x[-1], bottle_a, bottle_t)
        return t_d, t_s, albedo, light

class S_Res(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, ):
        return 

class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1, ksize=3):
        w = math.sqrt(2)
        super().__init__()

        self.c1=nn.Conv2d(n_in, n_out, ksize, stride=stride, padding = 1)
        nn.init.constant_(self.c1.weight, w)
        self.c2=nn.Conv2d(n_out, n_out, ksize, stride=1, padding = 1)
        nn.init.constant_(self.c2.weight, w)
        self.b1=nn.BatchNorm2d(n_out)
        self.b2=nn.BatchNorm2d(n_out)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return h + x

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=32, normal=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,      下采样的次数（每次对半）
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(UnetGenerator, self).__init__()
        self.normal = normal
        self.e1 = nn.Sequential(*[
            nn.Conv2d(input_nc, 32, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
        ])
        self.e2 = nn.Sequential(*[
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(64, stride=2)
        ])
        self.e3 = nn.Sequential(*[
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(128, stride=2)
        ])
        self.e4 = nn.Sequential(*[
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(256, stride=2)
        ])
        self.e5 = nn.Sequential(*[
            nn.Conv2d(256, 512, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(512, stride=2)
        ])
        self.e6 = nn.Sequential(*[
            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(512, stride=2)
        ])
        self.bottle = nn.Sequential(*[
            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True),
            BlurPool(512, stride=2)
        ])
        self.d1 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 512, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d2 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1024, 512, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d3 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(1024, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d4 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d5 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d6 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])


        if normal:
            self.d5_l = nn.Sequential(*[
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(256, 64, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True)
            ])
            self.d6_l = nn.Sequential(*[
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(128, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True)
            ])

            self.out = nn.Sequential(*[
                nn.Conv2d(64, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, output_nc // 2, 1, 1)
            ])
            self.out_l = nn.Sequential(*[
                nn.Conv2d(64, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, output_nc // 2, 1, 1)
            ])
        else:
            self.out = nn.Sequential(*[
                nn.Conv2d(32, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, output_nc, 1, 1),
                # nn.ReLU()
            ])


    def forward(self, x):
        o1 = self.e1(x)
        o2 = self.e2(o1)
        o3 = self.e3(o2)
        o4 = self.e4(o3)
        o5 = self.e5(o4)
        o6 = self.e6(o5)
        bottle = self.bottle(o6)
        d1 = self.d1(bottle)
        d2 = self.d2(torch.cat([o6, d1], 1))
        d3 = self.d3(torch.cat([o5, d2], 1))
        d4 = self.d4(torch.cat([o4, d3], 1))
        d5 = self.d5(torch.cat([o3, d4], 1))
        d6 = self.d6(torch.cat([o2, d5], 1))

        if self.normal:
            d5_l = self.d5_l(torch.cat([o3, d4], 1))
            d6_l = self.d6_l(torch.cat([o2, d5_l], 1))

            out = self.out(torch.cat([o1, d6], 1))
            out_l = self.out_l(torch.cat([o1, d6_l], 1))

            return torch.cat([out, out_l], 1)

        out = self.out(d6)

        return out


class specular(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=8, sh_num=25):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,      下采样的次数（每次对半）
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(specular, self).__init__()

        self.e1 = nn.Sequential(*[
            nn.Conv2d(input_nc, 8, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
        ])
        self.e2 = nn.Sequential(*[
            nn.Conv2d(8, 16, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(16, stride=2)
        ])
        self.e3 = nn.Sequential(*[
            nn.Conv2d(16, 32, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(32, stride=2)
        ])
        self.e4 = nn.Sequential(*[
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(64, stride=2)
        ])
        self.e5 = nn.Sequential(*[
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(128, stride=2)
        ])
        self.e6 = nn.Sequential(*[
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            # nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.2, True),
            BlurPool(256, stride=2)
        ])
        self.bottle = nn.Sequential(*[
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True),
            BlurPool(256, stride=2)
        ])
        self.fusion = nn.Sequential(*[
            nn.Conv2d(512, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True),
        ])
        self.d1 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d2 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d3 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d4 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d5 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 16, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.d6 = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 8, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        self.out = nn.Sequential(*[
            nn.Conv2d(16, 16, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, True)
        ])
        

        self.out1 = nn.Sequential(*[
            nn.Conv2d(16, 8, 3, 1, 1, bias=True),
            nn.Conv2d(8, output_nc, 1, 1),
            # nn.Softmax()
            # nn.ReLU(True)
            # nn.Sigmoid()
        ])

        self.fc_light = nn.Linear(25*3, 256)
        self.up_light = nn.Upsample(scale_factor=8)

    def forward(self, x, light):
        light = torch.flatten(light, 1).unsqueeze(-1).unsqueeze(-1)
        light = self.up_light(self.fc_light(light))
        o1 = self.e1(x)
        o2 = self.e2(o1)
        o3 = self.e3(o2)
        o4 = self.e4(o3)
        o5 = self.e5(o4)
        o6 = self.e6(o5)
        bottle = self.bottle(o6)
        bottle = self.fusion(torch.cat([bottle, light], 1))
        d1 = self.d1(bottle)
        d2 = self.d2(torch.cat([o6, d1], 1))
        d3 = self.d3(torch.cat([o5, d2], 1))
        d4 = self.d4(torch.cat([o4, d3], 1))
        d5 = self.d5(torch.cat([o3, d4], 1))
        d6 = self.d6(torch.cat([o2, d5], 1))
        out = self.out(torch.cat([o1, d6], 1))

        out = self.out1(out)

        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=32, norm='Batch', use_dropout=False, n_blocks=9,
                 padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0) # 若条件满足则继续往下运行，否则显示AssertError，停止往下运行。
        super(ResnetGenerator, self).__init__()
        use_bias = True
        norm_layer = nn.BatchNorm2d
        model = [nn.ReflectionPad2d(3),   # 上下左右均填充三行像素，最终就是行填充了6行，列填充了6列
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(1)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)]
        # model += [nn.ReLU(True)]
        model += [nn.LeakyReLU(0.7)]

        self.model = nn.Sequential(*model) # 将字典类的model转换为真正的可以运行的model

    def forward(self, x): 
        return self.model(x)

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


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a VGG discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        use_bias = True

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        self.linear = nn.Linear(ndf * nf_mult * 32 * 32, 1, bias=True)  # output 1 channel prediction map
        self.sigmoid = nn.Sigmoid()
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        out = self.model(input).reshape(input.shape[0], -1)
        logits = self.linear(out)
        sigmoid = self.sigmoid(logits)
        return logits, sigmoid


class GANLoss(nn.Module):  # 继承nn.Module时，需要用到super函数；数据集继承ABC时不用。
    def __init__(self, mask, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.mask = mask
        self.gan_type = gan_type.lower() # 全部转为小写
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val) # 返回与输入同样大小的tensor，其中所有值都置为self.real_label_val
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

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

def define_G(sh_num, use_res, init_type='normal', init_gain=0.02, Net="Unet"):
    net = VideoRelight(sh_num=sh_num, use_res=use_res)
    return init_net(net, init_type, init_gain)


def define_D(input_nc, ngf, local_rank, init_type='normal', init_gain=0.02):
    net = None
    net = NLayerDiscriminator(input_nc, ngf)
    return init_net(net, local_rank, init_type, init_gain)
