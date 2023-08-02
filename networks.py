import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as transforms
from antialiased_cnns.blurpool import BlurPool
import math

train_baseline = True

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

class CNNAE2ResNet(nn.Module):

    def __init__(self,in_channels=3,albedo_decoder_channels=4,train=True, sh_num=25):
        super(CNNAE2ResNet,self).__init__()
        self.sh_num = sh_num
        self.c0 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1) # 1024 -> 512
        nn.init.normal_(self.c0.weight, 0.0, 0.02)
        self.c1 = nn.Conv2d(64, 128, 4, stride=2, padding=1,bias=False)  # 512 -> 256
        nn.init.normal_(self.c1.weight, 0.0, 0.02)
        self.c2 = nn.Conv2d(128, 256, 4, stride=2, padding=1,bias=False) # 256 -> 128
        nn.init.normal_(self.c2.weight, 0.0, 0.02)
        self.c3 = nn.Conv2d(256, 512, 4, stride=2, padding=1,bias=False) # 128 -> 64
        nn.init.normal_(self.c3.weight, 0.0, 0.02)
        self.c4 = nn.Conv2d(512, 512, 4, stride=2, padding=1,bias=False) # 64 -> 32
        nn.init.normal_(self.c4.weight, 0.0, 0.02)

        self.ra = ResidualBlock(512, 512)
        self.rb = ResidualBlock(512, 512)

        self.dc0a = nn.Conv2d(512,512,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc0a.weight, 0.0, 0.02)
        self.up0a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dc1a = nn.Conv2d(1024, 512,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc1a.weight, 0.0, 0.02)
        self.up1a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc2a = nn.Conv2d(256*3, 256,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc2a.weight, 0.0, 0.02)
        self.up2a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc3a = nn.Conv2d(128*3, 128,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc3a.weight, 0.0, 0.02)
        self.up3a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dc3a1 = nn.Conv2d(128*3, 128,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc3a1.weight, 0.0, 0.02)
        self.up3a1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc4a = nn.Conv2d(64*3, 64,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc4a.weight, 0.0, 0.02)
        self.up4a = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dc4a1 = nn.Conv2d(64*3, 64,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc4a1.weight, 0.0, 0.02)
        self.up4a1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.trans = nn.Conv2d(64, sh_num,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.trans.weight, 0.0, 0.02)
        self.trans1 = nn.Conv2d(64, sh_num,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.trans1.weight, 0.0, 0.02)

        self.dc0b = nn.Conv2d(512, 512,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc0b.weight, 0.0, 0.02)
        self.up0b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dc1b = nn.Conv2d(1024, 512,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc1b.weight, 0.0, 0.02)
        self.up1b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dc2b = nn.Conv2d(256*3, 256,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc2b.weight, 0.0, 0.02)
        self.up2b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dc3b = nn.Conv2d(128*3, 128,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc3b.weight, 0.0, 0.02)
        self.up3b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dc4b = nn.Conv2d(64*3, 64,kernel_size=3,padding=1,bias=False)
        nn.init.normal_(self.dc4b.weight, 0.0, 0.02)
        self.up4b = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.albedo = nn.Conv2d(64, albedo_decoder_channels,kernel_size=3,padding=1)
        nn.init.normal_(self.albedo.weight, 0.0, 0.02)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.c0l = nn.Conv2d(512*3, 512, 4, stride=2, padding=1,bias=False) # 16 -> 8
        nn.init.normal_(self.c0l.weight, 0.0, 0.02)
        self.c1l = nn.Conv2d(512, 256, 4, stride=2, padding=1,bias=False) # 8 -> 4
        nn.init.normal_(self.c1l.weight, 0.0, 0.02)
        self.c2l = nn.Conv2d(256, 128, 4, stride=2, padding=1,bias=False) # 8 -> 4
        nn.init.normal_(self.c2l.weight, 0.0, 0.02)
        self.c3l = nn.Conv2d(128, sh_num * 3, 4, stride=2, padding=1) # 2 -> 1
        nn.init.normal_(self.c3l.weight, 0.0, 0.02)
            
        self.bnc1 = nn.BatchNorm2d(128)
        self.bnc2 = nn.BatchNorm2d(256)
        self.bnc3 = nn.BatchNorm2d(512)
        self.bnc4 = nn.BatchNorm2d(512)
        self.bnc5 = nn.BatchNorm2d(512)

        self.bndc0a = nn.BatchNorm2d(512)
        self.bndc1a = nn.BatchNorm2d(512)
        self.bndc2a = nn.BatchNorm2d(256)
        self.bndc3a = nn.BatchNorm2d(128)
        self.bndc3a1 = nn.BatchNorm2d(128)
        self.bndc4a = nn.BatchNorm2d(64)
        self.bndc4a1 = nn.BatchNorm2d(64)

        self.bndc0b = nn.BatchNorm2d(512)
        self.bndc1b = nn.BatchNorm2d(512)
        self.bndc2b = nn.BatchNorm2d(256)
        self.bndc3b = nn.BatchNorm2d(128)
        self.bndc4b = nn.BatchNorm2d(64)
            
        self.bnc0l = nn.BatchNorm2d(512)
        self.bnc1l = nn.BatchNorm2d(256)
        self.bnc2l = nn.BatchNorm2d(128)
        
        self.train_dropout = train


    def forward(self, xi):
        hc0 = F.leaky_relu(self.c0(xi),inplace=True) # 64*256*256
        hc1 = F.leaky_relu(self.bnc1(self.c1(hc0)),inplace=True) # 128*128*128
        hc2 = F.leaky_relu(self.bnc2(self.c2(hc1)),inplace=True) # 256*64*64
        hc3 = F.leaky_relu(self.bnc3(self.c3(hc2)),inplace=True) # 512*32*32
        hc4 = F.leaky_relu(self.bnc4(self.c4(hc3)),inplace=True) # 512*16*16

        # import ipdb; ipdb.set_trace()
        if train_baseline == True:
            hra = self.ra(hc4) # 512*16*16

            ha = self.up0a(F.relu(F.dropout(self.bndc0a(self.dc0a(hra)), 0.5, training=self.train_dropout),inplace=True))
            # till this line: 512*32*32
            ha = torch.cat((ha,hc3),1)
            ha = self.up1a(F.relu(F.dropout(self.bndc1a(self.dc1a(ha)), 0.5, training=self.train_dropout),inplace=True))
            # till this line: 512*64*64
            ha = torch.cat((ha,hc2),1)
            ha = self.up2a(F.relu(F.dropout(self.bndc2a(self.dc2a(ha)), 0.5, training=self.train_dropout),inplace=True))
            # till this line: 256*128*128
            ha_inter = torch.cat((ha,hc1),1)
            ha = self.up3a(F.relu(self.bndc3a(self.dc3a(ha_inter)),inplace=True))
            ha1 = self.up3a1(F.relu(self.bndc3a1(self.dc3a1(ha_inter)),inplace=True))
            # till this line: 128*256*256
            ha = torch.cat((ha,hc0),1)
            ha1 = torch.cat((ha1,hc0),1)
            ha = self.up4a(F.relu(self.bndc4a(self.dc4a(ha)),inplace=True))
            ha1 = self.up4a1(F.relu(self.bndc4a1(self.dc4a1(ha1)),inplace=True))
            # till this line: 64*512*512
            ha = self.trans(ha) # sh_num*512*512
            ha1 = self.trans1(ha1) # sh_num*512*512

        hrb = self.rb(hc4)
        hb = self.up0b(F.relu(F.dropout(self.bndc0b(self.dc0b(hrb)), 0.5, training=self.train_dropout),inplace=True))
        hb = torch.cat((hb,hc3),1)
        hb = self.up1b(F.relu(F.dropout(self.bndc1b(self.dc1b(hb)), 0.5, training=self.train_dropout),inplace=True))
        hb = torch.cat((hb,hc2),1)
        hb = self.up2b(F.relu(F.dropout(self.bndc2b(self.dc2b(hb)), 0.5, training=self.train_dropout),inplace=True))
        hb = torch.cat((hb,hc1),1)
        hb = self.up3b(F.relu(self.bndc3b(self.dc3b(hb)),inplace=True))
        hb = torch.cat((hb,hc0),1)
        hb = self.up4b(F.relu(self.bndc4b(self.dc4b(hb)),inplace=True))
        hb = self.albedo(hb)
        
        if train_baseline == True:
            hb = self.sig(hb)
        else:
            pass
        
        if train_baseline == True:
            hc = torch.cat((hc4, hra, hrb),1)
            hc = F.leaky_relu(self.bnc0l(self.c0l(hc)),inplace=True) # 512*8*8
            hc = F.leaky_relu(self.bnc1l(self.c1l(hc)),inplace=True) # 256*4*4
            hc = F.leaky_relu(self.bnc2l(self.c2l(hc)),inplace=True) # 128*2*2
            hc = torch.reshape(self.c3l(hc), (-1, self.sh_num, 3))
            
        # import ipdb; ipdb.set_trace()
        if train_baseline == True:
            return ha, ha1, hb, hc
        else:
            return hb

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

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
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

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, local_rank, init_type='normal', init_gain=0.02):
    device = torch.device('cuda', local_rank)
    net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, local_rank, normal=False, init_type='normal', init_gain=0.02, Net="Unet"):
    if Net == 'Unet':
        net = UnetGenerator(input_nc, output_nc, ngf, normal=normal)
    elif Net == "Res":
        net = ResnetGenerator(input_nc, output_nc, ngf, use_dropout=False)
    else:
        net = specular(input_nc, output_nc, ngf)
    return init_net(net, local_rank, init_type, init_gain)


def define_D(input_nc, ngf, local_rank, init_type='normal', init_gain=0.02):
    net = None
    net = NLayerDiscriminator(input_nc, ngf)
    return init_net(net, local_rank, init_type, init_gain)
