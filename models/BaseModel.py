import os
import torch
from abc import ABC, abstractmethod
from utils import ssim, psnr
import torchvision
import numpy as np
from PIL import Image


class BaseModel(torch.nn.Module):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        super(BaseModel, self).__init__()
        self.opt = opt
        self.device = torch.device("cuda", opt.local_rank)
        self.name = None # Validation or test image name.
        self.loss_name = None
        self.loss_res = None
        self.epoch = 1
        self.loss_all = {}
        self.net_name = []
        self.optimizer_name = []
        self.ssim = []
        self.psnr = []

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass
    
    @abstractmethod
    def concat_img(self, batch_id):
        pass
    
    def plot(self, epoch=1):
        # shape of output_vs_gt_plot [B, C, H, W]
        psnr_group = []
        for id in range(self.albedo_hat.shape[0]):
            output_vs_gt_plot = self.concat_img(id)
            out = torchvision.utils.make_grid(output_vs_gt_plot,
                                            scale_each=False,
                                            normalize=False,
                                            nrow=4 if self.opt.test else 5).detach().cpu().numpy()
            out = out.transpose(1, 2, 0)
            out = np.clip(out, 0.01, 0.99)
            scale_factor = 255
            tensor = (out * scale_factor).astype(np.uint8)
            img = Image.fromarray(tensor)
            img_path = os.path.join(self.opt.out_dir, "val_imgs")
            if self.opt.test:
                img_path = os.path.join(self.opt.out_dir, "test_imgs") 
            os.makedirs(img_path, exist_ok=True)
            img_dir = os.path.join(img_path, "{0}_{1}".format(epoch, self.name[id].split('.')[0] + '.png'))
            if self.opt.test:
                img_dir = os.path.join(img_path, "{}".format(self.name[id].split('.')[0] + '.png'))
            print('saving rendered img to {}'.format(img_dir))
            img.save(img_dir)

            # calculate ssim and psnr
            psnr_group.append(psnr(((self.input[id] * 0.5 + 0.5) * self.mask[id]).detach(), self.rendering[id].detach()))
        ssim_batch = [ssim(((self.input * 0.5 + 0.5) * self.mask).detach(), self.rendering.detach())]
        return ssim_batch, psnr_group

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.net_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()
    
    def train(self):
        """Make models eval mode during test time"""
        for name in self.net_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.train()
                
    def cal_metric(self, ssim_batch, psnr_batch):
        self.ssim += ssim_batch
        self.psnr += psnr_batch
        
    def print_metric(self):
        ssim = sum(self.ssim) / len(self.ssim)
        psnr = sum(self.psnr) / len(self.psnr)
        print("SSIM: %.3f" % ssim)
        print("PSNR: %.3f" % psnr)
        
    def test(self):
        """Process the test data with the trained model.

        Args:
            end (bool, optional): In the end, print metrics. Defaults to False.
        """           
        with torch.no_grad():
            self.forward()
            ssim_batch, psnr_batch = self.plot()
            self.cal_metric(ssim_batch, psnr_batch)

    def update_lr(self, epoch):
        decay_epoch = self.opt.epochs // 2
        lr = self.opt.lr * (1. - 1. * max(epoch - decay_epoch, 0) / (decay_epoch + 10))
        print("learning rate:", lr)
        for name in self.optimizer_name:
            optimizer = getattr(self, name)
            for param in optimizer.param_groups:
                param['lr'] = lr

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
        
    def load_ckpt(self):
        os.makedirs(self.opt.out_dir, exist_ok=True)
        start_epoch = 1
        ckpt_dir = os.path.join(self.opt.out_dir, "latest.pth")
        if not os.path.exists(ckpt_dir) or self.opt.discard_ckpt:
            print('No checkpoints!')
            return start_epoch

        print("Loading checkpoint:", ckpt_dir)
        ckpt = torch.load(ckpt_dir, map_location=self.device)
        
        for name in self.net_name:
            if isinstance(name, str) and name in list(ckpt.keys()):
                net = getattr(self, 'net_' + name)
                weight = ckpt[name]
                cur_weight = net.state_dict()
                if len(list(cur_weight.keys())) == len(list(weight.keys())):
                    net.load_state_dict(weight)
                    continue
                new_model_state_dict = {}
                for key, value in cur_weight.items():
                    new_model_state_dict[key] = value
                for key, value in weight.items():
                    new_model_state_dict[key] = value
                net.load_state_dict(new_model_state_dict)

        start_epoch = ckpt["epoch"] + 1
        for name in self.optimizer_name:
            if isinstance(name, str) and name in list(ckpt.keys()):
                optimizer = getattr(self, name)
                optimizer.load_state_dict(ckpt[name])
        return start_epoch

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.net_name:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

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
                    
    def initialize_loss(self):
        for idx, name in enumerate(self.loss_name):
            self.loss_all[name] = 0.
            
    def gather_loss(self, end_of_epoch=False):
        if end_of_epoch:
            loss_all = self.loss_all.copy()
            self.initialize_loss()
            return loss_all

        for idx, name in enumerate(self.loss_name):
            if self.epoch < self.opt.res_epoch and name == 'res':
                continue
            value = getattr(self, 'loss_' + name).item()
            self.loss_all[name] += value
