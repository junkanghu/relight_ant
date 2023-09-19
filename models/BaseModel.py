import os
import torch
from abc import ABC, abstractmethod
from utils import ssim, psnr
import torchvision
import numpy as np
from PIL import Image
from torch.cuda.amp import GradScaler
from video_utils import VideoWriter, VideoComposition


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
        self.light = None
        self.epoch = 1
        self.loss_all = {}
        self.net_name = []
        self.optimizer_name = []
        self.ssim = 0
        self.ssim_albedo = 0
        self.psnr = 0
        self.psnr_albedo = 0
        self.batch_num = 0
        self.scaler = GradScaler(enabled=opt.amp)
        if opt.video:
            os.makedirs(os.path.join(opt.out_dir, 'val_videos'), exist_ok=True)
            if opt.use_video_writer:
                self.video_writer_albedo = VideoWriter(
                    path=os.path.join(opt.out_dir, 'val_videos', f'{self.epoch}_{os.path.basename(opt.video_val_dir)}_albedo.mp4'),
                    frame_rate=30,
                    bit_rate=int(1000000))
                self.video_writer_rendering = VideoWriter(
                    path=os.path.join(opt.out_dir, 'val_videos', f'{self.epoch}_{os.path.basename(opt.video_val_dir)}_rendering.mp4'),
                    frame_rate=30,
                    bit_rate=int(1000000))
            else:
                self.video_writer = VideoComposition(path=os.path.join(opt.out_dir, 'val_videos'), format="'*[0-9][0-9][0-9][0-9][0-9][0-9].png'",
                    out_path=os.path.join(opt.out_dir, 'val_videos'), out_name=f'{os.path.basename(opt.video_val_dir)}.mp4')

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

    def plot_video(self, epoch=1):
        if self.opt.use_video_writer:
            self.video_writer_albedo.write(self.albedo_hat[0])
            self.video_writer_rendering.write(self.rendering[0])
        else:
            for id in range(self.albedo_hat.shape[1]):
                cated = torch.cat([self.albedo_hat[0][id:id+1],
                                # self.albedo[0][id:id+1],
                                self.shading_all_hat[0][id:id+1],
                                # self.shading[0][id:id+1],
                                self.sepc_all_hat[0][id:id+1],
                                # self.prt_s[0][id:id+1],
                                self.diffuse[0][id:id+1],
                                # self.prt_d[0][id:id+1],
                                self.rendering[0][id:id+1],
                                ((self.input[0][id:id+1] * 0.5 + 0.5) * self.mask[0][id:id+1]).detach()], dim=0)
                out = torchvision.utils.make_grid(cated,
                                            scale_each=False,
                                            normalize=False,
                                            nrow=3)
                img = self.tensor2image(out)
                self.video_writer.plot(img, self.name[id][0])
    
    def tensor2image(self, img):
        img = img.detach().cpu().numpy()
        img = img.transpose(1, 2, 0)
        img = np.clip(img, 0.01, 0.99)
        scale_factor = 255
        img = (img * scale_factor).astype(np.uint8)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, -1)
        img = Image.fromarray(img)
        return img

    def plot_image(self, epoch=1):
        # shape of output_vs_gt_plot [B, C, H, W]
        for id in range(self.albedo_hat.shape[0]):
            output_vs_gt_plot = self.concat_img(id)
            out = torchvision.utils.make_grid(output_vs_gt_plot,
                                            scale_each=False,
                                            normalize=False,
                                            nrow=4 if self.opt.test else 5)
            
            img = self.tensor2image(out)
            img_path = os.path.join(self.opt.out_dir, "val_imgs")
            if self.opt.test:
                img_path = os.path.join(self.opt.out_dir, "test_imgs") 
            os.makedirs(img_path, exist_ok=True)
            img_dir = os.path.join(img_path, "{0}_{1}".format(epoch, self.name[id].split('.')[0] + '.png'))
            if self.opt.test:
                img_dir = os.path.join(img_path, "{}".format(self.name[id].split('.')[0] + '.png'))
            print('saving rendered img to {}'.format(img_dir))
            img.save(img_dir)
            
        if not self.opt.test:

            ssim_batch = ssim(((self.input * 0.5 + 0.5) * self.mask).detach(),
                            self.rendering.detach() if not self.opt.use_res else self.rendering_final.detach())
            psnr_batch = psnr(((self.input * 0.5 + 0.5) * self.mask).detach(),
                            self.rendering.detach() if not self.opt.use_res else self.rendering_final.detach())
            
            ssim_batch_albedo = ssim(self.albedo.detach(), self.albedo_hat.detach())
            psnr_batch_albedo = psnr(self.albedo.detach(), self.albedo_hat.detach())
            return ssim_batch, psnr_batch, ssim_batch_albedo, psnr_batch_albedo

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
                
    def cal_metric(self, ssim_batch, psnr_batch, ssim_batch_albedo, psnr_batch_albedo):
        self.ssim += ssim_batch
        self.psnr += psnr_batch
        self.ssim_albedo += ssim_batch_albedo
        self.psnr_albedo += psnr_batch_albedo
        self.batch_num += 1
        
    def reset_metric(self):
        self.ssim = 0
        self.psnr = 0
        self.ssim_albedo = 0
        self.psnr_albedo = 0
        self.batch_num = 0
        
    def print_metric(self):
        ssim = self.ssim / self.batch_num
        psnr = self.psnr / self.batch_num
        ssim_albedo = self.ssim_albedo / self.batch_num
        psnr_albedo = self.psnr_albedo / self.batch_num
        self.reset_metric() # After validation of current epoch, reset the metrics.
        print("SSIM: %.3f" % ssim)
        print("PSNR: %.3f" % psnr)
        print("SSIM-albedo: %.3f" % ssim_albedo)
        print("PSNR-albedo: %.3f" % psnr_albedo)
        return ssim, psnr, ssim_albedo, psnr_albedo
    
    def val_imgs(self, epoch):       
        with torch.no_grad():
            self.forward()
            ssim_batch, psnr_batch, ssim_batch_albedo, psnr_batch_albedo = self.plot_image(epoch)
            self.cal_metric(ssim_batch, psnr_batch, ssim_batch_albedo, psnr_batch_albedo)
            
    def val_videos(self, epoch):
        with torch.no_grad():
            self.forward()
            self.plot_video(epoch)
            
    def val(self, epoch):
        if self.opt.video:
            self.val_videos(epoch)
        else:
            self.val_imgs(epoch)
        
    def test(self):
        """Process the test data with the trained model.

        Args:
            end (bool, optional): In the end, print metrics. Defaults to False.
        """           
        with torch.no_grad():
            self.eval()
            self.forward()
            # ssim_batch, psnr_batch = self.plot_image()
            self.plot_image()
            # self.cal_metric(ssim_batch, psnr_batch)

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
        if self.opt.tune:
            start_epoch = 1
        for name in self.optimizer_name:
            if isinstance(name, str) and name in list(ckpt.keys()):
                optimizer = getattr(self, name)
                if not len(ckpt[name]['param_groups'][0]['params']) == len(optimizer.state_dict()['param_groups'][0]['params']):
                    continue
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
            
    def get_print_format(self, epoch, writer, n_batch):
        # print losses and add them to writer
        loss = self.gather_loss(True)
        out = "[Epoch {} ]".format(epoch)
        for idx, key in enumerate(loss.keys()):
            average = loss[key] / n_batch
            writer.add_scalar(key, average, epoch)
            out += (key + ": " + str(average) + ("\n" if idx == len(loss.keys()) - 1 else " "))
        return out
