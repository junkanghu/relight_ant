import os
import yaml
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import logging
from torch.utils.tensorboard import SummaryWriter
import datetime

def workspace_config(opt):
    workspace = os.getenv('workspace')
    if workspace is not None:
        os.makedirs(workspace, exist_ok=True)
        opt.out_dir = os.path.join(workspace, opt.out_dir)
        opt.log_dir = os.path.join(workspace, opt.log_dir)
    return opt

def train_config(opt):
    yaml_path = opt.config
    with open(yaml_path, 'r') as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
    for key in info.keys():
        value = info[key]
        setattr(opt, key, value)
    return opt

def init_distributed_mode(args):
    """ init for distribute mode """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    '''
    This is commented due to the stupid icoding pylint checking.
    print('distributed init rank {}: {}'.format(args.rank, args.dist_url), flush=True)
    '''
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=18000))
    torch.distributed.barrier()

    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)

def init(opt):
    init_distributed_mode(opt)
    logging.basicConfig(filename=opt.log_dir, filemode='a', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    if opt.rank == 0:
        writer = SummaryWriter(opt.out_dir)
        return writer
    return None

@torch.no_grad()
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

@torch.no_grad()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

@torch.no_grad()
def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

@torch.no_grad()
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

@torch.no_grad()
def psnr(img1, img2):
    """
    img shape: [B, C, H, W]
    """
    img1 = img1.cpu()
    img2 = img2.cpu()
    b, c, h, w = img1.shape
    mse = torch.sum(torch.pow(img1 - img2, 2)) / (b * c * h * w)
    mse2psnr = lambda x: -10. * torch.log(x + 1e-8) / torch.log(torch.FloatTensor([10.]))
    return mse2psnr(mse).item()