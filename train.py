from dataset import create_dataset
from option import get_opt
from model import lumos
from tqdm import tqdm
from utils import init_distributed_mode
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import os

if __name__ == "__main__":
    opt = get_opt()
    init_distributed_mode(opt)
    logging.basicConfig(filename=opt.log_dir, filemode='a', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    dataloader_train, dataloader_val = create_dataset(opt)
    model = lumos(opt).to(torch.device("cuda", opt.local_rank))

    # configure ddp
    get_model = lambda model: model.module if opt.distributed else model
    if opt.distributed:
        # replace bn with synbn which uses data from all processes to determine the bn parameters.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
        logging.info(f"WorldSize {os.environ['WORLD_SIZE']} Rank {os.environ['RANK']} Local_Rank {os.environ['LOCAL_RANK']}")
        if dist.get_rank() == 0:
            writer = SummaryWriter(opt.out_dir)
    else:
        writer = SummaryWriter(opt.out_dir)
    start_epoch = get_model(model).load_ckpt()
    for epoch in tqdm(range(start_epoch, opt.epochs + 1), ascii=True, desc='epoch'):
        if opt.distributed:
            dataloader_train.sampler.set_epoch(epoch) # Used for data shuffling. If not set, no shuffling.
        for i, data in tqdm(enumerate(dataloader_train), ascii=True, desc='training iterations'):
            get_model(model).set_input(data)
            get_model(model).optimize_parameters()
            if (opt.distributed and dist.get_rank() == 0) or not opt.distributed:
                get_model(model).gather_loss()

        get_model(model).update_lr(epoch)
        if (opt.distributed and dist.get_rank() == 0) or not opt.distributed:
            # print losses and add them to writer
            loss = get_model(model).gather_loss(True)
            out = "[Epoch {} ]".format(epoch)
            for idx, key in enumerate(loss.keys()):
                average = loss[key] / (i + 1)
                writer.add_scalar(key, average, epoch)
                out += (key + ": " + str(average) + ("\n" if idx == len(loss.keys()) - 1 else " "))
            tqdm.write(out)
            logging.info(out)
            
            # save ckpt
            if epoch % opt.save_fre == 0:
                get_model(model).save_ckpt(epoch)

        if epoch % opt.val_fre == 0:
            ssim = []
            psnr = []
            for i, data in enumerate(dataloader_val):
                with torch.no_grad():
                    get_model(model).set_input(data, val=True)
                    get_model(model).eval()
                    get_model(model).forward(val=True)
                    ssim_batch, psnr_batch = get_model(model).plot_val(epoch)
                    get_model(model).train()
                    ssim += ssim_batch
                    psnr += psnr_batch
            
            if (opt.distributed and dist.get_rank() == 0) or not opt.distributed:
                ssim = sum(ssim) / len(ssim)
                psnr = sum(psnr) / len(psnr)
                writer.add_scalar('ssim', ssim, epoch)
                writer.add_scalar('psnr', psnr, epoch)