from data.TrainDataset import create_dataset
from option import get_opt
from models.TrainModel import lumos
from tqdm import tqdm
from utils import init_distributed_mode
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import os

if __name__ == "__main__":
    # torch.backends.cudnn.benchmark = True
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
    if opt.rank == 0:
        writer = SummaryWriter(opt.out_dir)
    start_epoch = get_model(model).load_ckpt()
    for epoch in tqdm(range(start_epoch, opt.epochs + 1), ascii=True, desc='epoch'):
        if opt.distributed:
            dataloader_train.sampler.set_epoch(epoch) # Used for data shuffling. If not set, no shuffling.
        for i, data in tqdm(enumerate(dataloader_train), ascii=True, desc='training iterations'):
            get_model(model).set_input(data)
            get_model(model).optimize_parameters(epoch)
            if opt.rank == 0:
                get_model(model).gather_loss()

        get_model(model).update_lr(epoch)
        if opt.rank == 0:
            out = get_model(model).get_print_format(epoch, writer, i + 1)
            tqdm.write(out)
            logging.info(out)
            if epoch % opt.save_fre == 0: # save ckpt
                get_model(model).save_ckpt(epoch)

        if epoch % opt.val_fre == 0:
            for i, data in enumerate(dataloader_val):
                get_model(model).set_input(data)
                get_model(model).val(epoch)
            
            if opt.rank == 0:
                ssim, psnr = get_model(model).print_metric()
                writer.add_scalar('ssim', ssim, epoch)
                writer.add_scalar('psnr', psnr, epoch)