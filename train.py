from data.TrainDataset import create_dataset
from option import get_opt
from models.TrainModel import lumos
from tqdm import tqdm
from utils import init
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import os

if __name__ == "__main__":
    opt = get_opt()
    writer = init(opt)
    dataloader_train, dataloader_val = create_dataset(opt)
    model = lumos(opt).to(torch.device("cuda", opt.local_rank))

    # configure ddp
    get_model = lambda model: model.module if opt.distributed else model
    if opt.distributed:
        # replace bn with synbn which uses data from all processes to determine the bn parameters.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)
        logging.info(f"WorldSize {os.environ['WORLD_SIZE']} Rank {os.environ['RANK']} Local_Rank {os.environ['LOCAL_RANK']}")
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
            get_model(model).eval()
            for i, data in enumerate(dataloader_val):
                get_model(model).set_input(data, val=True)
                get_model(model).val(epoch)

            if opt.distributed:
                torch.distributed.barrier() # Synchronize the processes to ensure all images are obtained before ffmpeg runs.
            if opt.rank == 0:
                if opt.video:
                    if opt.use_video_writer:
                        get_model(model).video_writer_albedo.close()
                        get_model(model).video_writer_rendering.close()
                    else:
                        get_model(model).video_writer.composite(epoch)
                else:
                    if opt.cal_metric:
                        ssim, psnr, ssim_albedo, psnr_albedo = get_model(model).print_metric()
                        writer.add_scalar('ssim', ssim, epoch)
                        writer.add_scalar('psnr', psnr, epoch)
                        writer.add_scalar('ssim_albedo', ssim_albedo, epoch)
                        writer.add_scalar('psnr_albedo', psnr_albedo, epoch)
                        logging.info(f'ssim: {ssim} psnr: {psnr} ssim-albedo: {ssim_albedo} psnr-albedo: {psnr_albedo}')
            get_model(model).train()