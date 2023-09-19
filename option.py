from utils import workspace_config, train_config
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    # model-independent options
    parser.add_argument('--config', type=str, default=None,
                        help='config file path')
    parser.add_argument('--discard_ckpt', action='store_true')
    parser.add_argument('--test', action='store_true')

    # debug
    parser.add_argument('--regular', action='store_true')
    parser.add_argument('--shading_vgg', action='store_true')
    parser.add_argument('--albedo_vgg', action='store_true')
    parser.add_argument('--tv', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--transport_sf', action='store_true')
    parser.add_argument('--ssim_albedo', action='store_true')
    parser.add_argument('--ssim_transport', action='store_true')
    parser.add_argument('--ssim_shading', action='store_true')
    parser.add_argument('--sil1', action='store_true')

    # DDP-related
    parser.add_argument('--dist_backend', default='nccl', help='which backend to use')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, help='whether to use distributed mode')
    parser.add_argument('--local_rank', type=int, default=0, help='gpu id for current process')
    parser.add_argument('--rank', type=int, default=0, help='rank id among all the world process')
    parser.add_argument('--world_size', type=int, default=1, help='total number of ddp processes')
    parser.add_argument('--data_load_works', type=int, default=4, help='dataloader workers for loading data')

    # Dir-related
    parser.add_argument('--test_dir', type=str, 
                        help='Test data directory')
    parser.add_argument('--test_light_dir', type=str,
                        help='Test data directory')
    parser.add_argument('--video_dir', type=str, default='/nas/home/hujunkang/tiktok_resized', 
                        help='Video data directory')
    parser.add_argument('--video_val_dir', type=str, default='/nas/home/hujunkang/wild_pose_dataset/_cRYAuM4pXI+003100+005700', 
                        help='Video validation data directory')
    parser.add_argument('--out_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--log_dir', type=str, default='train.log',
                        help='log directory')
    parser.add_argument('--yaml_dir', default='/nas/home/hujunkang/data.yaml', help='Directory for test input images')
    parser.add_argument('--light_dir', '-l0', default="/nas/home/hujunkang/sh_hdr", help='Light directory for training')
    parser.add_argument('--save_fre', type=int, default=5)
    parser.add_argument('--val_fre', type=int, default=1)

    # Base config
    parser.add_argument('--epochs', type=int, default=200,
                        help='Epoch for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training')
    parser.add_argument('--amp', action="store_true", help="whether use torch amp")

    # Model-dependent options
    parser.add_argument('--use_tfa', action="store_true", help='whether to use temporal feature alignment')
    parser.add_argument('--use_video_writer', action="store_true", help='whether to use av writer or ffmpeg')
    parser.add_argument('--tfa_att', action="store_true", help='Feature alignment algorithms. Default is deform conv.')
    parser.add_argument('--video', action="store_true", help='train for video stage')
    parser.add_argument('--res_epoch', type=int, default=30,
                        help='when to start training the residual net')
    parser.add_argument('--use_res', action="store_true", help="whether to use shading residual")
    parser.add_argument('--sh_num', default=25, type=int, help='number of sh coefficients')
    parser.add_argument('--frames', default=3, type=int, help='how many video frames for one iteration')
    parser.add_argument('--tune', action="store_true", help='if tuning, the start_epoch will be set 1')

    # weight
    parser.add_argument('--w_transport', '-tw0', default=1., type=float, help='')
    parser.add_argument('--w_albedo', '-tw1', default=1., type=float, help='')
    parser.add_argument('--w_light', '-tw2', default=1., type=float, help='')
    parser.add_argument('--w_shading_transport', '-tw5', default=1., type=float, help='')
    parser.add_argument('--w_shading_light', '-tw6', default=1., type=float, help='')
    parser.add_argument('--w_shading_all', '-tw7', default=1., type=float, help='')
    parser.add_argument('--w_rendering_albedo', '-tw8', default=1., type=float, help='')
    parser.add_argument('--w_rendering_transport', '-tw9', default=1., type=float, help='')
    parser.add_argument('--w_rendering_light', '-tw10', default=1., type=float, help='')
    parser.add_argument('--w_rendering_albedo_transport', '-tw11', default=1., type=float, help='')
    parser.add_argument('--w_rendering_transport_light', '-tw12', default=1., type=float, help='')
    parser.add_argument('--w_rendering_albedo_light', '-tw13', default=1., type=float, help='')
    parser.add_argument('--w_rendering_all', '-tw14', default=1., type=float, help='')
    parser.add_argument('--w_parsing', '-tw15', default=1., type=float, help='')
    parser.add_argument('--w_albedo_sf', '-tw16', default=1., type=float, help='')
    parser.add_argument('--w_shading_sf', '-tw17', default=1., type=float, help='')

    args = parser.parse_args()
    args = workspace_config(args)
    if not args.test:
        args = train_config(args)
    return args
