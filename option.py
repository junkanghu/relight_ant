from utils import workspace_config, train_config
import argparse

def get_opt():
    parser = argparse.ArgumentParser()
    # model-independent options
    parser.add_argument('--config', type=str, default=None,
                        help='config file path')
    parser.add_argument('--discard_ckpt', action='store_true')
    parser.add_argument('--test', action='store_true')

    # DDP-related
    parser.add_argument('--dist_backend', default='nccl', help='which backend to use')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='gpu id for current process')
    parser.add_argument('--rank', type=int, default=None, help='rank id among all the world process')
    parser.add_argument('--world_size', type=int, default=None, help='total number of ddp processes')
    parser.add_argument('--data_load_works', type=int, default=4, help='dataloader workers for loading data')

    parser.add_argument('--data_dir', type=str, 
                        help='Data directory')
    parser.add_argument('--out_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--log_dir', type=str, default='train.log',
                        help='log directory')

    parser.add_argument('--save_fre', type=int, default=5)
    parser.add_argument('--val_fre', type=int, default=5)

    parser.add_argument('--test_dir', type=str, 
                        help='Test data directory')

    # model-dependent options
    parser.add_argument('--epochs', type=int, default=200,
                        help='Epoch for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training')

    parser.add_argument('--yaml_dir', default='/nas/home/hujunkang/data.yaml', help='Directory for test input images')
    parser.add_argument('--light_dir', '-l0', default="/nas/home/hujunkang/sh_hdr", help='Light directory for training')
    parser.add_argument('--sh_num', default=25, type=int, help='number of sh coefficients')
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
