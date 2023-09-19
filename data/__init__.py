import os
import yaml
from natsort import natsorted
from tqdm import tqdm

def get_dir(opt):
    if not opt.test:
        with open(opt.yaml_dir) as f:
            data_info = yaml.safe_load(f)
        scan_path = []

        for idx, scan in enumerate(data_info):
            scan_dir = scan[0]['albedo'].split('/albedo')[0] # xxx/../0000
            scan_path.append(scan_dir)
        # get scan dirs
        test_scan_dir = scan_path[::50]
        train_scan_dir = [p for p in scan_path if p not in test_scan_dir]
        
        # get rendered image dirs
        prt_all_path_train = []
        prt_all_path_test = []
        light_yaml_info = {}
        for idx, scan_dir in enumerate(scan_path):
            prt_dir = os.path.join(scan_dir, 'prt')
            prt_path = sorted(os.listdir(prt_dir))
            prt_path = [os.path.join(prt_dir, p) for p in prt_path if not p[0] == '.']
            if scan_dir in train_scan_dir:
                prt_all_path_train += prt_path
            else:
                prt_all_path_test += prt_path
            
            l_yaml = os.path.join(scan_dir, 'yaml/data.yaml')
            with open(l_yaml) as f:
                yaml_info = yaml.safe_load(f)
            light_yaml_info[os.path.basename(scan_dir)] = yaml_info

        return prt_all_path_train[::100], prt_all_path_train[::100], light_yaml_info
    else:
        img_name = sorted(os.listdir(opt.test_dir))
        mask_dir = []
        input_dir = []
        name = []
        for p in img_name:
            if not p[0] == '.':
                mask_dir.append(os.path.join(opt.test_dir, p))
                input_dir.append(os.path.join(opt.test_dir.replace('images', 'masks'), p[:-3] + 'png'))
                name.append(p.split('.')[0])
        return (input_dir, mask_dir, name)
    
def read_txt(path):
    all = []
    with open(path, 'r') as f:
        info = f.readlines()
    for line in info:
        group = line[:-1].split(' ')[:-1]
        all.append(group)
    return all


def get_dir_video(opt):
    root_dir = opt.video_dir
    all_image_dir = []
    all_albedo_dir = []
    all_light_dir = []
    all_parsing_dir = []
    all_prtd_dir = []
    all_prts_dir = []
    all_shading_dir = []
    all_transportd_dir = []
    all_transports_dir = []
    all_mask_dir = []
    all_corre = []
    train_names = []
    video_names = natsorted(os.listdir(root_dir))
    for _, name in enumerate(tqdm(video_names)):
        video_dir = os.path.join(root_dir, name)
        video_corre_dir = os.path.join(video_dir, f'correspondences_{opt.frames}', 'corr_mat.txt')
        video_pair = read_txt(video_corre_dir)
        for idx, pair in enumerate(video_pair):
            group_images = []
            group_albedos = []
            group_light = []
            group_parsing = []
            group_prtd = []
            group_prts = []
            group_shading = []
            group_transportd = []
            group_transports = []
            group_masks = []
            for f in pair:
                group_images.append(os.path.join(video_dir, 'images', f + '.png'))
                group_light.append(os.path.join(video_dir, 'light', f + '.npy'))
                group_parsing.append(os.path.join(video_dir, 'parsing', f + '.png'))
                group_prtd.append(os.path.join(video_dir, 'prt_d', f + '.png'))
                group_prts.append(os.path.join(video_dir, 'prt_s', f + '.png'))
                group_shading.append(os.path.join(video_dir, 'shading', f + '.png'))
                group_transportd.append(os.path.join(video_dir, 'transport_d', f + '.npy'))
                group_transports.append(os.path.join(video_dir, 'transport_s', f + '.npy'))
                group_masks.append(os.path.join(video_dir, 'masks', f + '.png'))
                group_albedos.append(os.path.join(video_dir, 'albedos', f + '.png'))
            all_corre.append(os.path.join(video_dir, f'correspondences_{opt.frames}', 'corrs', str(idx) + '.npz'))
            all_image_dir.append(group_images)
            all_albedo_dir.append(group_albedos)
            all_light_dir.append(group_light)
            all_parsing_dir.append(group_parsing)
            all_prtd_dir.append(group_prtd)
            all_prts_dir.append(group_prts)
            all_shading_dir.append(group_shading)
            all_transportd_dir.append(group_transportd)
            all_transports_dir.append(group_transports)
            all_mask_dir.append(group_masks)
            train_names.append(pair)

    img_names = natsorted(os.listdir(os.path.join(opt.video_val_dir, 'images')))
    val_image_dir = [os.path.join(opt.video_val_dir, 'images', n) for n in img_names]
    val_mask_dir = [f.replace('images', 'masks') for f in val_image_dir]
    val_image_sequence_dir = []
    val_mask_sequence_dir = []
    val_names = []
    for i in range(0, len(img_names), opt.frames):
        val_image_sequence_dir.append(val_image_dir[i:(i+opt.frames)])
        val_mask_sequence_dir.append(val_mask_dir[i:(i+opt.frames)])
        val_names.append(img_names[i:(i+opt.frames)])
            
    return [all_image_dir, all_mask_dir, train_names, all_albedo_dir, all_light_dir, all_parsing_dir, all_prtd_dir, all_prts_dir, all_shading_dir,
        all_transportd_dir, all_transports_dir, all_corre], [val_image_sequence_dir, val_mask_sequence_dir, val_names]


# def get_dir_video(opt):
#     root_dir = opt.video_dir
#     all_image_dir = []
#     all_albedo_dir = []
#     all_mask_dir = []
#     all_corre = []
#     train_names = []
#     video_names = natsorted(os.listdir(root_dir))
#     for _, name in enumerate(tqdm(video_names)):
#         video_dir = os.path.join(root_dir, name)
#         video_corre_dir = os.path.join(video_dir, f'correspondences_{opt.frames}', 'corr_mat.txt')
#         video_pair = read_txt(video_corre_dir)
#         for idx, pair in enumerate(video_pair):
#             group_images = []
#             group_albedos = []
#             group_masks = []
#             for f in pair:
#                 group_images.append(os.path.join(video_dir, 'images', f + '.png'))
#                 group_masks.append(os.path.join(video_dir, 'masks', f + '.png'))
#                 group_albedos.append(os.path.join(video_dir, 'albedos', f + '.png'))
#             all_corre.append(os.path.join(video_dir, f'correspondences_{opt.frames}', 'corrs', str(idx) + '.npz'))
#             all_image_dir.append(group_images)
#             all_albedo_dir.append(group_albedos)
#             all_mask_dir.append(group_masks)
#             train_names.append(pair)

#     img_names = natsorted(os.listdir(os.path.join(opt.video_val_dir, 'images')))
#     val_image_dir = [os.path.join(opt.video_val_dir, 'images', n) for n in img_names]
#     val_mask_dir = [f.replace('images', 'masks') for f in val_image_dir]
#     val_image_sequence_dir = []
#     val_mask_sequence_dir = []
#     val_names = []
#     for i in range(0, len(img_names), opt.frames):
#         val_image_sequence_dir.append(val_image_dir[i:(i+opt.frames)])
#         val_mask_sequence_dir.append(val_mask_dir[i:(i+opt.frames)])
#         val_names.append(img_names[i:(i+opt.frames)])
            
#     return all_image_dir, all_mask_dir, all_albedo_dir, all_corre, train_names, val_image_sequence_dir, val_mask_sequence_dir, val_names