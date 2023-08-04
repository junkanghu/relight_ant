# from .TrainDataset import Dataset
import os
import yaml

def get_dir(opt):
    if not opt.test:
        with open(opt.yaml_dir) as f:
            data_info = yaml.safe_load(f)
        scan_path = []

        for idx, scan in enumerate(data_info[:2]):
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

        return prt_all_path_train, prt_all_path_test, light_yaml_info
    else:
        img_name = sorted(os.listdir(opt.test_dir))
        mask_dir = []
        input_dir = []
        name = []
        for p in img_name:
            if not p[0] == '.':
                mask_dir.append(os.path.join(opt.test_dir, p))
                input_dir.append(os.path.join(opt.test_dir.replace('img', 'mask'), p[:-3] + 'png'))
                name.append(p.split('.')[0])
        return (input_dir, mask_dir, name)