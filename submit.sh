# shading_vgg
export workspace=/mnt/data4/home/hujunkang/debug/shading_vgg && mkdir -p workspace && OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 python train.py \
--config ./configs/train.yml --batch_size 4 --shading_vgg

# albedo_vgg
export workspace=/mnt/data4/home/hujunkang/debug/albedo_vgg && mkdir -p workspace && OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 python train.py \
--config ./configs/train.yml --batch_size 4 --albedo_vgg

# transport_sf
export workspace=/mnt/data4/home/hujunkang/debug/transport_sf && mkdir -p workspace && OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 python train.py \
--config ./configs/train.yml --batch_size 4 --transport_sf

# ssim_albedo
export workspace=/mnt/data4/home/hujunkang/debug/ssim_albedo && mkdir -p workspace && OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 python train.py \
--config ./configs/train.yml --batch_size 4 --ssim_albedo

# ssim_transport
export workspace=/mnt/data4/home/hujunkang/debug/ssim_transport && mkdir -p workspace && OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 python train.py \
--config ./configs/train.yml --batch_size 4 --ssim_transport

# ssim_shading
export workspace=/mnt/data4/home/hujunkang/debug/ssim_shading && mkdir -p workspace && OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 python train.py \
--config ./configs/train.yml --batch_size 4 --ssim_shading

# base
export workspace=/mnt/data4/home/hujunkang/debug/base && mkdir -p workspace && OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 python train.py \
--config ./configs/train.yml --batch_size 4

# regular
export workspace=/mnt/data4/home/hujunkang/debug/regular && mkdir -p workspace && OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=1 python train.py \
--config ./configs/train.yml --batch_size 4 --regular