CUDA_VISIBLE_DEVICES=1,2,3,4 \
torchrun \
--nproc_per_node=4 \
--master_port 30000 \
train_caption.py \
--config configs/caption_coco.yaml \
--output_dir output/blip