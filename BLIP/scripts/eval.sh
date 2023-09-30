CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
--nproc_per_node=4 \
--master_port 30010 \
train_caption.py \
--evaluate \
--eval_split test \
--config configs/caption_coco.yaml \
--output_dir output/blip