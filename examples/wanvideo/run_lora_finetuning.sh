CUDA_VISIBLE_DEVICES="4,5,6" python train_wan_t2v.py \
  --task train \
  --train_architecture lora \
  --dataset_path /home/schu23/store/datasets/celebv/data \
  --output_path ./models \
  --dit_path "models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 10 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing
