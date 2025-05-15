for idx in {0..9}; do
    python testbase.py \
        --model deit_base_patch16_224 \
        --datatrain_path ../TinyImageNet/tiny-imagenet-200 \
        --dataval_path ../TinyImageNet/tiny-imagenet-200 \
        --batch_size 64 \
        --neuron_pruning \
        --head_pruning \
        --seed 2025 \
        --idx $idx
done
