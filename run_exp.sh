# run: CUDA_VISIBLE_DEVICES=<номер GPU> <команда>
set -e

python3 train.py \
    --dir=logs/oct \
    --data_path=./data/ \
    --dataset=SVHN \
    --transform=ResNet_9 \
    --batch_size=500 \
    --model=ResNet9 \
    --save_freq=51 \
    --print_freq=5 \
    --epochs=50 \
    --wd=0.001 \
    --lr=0.0032 \
    --optimizer=Adam \
    --lr-shed=OneCycleLR \
    --dropout 0.5 \
    --comment ResNet9_reverse_margin_boost_SVHN \
    --wandb_group delete_me \
    --gap_size simple_gap \
    --seed 25477 \
    --width 64 \
    --num-nets=15

python3 train.py \
    --dir=logs/oct \
    --data_path=./data/ \
    --dataset=SVHN \
    --transform=ResNet_9 \
    --batch_size=500 \
    --model=ResNet9 \
    --save_freq=51 \
    --print_freq=5 \
    --epochs=50 \
    --wd=0.001 \
    --lr=0.0032 \
    --optimizer=Adam \
    --lr-shed=OneCycleLR \
    --dropout 0.5 \
    --comment ResNet9_margin_boost_SVHN \
    --wandb_group ResNet9_boost_by_margin_in_logits_SVHN \
    --gap_size simple_gap \
    --seed 25477 \
    --width 64 \
    --num-nets=15

python3 train.py \
    --dir=logs/oct \
    --data_path=./data/ \
    --dataset=SVHN \
    --transform=ResNet_9 \
    --batch_size=500 \
    --model=ResNet9 \
    --save_freq=51 \
    --print_freq=5 \
    --epochs=50 \
    --wd=0.001 \
    --lr=0.0032 \
    --optimizer=Adam \
    --lr-shed=OneCycleLR \
    --dropout 0.5 \
    --comment ResNet9_cumgap_margin_boost_SVHN \
    --wandb_group ResNet9_boost_by_margin_in_logits_SVHN \
    --gap_size cummulative_gap \
    --seed 25477 \
    --width 64 \
    --num-nets=15

python3 train.py \
    --dir=logs/oct \
    --data_path=./data/ \
    --dataset=SVHN \
    --transform=ResNet_9 \
    --batch_size=500 \
    --model=ResNet9 \
    --save_freq=51 \
    --print_freq=5 \
    --epochs=50 \
    --wd=0.001 \
    --lr=0.0032 \
    --optimizer=Adam \
    --lr-shed=OneCycleLR \
    --dropout 0.5 \
    --comment ResNet9_meancumgap_margin_boost_SVHN \
    --wandb_group ResNet9_boost_by_margin_in_logits_SVHN \
    --gap_size mean_cummulative_gap \
    --seed 25477 \
    --width 64 \
    --num-nets=15