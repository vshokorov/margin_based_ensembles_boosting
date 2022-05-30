# run: CUDA_VISIBLE_DEVICES=<номер GPU> <команда>
wandb_api_key="967df0955ba82554544659fe56aae719bbad58c6"


for gap in 10.5 11 11.5 12 12.5 13 13.5 14 14.5 15 15.5 16 16.5 17 17.5 18 18.5 19 19.5 20 20.5 21 21.5 22 22.5 23 23.5 24 24.5 25
do
    python3 train.py --dir=logs/oct \
                        --data_path=./data/ \
                        --dataset=CIFAR100 \
                        --transform=VGG_noDA \
                        --batch_size=500 \
                        --model=ResNet9 \
                        --save_freq=50 \
                        --print_freq=5 \
                        --epochs=50 \
                        --wd=0.001 \
                        --lr=0.0032 \
                        --optimizer=Adam \
                        --lr-shed=OneCycleLR \
                        --dropout 0.5 \
                        --comment "ResNet9_gap_${gap}_CIFAR100_noDA" \
                        --wandb_group ResNet9_margin_in_logits_CIFAR100 \
                        --gap_size $gap \
                        --seed 25477 \
                        --width 64 \
                        --num-nets=2 \
                        --num-exps=1 \
                        --wandb_api_key=$wandb_api_key
done