# run: CUDA_VISIBLE_DEVICES=<номер GPU> <команда>
wandb_api_key="967df0955ba82554544659fe56aae719bbad58c6"


for gap in 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do
    python3 train.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=CIFAR100 \
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
                     --comment "ResNet9_gap_${gap}_CIFAR100" \
                     --wandb_group "ResNet9_margin_in_logits_CIFAR100" \
                     --gap_size $gap \
                     --seed 25477 \
                     --width 64 \
                     --num-nets=2 \
                     --num-exps=1 \
                     --wandb_api_key=$wandb_api_key
done


for gap in 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do
    python3 train.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=CIFAR100 \
                     --transform=ResNet_9 \
                     --batch_size=500 \
                     --model=ResNet18 \
                     --save_freq=51 \
                     --print_freq=5 \
                     --epochs=50 \
                     --wd=0.001 \
                     --lr=0.01 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
                     --comment "ResNet18_gap_${gap}_CIFAR100" \
                     --wandb_group "ResNet18_margin_in_logits_CIFAR100" \
                     --gap_size $gap \
                     --seed 25477 \
                     --width 64 \
                     --num-nets=2 \
                     --num-exps=1 \
                     --wandb_api_key=$wandb_api_key
done


for gap in 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 6 7 8 9 10
do
    python3 train.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=CIFAR10 \
                     --transform=VGG \
                     --batch_size=500 \
                     --model=VGG16 \
                     --save_freq=51 \
                     --print_freq=5 \
                     --epochs=50 \
                     --wd=0.001 \
                     --lr=0.0032 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
                     --comment "width16_gap_${gap}_CIFAR10" \
                     --wandb_group "width16_margin_in_logits_CIFAR10" \
                     --gap_size $gap \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=2 \
                     --num-exps=1 \
                     --wandb_api_key=$wandb_api_key
done
