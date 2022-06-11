# run: CUDA_VISIBLE_DEVICES=<номер GPU> <команда>
wandb_api_key="967df0955ba82554544659fe56aae719bbad58c6"

for gap in 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50
do
    python3 train.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=CIFAR100 \
                     --transform=ResNet_9_noDA \
                     --batch_size=500 \
                     --model=ResNet18 \
                     --save_freq=51 \
                     --print_freq=5 \
                     --epochs=50 \
                     --wd=0.001 \
                     --lr=0.004 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
                     --comment "ResNet18_gap_${gap}_CIFAR100_noDA" \
                     --wandb_group "ResNet18_margin_in_logits_CIFAR100" \
                     --gap_size ${gap} \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=2 \
                     --num-exps=1 \
                     --wandb_api_key=$wandb_api_key
done


for gap in 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50
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
                     --lr=0.004 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
                     --comment "ResNet18_gap_${gap}_CIFAR100" \
                     --wandb_group "ResNet18_margin_in_logits_CIFAR100" \
                     --gap_size ${gap} \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=2 \
                     --num-exps=1 \
                     --wandb_api_key=$wandb_api_key
done