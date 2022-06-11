# run: CUDA_VISIBLE_DEVICES=<номер GPU> <команда>
wandb_api_key="967df0955ba82554544659fe56aae719bbad58c6"

for gap in 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20
do
    python3 train.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=CIFAR10 \
                     --transform=VGG_noDA \
                     --batch_size=500 \
                     --model=VGG16 \
                     --save_freq=51 \
                     --print_freq=5 \
                     --epochs=50 \
                     --wd=0.001 \
                     --lr=0.004 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
                     --comment "width16_gap_${gap}_CIFAR10_noDA_ADAM" \
                     --wandb_group "width16_margin_in_logits_CIFAR10_ADAM" \
                     --gap_size ${gap} \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=2 \
                     --num-exps=1 \
                     --wandb_api_key=$wandb_api_key
done


for gap in 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20
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
                     --lr=0.004 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
                     --comment "width16_gap_${gap}_CIFAR10_ADAM" \
                     --wandb_group "width16_margin_in_logits_CIFAR10_ADAM" \
                     --gap_size ${gap} \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=2 \
                     --num-exps=1 \
                     --wandb_api_key=$wandb_api_key
done