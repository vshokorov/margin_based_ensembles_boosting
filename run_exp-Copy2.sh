# run: CUDA_VISIBLE_DEVICES=<номер GPU> <команда>
wandb_api_key="967df0955ba82554544659fe56aae719bbad58c6"


for gap in 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60
do
    python3 train.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=SVHN \
                     --transform=ResNet_9 \
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
                     --comment "ResNet9_gap_${gap}_SVHN" \
                     --wandb_group "ResNet9_margin_in_logits_SVHN" \
                     --gap_size $gap \
                     --seed 25477 \
                     --width 64 \
                     --num-nets=2 \
                     --num-exps=1 \
                     --wandb_api_key=$wandb_api_key
done