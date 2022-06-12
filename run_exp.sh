# run: CUDA_VISIBLE_DEVICES=<номер GPU> <команда>

python3 train_ens.py --dir=logs/oct \
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
                     --comment ResNet9_margin_boost_SVHN \
                     --wandb_group ResNet9_boost_by_margin_in_logits_SVHN \
                     --seed 25477 \
                     --width 64 \
                     --num-nets=15 \
                     --num-exps=1