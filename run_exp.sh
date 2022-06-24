# run: CUDA_VISIBLE_DEVICES=<номер GPU> <команда>

<<<<<<< HEAD
python3 train_ens.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=SVHN \
                     --transform=ResNet_9 \
                     --batch_size=500 \
                     --model=ResNet9 \
                     --save_freq=50 \
=======
for run in 0 1
do
    python3 train_ens.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=SVHN \
                     --transform=ResNet_9_noDA \
                     --batch_size=500 \
                     --model=ResNet9 \
                     --save_freq=51 \
>>>>>>> boosting_by_gap_ens
                     --print_freq=5 \
                     --epochs=50 \
                     --wd=0.001 \
                     --lr=0.0032 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
<<<<<<< HEAD
                     --comment ResNet9_margin_boost_SVHN \
                     --wandb_group ResNet9_boost_by_margin_in_logits_SVHN \
                     --seed 25477 \
                     --width 64 \
                     --num-nets=15 \
                     --num-exps=1
=======
                     --comment "ResNet9_margin_boost_SVHN_noDA" \
                     --wandb_group "ResNet9_boost_by_margin_in_logits_SVHN" \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=15 \
                     --num-exps=1 \
                     --wandb_api_key=$(head -n 1 wandb_key)
done

for run in 0 1
do
    python3 train_ens_cumgap.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=SVHN \
                     --transform=ResNet_9_noDA \
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
                     --comment "ResNet9_cummargin_boost_SVHN_noDA" \
                     --wandb_group "ResNet9_boost_by_margin_in_logits_SVHN" \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=15 \
                     --num-exps=1 \
                     --wandb_api_key=$(head -n 1 wandb_key)
done
>>>>>>> boosting_by_gap_ens
