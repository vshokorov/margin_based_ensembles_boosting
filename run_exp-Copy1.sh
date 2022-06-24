# run: CUDA_VISIBLE_DEVICES=<номер GPU> <команда>

<<<<<<< HEAD
for gap in 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50
do
    python3 train.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=CIFAR10 \
                     --transform=ResNet_9_noDA \
                     --batch_size=500 \
                     --model=ResNet18 \
=======
for run in 0 1
do
    python3 train_ens_meancumgap.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=SVHN \
                     --transform=ResNet_9 \
                     --batch_size=500 \
                     --model=ResNet9 \
>>>>>>> boosting_by_gap_ens
                     --save_freq=51 \
                     --print_freq=5 \
                     --epochs=50 \
                     --wd=0.001 \
<<<<<<< HEAD
                     --lr=0.004 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
                     --comment "ResNet18_gap_${gap}_CIFAR10_noDA" \
                     --wandb_group "ResNet18_margin_in_logits_CIFAR10" \
                     --gap_size ${gap} \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=2 \
                     --num-exps=1
done


for gap in 0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 25 30 35 40 45 50
do
    python3 train.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=CIFAR10 \
                     --transform=ResNet_9 \
                     --batch_size=500 \
                     --model=ResNet18 \
=======
                     --lr=0.0032 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
                     --comment "ResNet9_meancummargin_boost_SVHN" \
                     --wandb_group "ResNet9_boost_by_margin_in_logits_SVHN" \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=15 \
                     --num-exps=1 \
                     --wandb_api_key=$(head -n 1 wandb_key)
done

for run in 0 1
do
    python3 train_ens_meancumgap.py --dir=logs/oct \
                     --data_path=./data/ \
                     --dataset=SVHN \
                     --transform=ResNet_9_noDA \
                     --batch_size=500 \
                     --model=ResNet9 \
>>>>>>> boosting_by_gap_ens
                     --save_freq=51 \
                     --print_freq=5 \
                     --epochs=50 \
                     --wd=0.001 \
<<<<<<< HEAD
                     --lr=0.004 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
                     --comment "ResNet18_gap_${gap}_CIFAR10" \
                     --wandb_group "ResNet18_margin_in_logits_CIFAR10" \
                     --gap_size ${gap} \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=2 \
                     --num-exps=1
=======
                     --lr=0.0032 \
                     --optimizer=Adam \
                     --lr-shed=OneCycleLR \
                     --dropout 0.5 \
                     --comment "ResNet9_meancummargin_boost_SVHN_noDA" \
                     --wandb_group "ResNet9_boost_by_margin_in_logits_SVHN" \
                     --seed 25477 \
                     --width 16 \
                     --num-nets=15 \
                     --num-exps=1 \
                     --wandb_api_key=$(head -n 1 wandb_key)
>>>>>>> boosting_by_gap_ens
done