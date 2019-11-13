#!/usr/bin/env bash
cuda="1"
lr=1e-6
batch=2
opt="adam"
early=10
backstep=1
lb_weight=5
ae_lb_weight=1
dp=0.5
hidden=350
loss_alpha=2
dir="b"$batch"-"$opt$lr"-w"$lb_weight"_"$ae_lb_weight"-dp"$dp"-h"$hidden"-alpha"$loss_alpha"-"$1
hps="{'wemb_dim': 300, 'wemb_ft': True, 'wemb_dp':$dp , 'pemb_dim': 50, 'pemb_dp': $dp, 'eemb_dim': 50, 'eemb_dp': $dp, 'psemb_dim': 50, 'psemb_dp': $dp, 'lstm_dim': $hidden, 'lstm_layers': 1, 'lstm_dp': 0,'lstm_use_bn':False, 'gcn_et': 3, 'gcn_use_bn': True, 'gcn_layers': 0, 'gcn_dp': $dp, 'sa_dim': $hidden, 'use_highway': True, 'loss_alpha': $loss_alpha}"

mkdir -p models/$dir
cp scripts/train.sh models/$dir
python -m enet.run.ee.runner --train "ace-05-splits/train.json" --test "ace-05-splits/test.json" --dev "ace-05-splits/dev.json" --earlystop $early --restart 999999 --optimizer $opt --lr $lr --webd "embedding/glove.6B.300d.txt" --lb_weight $lb_weight --ae_lb_weight $ae_lb_weight  --batch $batch --back_step $backstep --epochs 99999 --device "cuda:$cuda" --out "models/$dir" --l2decay 1e-8 --hps "$hps" >& models/$dir/log &
