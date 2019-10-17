#!/usr/bin/env bash
lr=1e-3
batch=16
opt="adam"
cuda="0"
early=3
backstep=1
dir=$opt$lr-b$batch"x"$backstep-early$early-new"-"$cuda
hps="{'wemb_dim': 300, 'wemb_ft': True, 'wemb_dp': 0.5, 'pemb_dim': 50, 'pemb_dp': 0.5, 'eemb_dim': 50, 'eemb_dp': 0.5, 'psemb_dim': 50, 'psemb_dp': 0.5, 'lstm_dim': 300, 'lstm_layers': 1, 'lstm_dp': 0, 'gcn_et': 3, 'gcn_use_bn': True, 'gcn_layers': 3, 'gcn_dp': 0.5, 'sa_dim': 300, 'use_highway': True, 'loss_alpha': 5}"

mkdir -p models/$dir
cp scripts/train.sh models/$dir
python -m enet.run.ee.runner --train "ace-05-splits/train.json" --test "ace-05-splits/test.json" --dev "ace-05-splits/dev.json" --earlystop $early --restart 999999 --optimizer $opt --lr $lr --webd "embedding/glove.6B.300d.txt" --batch $batch --back_step $backstep --epochs 99999 --device "cuda:$cuda" --out "models/$dir" --l2decay 1e-8 --hps "$hps" >& models/$dir/log &
