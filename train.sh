#!/bin/bash
cd /storage/brno8/home/hudler/DOM-Extraction/
source init_script.sh
./tools/train_DOM_net.py --gpu 0 --imdb alza_czc_10k_train --solver models/CaffeNet_eshops/no_bbox_reg/solver.prototxt --weights data/imagenet_models/CaffeNet.v2.caffemodel --cfg experiments/cfgs/eshops.yml --iters 5000

