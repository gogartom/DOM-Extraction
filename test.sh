#!/bin/bash
cd /storage/brno8/home/hudler/DOM-Extraction/
source init_script.sh
./tools/test_DOM_net.py --gpu 0  --imdb alza_czc_10k_test --def models/CaffeNet_eshops/no_bbox_reg/test.prototxt --net output/eshops/eshops/caffenet_fast_rcnn_iter_51000.caffemodel --cfg experiments/cfgs/eshops.yml
