#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import matplotlib
matplotlib.use('Agg')

import datasets
import datasets.eshops
import _init_paths
from fast_rcnn.train_DOM_regres import train_net, get_DOM_regres_training_roidb
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import datetime
import time
import shutil
import os

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default='models/CaffeNet_DOM_regres/solver.prototxt', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default='data/imagenet_models/CaffeNet.v2.caffemodel', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='experiments/cfgs/DOM_regres.yml', type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='master_train', type=str)
    parser.add_argument('--imdb_test', dest='imdb_test_name',
                        help='dataset to test on',
                        default='master_not_seen_test', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    
    imdb = datasets.eshops(args.imdb_name)
    imdb_test = datasets.eshops(args.imdb_test_name)

    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb = get_DOM_regres_training_roidb(imdb)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = os.path.join(get_output_dir(imdb, None), args.imdb_name + '_' + st)
    os.makedirs(output_dir)
    shutil.copy2(args.cfg_file, output_dir)
    print 'Output will be saved to `{:s}`'.format(output_dir)


    train_net(args.solver, roidb, imdb_test, 
              args.imdb_test_name, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
