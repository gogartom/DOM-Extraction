#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""
import matplotlib
matplotlib.use('Agg')

import _init_paths
from fast_rcnn.test_DOM_regres import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import datasets
import caffe
import argparse
import pprint
import time, os, sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='models/CaffeNet_DOM_regres/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='output/eshops/eshops/master_train_2016-01-12_15:55:02/iters/caffenet_DOM_regres_iter_3000.caffemodel', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default='experiments/cfgs/DOM_regres.yml', type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='master_not_seen_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

#    if len(sys.argv) == 1:
#        parser.print_help()
#        sys.exit(1)

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

    while not os.path.exists(args.caffemodel) and args.wait:
        print('Waiting for {} to exist...'.format(args.caffemodel))
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    #imdb = get_imdb(args.imdb_name)
    imdb = datasets.eshops(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    precision, aver_IOU = test_net(net, imdb, args.imdb_name, num_images=100, save_images=True)
    print '======================='
    print 'Precision:'
    print 'Price',precision[0],'| Main image:',precision[1],'| Name:', precision[2]
    print '======================='
    print 'Average IOU:'
    print 'Price',aver_IOU[0],'| Main image:',aver_IOU[1],'| Name:', aver_IOU[2]
    print '======================='
