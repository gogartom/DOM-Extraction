# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.eshops
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import json
import math

class eshops(datasets.imdb):

    def __init__(self, image_set, data_path=None):
        datasets.imdb.__init__(self, 'eshops')
        self._image_set = image_set
        self._data_path = os.path.join(datasets.ROOT_DIR, 'data', 'eshops')
        self._classes = ('__background__', 'price', 'main_image', 'name')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpeg'
        self._image_index = self._load_image_set_index()

        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        assert os.path.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.abspath(os.path.join(self._data_path, 'images', index + self._image_ext))
        assert os.path.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'image_sets', self._image_set + '.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self): 
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'eshops')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self._image_set + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self._image_set, cache_file)
            return roidb

        gt_roidb = [self._load_annotation(index) for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self._image_set + '_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self._image_set, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        box_list = []
        for index in self.image_index:
            filename = os.path.join(self._data_path, 'annotations', index + '.json')
            with open(filename) as data:    
                annotation = json.load(data)
                objs = annotation['visibleBB']
                num_objs = len(objs)
                boxes = np.zeros((num_objs, 4), dtype=np.uint16)

                # Load object bounding boxes into a data frame.
                for ix, obj in enumerate(objs):
                    x1, y1, x2, y2 = [int(math.ceil(x)) for x in obj]
                    boxes[ix, :] = [x1, y1, x2-1, y2-1]
                box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'annotations', index + '.json')
        with open(filename) as data:    
            annotation = json.load(data)

            objs = annotation['typedObjects']
            num_objs = len(objs)

            boxes = np.zeros((self.num_classes-1, 4), dtype=np.uint16)
            gt_classes = np.zeros((self.num_classes-1), dtype=np.int32)
            overlaps = np.zeros((self.num_classes-1, self.num_classes), dtype=np.float32)

            # Load object bounding boxes into a data frame.
            loaded = 0
            for obj in objs:
                if obj['type'] not in self._class_to_ind:
                    continue
                cls = self._class_to_ind[obj['type']]
                x1, y1, x2, y2 = [int(math.ceil(x)) for x in obj['boundingBox']]
                boxes[loaded, :] = [x1, y1, x2-1, y2-1]
                gt_classes[loaded] = cls
                overlaps[loaded, cls] = 1.0
                loaded += 1

            overlaps = scipy.sparse.csr_matrix(overlaps)

            #print boxes

            return {'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps' : overlaps,
                    'flipped' : False}

    def _write_results_file(self, all_boxes):
        comp_id = '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._data_path, 'results')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = os.path.join(path, self._image_set + '_' + cls + comp_id + '.txt')
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3]))
        return comp_id


    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_results_file(all_boxes)

if __name__ == '__main__':
    #d = datasets.eshops('all')
    d = datasets.eshops('alza_test')
    #d._load_annotation('xaa-433')
    #print d.image_path_from_index('xaa-433')
    res = d.roidb
    #from IPython import embed; embed()
