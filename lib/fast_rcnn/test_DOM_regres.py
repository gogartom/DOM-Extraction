# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.DOM_regres_evaluation import get_results 
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from utils.cython_nms import nms
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import matplotlib.pyplot as plt

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
    heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def im_detect(net, im, boxes):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    #if cfg.DEDUP_BOXES > 0:
    #    v = np.array([1, 1e3, 1e6, 1e9, 1e12])
    #    hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
    #    _, index, inv_index = np.unique(hashes, return_index=True,
    #                                    return_inverse=True)
    #    blobs['rois'] = blobs['rois'][index, :]
    #    boxes = boxes[index, :]

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    blobs_out = net.forward(data=blobs['data'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False))
    #if cfg.TEST.SVM:
    #    # use the raw scores before softmax under the assumption they
    #    # were trained as linear SVMs
    #    scores = net.blobs['cls_score'].data
    #else:
    #    # use softmax estimated probabilities
    #    scores = blobs_out['cls_prob']

    #if cfg.TEST.BBOX_REG:
    #    # Apply bounding-box regression deltas
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = _bbox_pred(boxes, box_deltas)
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
    #else:
    #    # Simply repeat the boxes, once for each class
    #    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    #if cfg.DEDUP_BOXES > 0:
    #    # Map scores and predictions back to the original set of boxes
    #    scores = scores[inv_index, :]
    #    pred_boxes = pred_boxes[inv_index, :]

    #return scores, pred_boxes
    return pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(net, imdb, imdb_name, num_images=300):
    """Test a Fast R-CNN network on an image database."""
    total_num_images = len(imdb.image_index)
    
    # output dir should be changed
    #output_dir = get_output_dir(imdb, net)
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)

    images_dir = 'test_DOM_regres_'+imdb_name
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    roidb = imdb.roidb
    indices = np.random.randint(low=0,high=total_num_images,size=num_images)

    IOUs = np.zeros((num_images,3),dtype='float32')
    matches = np.zeros((num_images,3),dtype='float32')

    for i in xrange(num_images):
        ind = indices[i]
        im = cv2.imread(imdb.image_path_at(ind))
    
        #-- get boxes from network
        #dim_w = 1920-1
        #dim_h = 1080-1
        #w = 800
        #h = 600
        #proposals = []
        #for x0 in np.linspace(0, dim_w-w, num = 3):
        #    for y0 in np.linspace(0, dim_h-h, num = 2):
        #        proposals.append([x0, y0, x0+w, y0+h])
    
        boxes = im_detect(net, im, np.array([[0, 0, 1920-1, 1000-1]]))
        results_per_class = get_results(boxes,roidb[ind])        
        
        for j in xrange(0,3):
            IOUs[i,j] = results_per_class[j][0]
            matches[i,j] = results_per_class[j][1]


        #boxes = im_detect(net, im, np.array(proposals))

        #-- show in image
        #plt.imshow(im)

        #rects = []                
        
        #for k in xrange(len(boxes)):
        #    for j in xrange(1,4):
        #        bbox = boxes[k,j*4:j*4+4]
        #        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0],
        #               bbox[3] - bbox[1], fill=False,
        #               edgecolor='r', linewidth=3)

        #        rects.append(rect)
        #        plt.gca().add_patch(rect)

        #plt.savefig(images_dir+'/im_'+str(i)+'.png')
      
        #for rect in rects:
        #    rect.remove()


    return np.mean(matches, axis=0), np.mean(IOUs, axis=0)
