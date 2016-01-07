# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import time

def get_minibatch(roidb, num_classes, means, stds):
    """Given a roidb, construct a minibatch sampled from it."""

    #start = time.time()
    num_images = len(roidb)

    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)
    bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    for im_i in xrange(num_images):
        im_rois, bbox_targets, bbox_loss \
            = _sample_rois(roidb[im_i], num_classes)

        # Add to RoIs blob
        rois = _project_im_rois(im_rois, im_scales[im_i])
        batch_ind = im_i * np.ones((rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, rois))
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels, bbox targets, and bbox loss blobs
        bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))

    if cfg.TRAIN.VIS_MINIBATCH:
        # For debug visualizations
        _vis_minibatch(im_blob, rois_blob, bbox_targets_blob, means, stds)

    blobs = {'data': im_blob,
             'rois': rois_blob,
             'bbox_targets' : bbox_targets_blob,
             'bbox_loss_weights' : bbox_loss_blob}

    return blobs

def _sample_rois(roidb, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    
    # Randomly choose one of possible boxes
    possible_boxes = roidb['DOM_bbox_targets'].shape[0]
    chosen_ind = np.random.randint(low=0, high=possible_boxes, size=1)

    rois = roidb['boxes'][chosen_ind]
    targets = roidb['DOM_bbox_targets'][chosen_ind,:,:][0]
    bbox_targets, bbox_loss_weights = \
                         _get_bbox_regression_labels(targets,num_classes, roidb['gt_classes'])

    return rois, bbox_targets, bbox_loss_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])

        if cfg.TRAIN.CHANGE_HUE:
            # Change hue randomly
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) 
            add_value = np.random.randint(low=0,high=180,size=1)[0]
            add_matrix = np.ones(hsv.shape[0:2])*add_value
            hsv2 = hsv
            hsv2[:,:,0] += add_matrix
            im = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes, target_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_loss_weights (ndarray): N x 4K blob of loss weights
    """

    clss = target_classes

    bbox_targets = np.zeros((1, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)

    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[0, start:end] = bbox_target_data[ind, :]
        bbox_loss_weights[0, start:end] = [1., 1., 1., 1.]

    return bbox_targets, bbox_loss_weights

def _vis_minibatch(im_blob, rois_blob, bbox_targets_blob, means, stds):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
   
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)

        plt.imshow(im)
        rect = plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
        
        
        # show bbox regression
        circles = []
        for j in xrange(1,4):
            bbox = bbox_targets_blob[i,j*4:j*4+4]
            m = means[j*4:j*4+4]
            s = stds[j*4:j*4+4]

            dx = bbox[0]*s[0]+m[0]
            dy = bbox[1]*s[1]+m[1]

            width = roi[2] - roi[0]
            height = roi[3] - roi[1]

            center_x = width/2 + dx*width
            center_y = height/2 + dy*height

            circle=plt.Circle((center_x+roi[0],center_y+roi[1]),10,color='b')
            circles.append(circle)
            plt.gca().add_patch(circles[j-1])

        plt.gca().add_patch(rect)
        #plt.show()
        plt.savefig('blobs/blob_'+str(i)+'.png')

        rect.remove()
        for circ in circles:
            circ.remove()