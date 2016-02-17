# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import pickle
import cv2
from sklearn.preprocessing import normalize

def load_text_map(filename, scale, hide_rois_elements=None, crop_height=None):
    with open(filename,'rb') as f:
        obj = pickle.load(f)

    shape = obj['shape']
    text_nodes = obj['text_nodes']        

    map_shape = [round(x*scale) for x in shape[0:2]]
    n_features = shape[2]

    features = np.zeros((map_shape[0],map_shape[1],n_features), dtype=np.float)
    for node in text_nodes:
        bb = node[0]
        bb_scaled = [round(x*scale) for x in bb]
        encoded_text = node[1]
        encoded_text = normalize(encoded_text, axis=1, norm='l2')
        vector = np.asarray(encoded_text.todense())[0]
        features[bb_scaled[1]:bb_scaled[3],bb_scaled[0]:bb_scaled[2],:] = vector

    # HIDE ELEMENTS
    if hide_rois_elements:
        for elem in hide_rois_elements:
            sc_elem = [round(e*scale) for e in elem]
            features[sc_elem[1]:sc_elem[3],sc_elem[0]:sc_elem[2],:] = 0

    # CROP
    if crop_height:
        scaled_height = round(crop_height*scale)
        features = features[:scaled_height,:,:]

    return features

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    n_channels = ims[0].shape[2]
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], n_channels),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
