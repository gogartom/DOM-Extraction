import sys
import cPickle
import numpy as np
import numpy.matlib as npmat
from numpy import linalg as LA
import json
import datasets

def area(box):
    return (int(box[2])-int(box[0])) * (int(box[3])-int(box[1]))

def calculate_iou(box1, box2):
    axmin, aymin, axmax, aymax = box1
    bxmin, bymin, bxmax, bymax = box2
    dx = int(min(axmax, bxmax)) - int(max(axmin, bxmin))
    dy = int(min(aymax, bymax)) - int(max(aymin, bymin))
    
    if(dx<0 and dy<0):
        return 0
    else:
        #S12 = max(dx*dy, 0)
        S12 = dx*dy
        S1 = area(box1)
        S2 = area(box2)
    iou = (S12*1.0)/(S1+S2-S12)
    return iou

def area_matrix(boxes):
    return np.multiply(boxes[:,2]-boxes[:,0], boxes[:,3]-boxes[:,1])

def calculate_iou_with_matrix(pred_box, dom_elements):

    pred_box = np.asmatrix(pred_box,dtype='float32')
    dom_elements = np.asmatrix(dom_elements,dtype='float32')
    dom_count = dom_elements.shape[0]    

    pred_matrix = npmat.repmat(pred_box,dom_count,1)

    dx = np.minimum(pred_matrix[:,2],dom_elements[:,2]) - np.maximum(pred_matrix[:,0],dom_elements[:,0])
    dy = np.minimum(pred_matrix[:,3],dom_elements[:,3]) - np.maximum(pred_matrix[:,1],dom_elements[:,1])

    non_zero_inds = np.squeeze(np.where(np.logical_and(dy>=0,dx>=0))[0])    

    S12 = np.maximum(np.multiply(dx,dy),np.zeros((dom_count,1)))
    S1 = area_matrix(pred_matrix)
    S2 = area_matrix(dom_elements)
    IOU = (S12*1.0)/(S1+S2-S12)

    res = np.zeros((dom_count,1))
    res[non_zero_inds] = IOU[non_zero_inds]
    
    return res

def centers_distance(gt_box, pred_box):
    cent_x_gt = gt_box[0]+((gt_box[2]-gt_box[0])/2)
    cent_y_gt = gt_box[1]+((gt_box[3]-gt_box[1])/2)

    cent_x_pred = pred_box[0]+((pred_box[2]-pred_box[0])/2)
    cent_y_pred = pred_box[1]+((pred_box[3]-pred_box[1])/2)

    cent_gt = np.array([cent_x_gt, cent_y_gt],dtype=np.float32)
    cent_pred = np.array([cent_x_pred, cent_y_pred],dtype=np.float32)

    dist = LA.norm(cent_gt-cent_pred)
    return dist

def get_results(pred_boxes,roidb):
    
    # Get gt_boxes and other_boxes
    gt_boxes = roidb['boxes'][0:3]
    gt_classes = roidb['gt_classes'][0:3]

    max_overlaps = roidb['gt_overlaps'].max(axis=1).todense()
    non_gt_indices = np.squeeze(np.asarray(np.where(max_overlaps!=1.0)[0]))
    other_boxes = roidb['boxes'][non_gt_indices,:]

    gt_boxes = gt_boxes.astype(np.float32)
    other_boxes = other_boxes.astype(np.float32)

    # For each class get results
    class_results = []
    for i in xrange(1,4):

        ind = np.where(gt_classes==i)
        gt_box = np.squeeze(gt_boxes[ind])
        pred_box = pred_boxes[0,i*4:i*4+4]

        # Compute distance of centers
        dist = centers_distance(gt_box, pred_box)

        # Compute iou of GT and others
        gt_iou = calculate_iou_with_matrix(pred_box,gt_box)[0,0]
        others_iou = calculate_iou_with_matrix(pred_box,other_boxes)
        max_others = others_iou.max()
       
        if gt_iou>=max_others:
            chosen = 1
        else:
            chosen = 0
        class_results.append((gt_iou,chosen,dist))
    
    return class_results

if __name__ == "__main__":

    pred_box = [0,0,11,21]
    dom_elements = np.matrix('0 0 11 21; 1 1 10 20; 2 2 11 16; 14 14 15 15')

    print 'PREDICTION BOX', pred_box
    print 'Individuals:'
    for i in xrange(dom_elements.shape[0]):
        box = np.squeeze(np.asarray(dom_elements[i,:]))
        print box, calculate_iou(pred_box,box)

    print 'matrix'
    print calculate_iou_with_matrix(pred_box,dom_elements)

