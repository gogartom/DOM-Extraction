import sys
import cPickle
import numpy


def area(box):
    return (box[2]-box[0]) * (box[3]-box[1])

def calculate_iou(box1, box2):
    axmin, aymin, axmax, aymax = box1
    bxmin, bymin, bxmax, bymax = box2
    #print axmin, aymin, axmax, aymax 
    #print bxmin, bymin, bxmax, bymax 
    dx = min(axmax, bxmax) - max(axmin, bxmin)
    dy = min(aymax, bymax) - max(aymin, bymin)
    S12 = max(dx*dy, 0)
    S1 = area(box1)
    S2 = area(box2)
    #print S12, S1, S2
    iou = (S12*1.0)/(S1+S2-S12)
    return iou

gt_file = sys.argv[1]

titles = sys.argv[2]
prices = sys.argv[3]
images = sys.argv[4]

with open(gt_file, 'rb') as fid:
    gt_roidb = cPickle.load(fid)

gt = []
for k in gt_roidb:
    gt.append(k['boxes'])

n = len(gt)

IOU_THRESHOLD = 0.85

correct = 0
with open(titles, 'r') as f:
    for idx, line in enumerate(f):
        box = [int(float(x)) for x in line.split()[2:]]
        IOU = calculate_iou(gt[idx][0], box)
        if IOU > IOU_THRESHOLD:
            correct += 1

print 'Title accuracy: ', correct/(n*1.0)

correct = 0
with open(prices, 'r') as f:
    for idx, line in enumerate(f):
        box = [int(float(x)) for x in line.split()[2:]]
        IOU = calculate_iou(gt[idx][1], box)
        if IOU > IOU_THRESHOLD:
            correct += 1

print 'Price accuracy: ', correct/(n*1.0)

correct = 0
with open(images, 'r') as f:
    for idx, line in enumerate(f):
        box = [int(float(x)) for x in line.split()[2:]]
        IOU = calculate_iou(gt[idx][1], box)
        if IOU > IOU_THRESHOLD:
            correct += 1
print 'Image accuracy: ', correct/(n*1.0)
