import sys
import cPickle
import numpy
import json
import datasets

def area(box):
    return (int(box[2])-int(box[0])) * (int(box[3])-int(box[1]))

def calculate_iou(box1, box2):
    axmin, aymin, axmax, aymax = box1
    bxmin, bymin, bxmax, bymax = box2
    dx = int(min(axmax, bxmax)) - int(max(axmin, bxmin))
    dy = int(min(aymax, bymax)) - int(max(aymin, bymin))
    S12 = max(dx*dy, 0)
    S1 = area(box1)
    S2 = area(box2)
    iou = (S12*1.0)/(S1+S2-S12)
    return iou

imdb = sys.argv[1]
run = sys.argv[2]

d = datasets.eshops(imdb)

gt = [ x['boxes'].tolist() for x in d.gt_roidb()]

#gt = {}
#for index in d._image_index:
    #gt[index] = d._load_annotation(index)['boxes']

n = len(gt)


classes = ('name', 'price', 'main_image')

IOU_THRESHOLD = 1

for i,cls in enumerate(classes):
    correct = 0
    filename = 'data/eshops/results/' + imdb + '_' + cls + '-' + run + '.txt'
    with open(filename, 'r') as f:
        prev = ''
        idx = 0
        for line in f:
            index = line.split()[0]
            if index == prev:
                continue
            box1 = [int(float(x)) for x in line.split()[2:]]
            box2 = gt[idx][i] 
            '''
            box3 = gt[index][i]
            for j in range(len(box2)):
                if box2[j] != box3[j]:
                    print idx+1, index, cls
                    print box2, box3
            '''
            IOU = calculate_iou(box1, box2)
            if IOU >= IOU_THRESHOLD:
                correct += 1
            prev = index
            idx += 1

    print cls, 'accuracy: ', correct/(n*1.0)

