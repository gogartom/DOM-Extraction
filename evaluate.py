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

IOU_THRESHOLD = 0.5

correct = {}
img_cnt = {}

for cls in classes:
    correct[cls] = {}
    correct[cls]['total'] = 0
    img_cnt[cls] = {}
    img_cnt[cls]['total'] = 0

for i,cls in enumerate(classes):
    #c = 0
    filename = 'data/eshops/results/' + imdb + '_' + cls + '-' + run + '.txt'
    with open(filename, 'r') as f:
        prev = ''
        idx = 0
        for line in f:
            index = line.split()[0]
            if index == prev:
                continue
            shop = ''.join(index.split('-')[:-1])
            if shop not in correct[cls]:
                correct[cls][shop] = 0
                img_cnt[cls][shop] = 0

            box1 = [int(float(x)) for x in line.split()[2:]]
            box2 = gt[idx][i] 
            IOU = calculate_iou(box1, box2)
            if IOU >= IOU_THRESHOLD:
                correct[cls]['total'] += 1
                correct[cls][shop] += 1
                #c += 1
            prev = index
            idx += 1

            img_cnt[cls]['total'] += 1
            img_cnt[cls][shop] += 1

    for shop in correct[cls]:
        print shop, cls, 'accuracy: ', correct[cls][shop]/(img_cnt[cls][shop]*1.0)
    print ''
    #print cls, 'accuracy: ', correct[cls]['total']/(img_cnt[cls]['total']*1.0)
    #print cls, 'accuracy: ', c/(n*1.0)

