# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import numpy.matlib
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        print 'Computing bounding-box regression targets...'
        #self.bbox_means, self.bbox_stds = \
        #        rdl_roidb.add_bbox_regression_targets(roidb)
        self.bbox_means, self.bbox_stds = self.prepare_bbox_regression_targets(roidb)
        print 'done'
        
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb,self.bbox_means,self.bbox_stds)

    def prepare_bbox_regression_targets(self,roidb):
        num_images = len(roidb)
        num_classes = roidb[0]['gt_boxes'].shape[0] + 1

        #-- Get targets
        for im_i in xrange(num_images):
            targets = roidb[im_i]['gt_boxes']
            target_classes = roidb[im_i]['gt_classes']
            boxes = roidb[im_i]['boxes']
            roidb[im_i]['DOM_bbox_targets'] = self._compute_targets(targets, target_classes, boxes)

        #-- Compute values needed for means and stds
        #-- var(x) = E(x^2) - E(x)^2
        sums = np.zeros((num_classes, 4))
        squared_sums = np.zeros((num_classes, 4))
        boxes_counts = 0     
        for im_i in xrange(num_images):
            targets = roidb[im_i]['DOM_bbox_targets']
            target_classes = roidb[im_i]['gt_classes']

            boxes_counts += targets.shape[0]
            for cls in xrange(1, num_classes):
                cls_ind = np.where(target_classes == cls)[0][0]                
                sums[cls, :] += targets[:,cls_ind,:].sum(axis=0)
                squared_sums[cls, :] += (targets[:,cls_ind,:] ** 2).sum(axis=0)
        
        means = sums / boxes_counts
        stds = np.sqrt(squared_sums / boxes_counts - means ** 2)

        #-- Normalize targets
        for im_i in xrange(num_images):
            targets = roidb[im_i]['DOM_bbox_targets']
            for cls in xrange(1, num_classes):
                cls_ind = np.where(target_classes == cls)[0][0]

                roidb[im_i]['DOM_bbox_targets'][:,cls_ind, :] -= means[cls, :]
                roidb[im_i]['DOM_bbox_targets'][:,cls_ind, :] /= stds[cls, :]

        return means.ravel(), stds.ravel()

    def _compute_targets(self, gt, gt_classes, boxes):
        #top_page_roi = [0, 0, 1920-1, 1000-1]
        
        gt_count = len(gt_classes)
        boxes_count = len(boxes)

        # ground truth positions
        gt_widths = np.asmatrix(gt[:, 2] - gt[:, 0] + cfg.EPS)
        gt_heights = np.asmatrix(gt[:, 3] - gt[:, 1] + cfg.EPS)
        gt_ctr_x = np.asmatrix(gt[:, 0] + 0.5 * gt_widths)
        gt_ctr_y = np.asmatrix(gt[:, 1] + 0.5 * gt_heights)

        gt_widths = numpy.matlib.repmat(gt_widths,boxes_count,1)
        gt_heights = numpy.matlib.repmat(gt_heights,boxes_count,1)
        gt_ctr_x = numpy.matlib.repmat(gt_ctr_x,boxes_count,1)
        gt_ctr_y = numpy.matlib.repmat(gt_ctr_y,boxes_count,1)

        # boxes position        
        ex_widths = np.asmatrix(boxes[:,2] - boxes[:,0] + cfg.EPS)
        ex_heights = np.asmatrix(boxes[:,3] - boxes[:,1] + cfg.EPS)
        ex_ctr_x = np.asmatrix(boxes[:,0] + 0.5 * ex_widths)
        ex_ctr_y = np.asmatrix(boxes[:,1] + 0.5 * ex_heights)

        ex_widths = numpy.matlib.repmat(ex_widths.T,1,gt_count)
        ex_heights = numpy.matlib.repmat(ex_heights.T,1,gt_count)
        ex_ctr_x = numpy.matlib.repmat(ex_ctr_x.T,1,gt_count)
        ex_ctr_y = numpy.matlib.repmat(ex_ctr_y.T,1,gt_count)

        # regression targets
        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = np.log(gt_widths / ex_widths)
        targets_dh = np.log(gt_heights / ex_heights)

        targets = np.zeros((boxes_count,gt_count, 4), dtype=np.float32)
        targets[:,:,0] = targets_dx
        targets[:,:,1] = targets_dy
        targets[:,:,2] = targets_dw
        targets[:,:,3] = targets_dh

        return targets

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        if cfg.TRAIN.BBOX_REG:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1

    def train_model(self, max_iters):
        """Network training loop."""
        print 'Max iters:',max_iters
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

def get_DOM_regres_training_roidb(imdb):
    roidb = imdb.gt_roidb()

    box_count = 5

    # define coordinates of max box
    x_min,x_max = 0,1920-1
    y_min,y_max = 0,1000-1

    for i in xrange(len(imdb.image_index)):
        # add image
        roidb[i]['image'] = imdb.image_path_at(i)
        
        # change key names
        roidb[i]['gt_boxes'] = roidb[i]['boxes']
        roidb[i].pop("gt_overlaps", None)
        roidb[i].pop("boxes", None)

        # get the smallest area where are all the boxes
        mins = np.min(roidb[i]['gt_boxes'][:,0:2],axis=0)
        maxs = np.max(roidb[i]['gt_boxes'][:,2:4],axis=0)
        b_x_min,b_y_min = mins[0],mins[1]
        b_x_max,b_y_max = maxs[0],maxs[1]

        # enlarge max box if needed (we need to move only y_max)
        y_max = max(y_max,b_y_max)

        # generate boxes that include all gt_boxes
        x_mins = np.random.randint(low=x_min, high=b_x_min+1, size=box_count)
        y_mins = np.random.randint(low=y_min, high=b_y_min+1, size=box_count)
        x_maxes = np.random.randint(low=b_x_max, high=x_max+1, size=box_count)
        y_maxes = np.random.randint(low=b_y_max, high=y_max+1, size=box_count)

        boxes = np.zeros((box_count,4),dtype=np.float)
        boxes[:,0] = x_mins
        boxes[:,1] = y_mins
        boxes[:,2] = x_maxes
        boxes[:,3] = y_maxes

        roidb[i]['boxes'] = boxes

    return roidb

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
