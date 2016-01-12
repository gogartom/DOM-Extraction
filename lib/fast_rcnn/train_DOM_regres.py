# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""
import caffe
from fast_rcnn.test_DOM_regres import test_net
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import numpy.matlib
import os
import sys

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, imdb_test, imdb_test_name, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.imdb_test = imdb_test
        self.imdb_test_name = imdb_test_name

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
        num_classes = roidb[0]['gt_elements'].shape[0] + 1

        #-- Get targets
        for im_i in xrange(num_images):
            targets = roidb[im_i]['gt_elements']
            target_classes = roidb[im_i]['gt_classes']
            boxes = roidb[im_i]['boxes']
            roidb[im_i]['DOM_bbox_targets'] = self._compute_targets(targets, target_classes, boxes)

        #-- Compute values needed for means and stds
        #-- var(x) = E(x^2) - E(x)^2
        sums = np.zeros((num_classes, 4))
        squared_sums = np.zeros((num_classes, 4))
        class_counts = np.zeros((num_classes, 1)) + cfg.EPS
        #boxes_counts = 0     
        for im_i in xrange(num_images):
            targets = roidb[im_i]['DOM_bbox_targets']
            target_classes = roidb[im_i]['gt_classes']
            #boxes_counts += targets.shape[0]
            include_classes = roidb[im_i]['include_gt_elements']

            for cls in xrange(1, num_classes):
                gt_element_ind = np.where(target_classes == cls)[0][0]
                boxes_inds = np.where(include_classes[:, gt_element_ind])[0]

                if boxes_inds.size > 0:                
                    class_counts[cls] += boxes_inds.size
                    sums[cls, :] += targets[boxes_inds,gt_element_ind,:].sum(axis=0)
                    squared_sums[cls, :] += (targets[boxes_inds,gt_element_ind,:] ** 2).sum(axis=0)
        
        means = sums / class_counts
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

        #-- Normalize targets
        for im_i in xrange(num_images):
            targets = roidb[im_i]['DOM_bbox_targets']
            include_classes = roidb[im_i]['include_gt_elements']

            for cls in xrange(1, num_classes):

                gt_element_ind = np.where(target_classes == cls)[0][0]
                boxes_inds = np.where(include_classes[:, gt_element_ind])[0]

                roidb[im_i]['DOM_bbox_targets'][boxes_inds,gt_element_ind, :] -= means[cls, :]
                roidb[im_i]['DOM_bbox_targets'][boxes_inds,gt_element_ind, :] /= stds[cls, :]


        return means.ravel(), stds.ravel()

    def _compute_targets(self, gt, gt_classes, boxes):
        
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

        iters_dir = os.path.join(self.output_dir, 'iters')
        if not os.path.exists(iters_dir):
            os.makedirs(iters_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(iters_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1

    def test_model(self):
        iters_dir = os.path.join(self.output_dir, 'iters')
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = str(os.path.join(iters_dir, filename))

        print 'Testing net using net at:', filename        
        # TODO: add test.prototxt to params
        net = caffe.Net('models/CaffeNet_DOM_regres/test.prototxt', filename, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(filename))[0]

        precision, aver_IOU, aver_dist = test_net(net, self.imdb_test, self.imdb_test_name)
        print 'Precision:', 'Price:',precision[0],'| Main image:',precision[1],'| Name:', precision[2]
        print 'Average_IOU:', 'Price:',aver_IOU[0],'| Main image:',aver_IOU[1],'| Name:', aver_IOU[2]
        print 'Average_Distance:', 'Price:',aver_dist[0],'| Main image:',aver_dist[1],'| Name:', aver_dist[2]
        sys.stdout.flush()

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
                self.test_model()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()
            self.test_model()

def get_non_overlaping_elements(boxes, overlaps):
    inds_to_delete = overlaps.max(axis=1).nonzero()[0]
    boxes = np.delete(boxes,inds_to_delete, axis=0)
    return boxes

# returns bounds for generating custom boxes
def get_bounds(gt_elements, gt_classes, single_element=None):

    num_elements = len(gt_elements)
    contain_boxes = np.zeros((0,4))
    exclude_boxes = np.zeros((0,4))

    for el_ind in xrange(num_elements):
        #gt_ind = np.where(gt_classes==cls)[0][0]

        # if add all elements or particular element
        if (single_element is None or single_element == el_ind):
            contain_boxes = np.vstack((contain_boxes,np.array(gt_elements[el_ind,:])))  
        else:
            exclude_boxes = np.vstack((exclude_boxes,np.array(gt_elements[el_ind,:])))


    # Compute inner bounds
    # get the smallest area where are all contained boxes
    inner_bounds = [np.min(contain_boxes[:,0],axis=0),
                    np.min(contain_boxes[:,1],axis=0),
                    np.max(contain_boxes[:,2],axis=0),
                    np.max(contain_boxes[:,3],axis=0)]

    # Computer outer edges
    # right_exclude_edges left from left_inner_edge
    outer_x_mins = exclude_boxes[exclude_boxes[:,2]<inner_bounds[0],2].tolist()
    # left_exclude_edges right from right_inner_edge
    outer_x_maxs = exclude_boxes[exclude_boxes[:,0]>inner_bounds[2],0].tolist()
    # bottom_exclude_edges up from top_inner_edge
    outer_y_mins = exclude_boxes[exclude_boxes[:,3]<inner_bounds[1],3].tolist()
    # top_exclude_edges down from bottom_inner_edge
    outer_y_maxs = exclude_boxes[exclude_boxes[:,1]>inner_bounds[3],1].tolist()

    # Prepare absolute bounds 
    abs_x_min,abs_x_max = 0,1920-1
    abs_y_min,abs_y_max = 0,1000-1

    # if there is y_max larger then abs_max
    # enlarge abs_max box (we need to check only y_max)
    if(len(outer_y_maxs)>0):
        abs_y_max = max(abs_y_max,max(outer_y_maxs))
        

    # Add absolute bounds and get the right value
    outer_x_mins.append(abs_x_min)
    outer_y_mins.append(abs_y_min)
    outer_x_maxs.append(abs_x_max)
    outer_y_maxs.append(abs_y_max)
  
    # Get bounds
    outer_x_min = max(outer_x_mins)
    outer_y_min = max(outer_y_mins)
    outer_x_max = min(outer_x_maxs)
    outer_y_max = min(outer_y_maxs)

    # Check that outer_max_y is lower than inner_max_y
    outer_y_max = max(outer_y_max,inner_bounds[3])

    # save outer bounds
    outer_bounds = [outer_x_min, outer_y_min, outer_x_max, outer_y_max]

    return inner_bounds, outer_bounds

def generate_boxes(gt_elements, gt_classes, box_count, single_element = None):
    inner_bounds, outer_bounds = get_bounds(gt_elements, gt_classes, single_element)
  
    x_mins = np.random.randint(low=outer_bounds[0], high=inner_bounds[0]+1, size=box_count)
    y_mins = np.random.randint(low=outer_bounds[1], high=inner_bounds[1]+1, size=box_count)
    x_maxes = np.random.randint(low=inner_bounds[2], high=outer_bounds[2]+1, size=box_count)
    y_maxes = np.random.randint(low=inner_bounds[3], high=outer_bounds[3]+1, size=box_count)

    boxes = np.zeros((box_count,4),dtype=np.float)
    
    boxes[:,0] = x_mins
    boxes[:,1] = y_mins
    boxes[:,2] = x_maxes
    boxes[:,3] = y_maxes
    return boxes

def get_DOM_regres_training_roidb(imdb):
    gt_roidb = imdb.gt_roidb()
    roidb = imdb.roidb    
    my_roidb = []

    box_count = 10

    # define coordinates of max box
    x_min,x_max = 0,1920-1
    y_min,y_max = 0,1000-1

    for i in xrange(len(imdb.image_index)):
        my_roidb.append({})

        # add image
        my_roidb[i]['image'] = imdb.image_path_at(i)
        
        # copy gt boxes and their classes
        my_roidb[i]['gt_elements'] = gt_roidb[i]['boxes']
        my_roidb[i]['gt_classes'] = gt_roidb[i]['gt_classes']
        my_roidb[i]['flipped'] = gt_roidb[i]['flipped']

        # add non overlaping elements
        my_roidb[i]['non_overlap_elements'] =   \
            get_non_overlaping_elements(roidb[i]['boxes'],roidb[i]['gt_overlaps'])

        # COMPUTE BOXES
        gt_count = my_roidb[i]['gt_elements'].shape[0]
        my_roidb[i]['boxes'] = generate_boxes(my_roidb[i]['gt_elements'],my_roidb[i]['gt_classes'],box_count,)
        my_roidb[i]['include_gt_elements'] = np.ones((box_count, gt_count), dtype=bool) #[my_roidb[i]['gt_classes']]*box_count

        # TODO: IF SINGLE_CLASS_BOXES        
        if cfg.TRAIN.SINGLE_CLASS_BOXES:
            for j in xrange(gt_count):
                single_boxes = generate_boxes(my_roidb[i]['gt_elements'],my_roidb[i]['gt_classes'], box_count, single_element=j)
                include_classes = np.zeros((box_count, gt_count), dtype=bool)
                include_classes[:,j] = True
                my_roidb[i]['boxes'] = np.vstack((my_roidb[i]['boxes'],single_boxes))
                my_roidb[i]['include_gt_elements'] = np.vstack((my_roidb[i]['include_gt_elements'],include_classes))
        
    return my_roidb


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

def train_net(solver_prototxt, roidb, imdb_test, imdb_test_name, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb, imdb_test, imdb_test_name, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
