# -*- coding: utf-8 -*-
"""
Yolo V1 by tensorflow
"""

import numpy as np
import tensorflow as tf

import config as cfg

def leak_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)

class Yolo(object):
    def __init__(self,is_training = True):
        self.verbose = True
        # detection params
        self.S = cfg.CELL_SIZE  # cell size
        self.B = cfg.BOXES_PER_CELL  # boxes_per_cell
        self.classes = ["recoon","other","other","other","other","other","other","other",
                        "other","other","other","other","other","other","other","other","other","other","other","other"]
        self.image_size = cfg.IMAGE_SIZE
        self.C = len(self.classes) # number of classes
        # offset for box center (top left point of each cell)
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)]*self.S*self.B),
                                              [self.B, self.S, self.S]), [1, 2, 0])
		#将第0维放置第三维
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])
        self.boxes_per_cell= cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD  # confidence scores threshold
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.batch_size = cfg.BATCH_SIZE
        self.class_scale = cfg.CLASS_SCALE
        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.coord_scale = cfg.COORD_SCALE
        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        self.logits = self._build_net()

        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.S, self.S, 5 + 20])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)
    def _build_net(self):
        """build the network"""
        if self.verbose:
            print("Start to build the network ...")
        self.images = tf.placeholder(tf.float32, [None, 448, 448, 3])
        net = self._conv_layer(self.images, 1, 64, 7, 2)
        net = self._maxpool_layer(net, 1, 2, 2)
        net = self._conv_layer(net, 2, 192, 3, 1)
        net = self._maxpool_layer(net, 2, 2, 2)
        net = self._conv_layer(net, 3, 128, 1, 1)
        net = self._conv_layer(net, 4, 256, 3, 1)
        net = self._conv_layer(net, 5, 256, 1, 1)
        net = self._conv_layer(net, 6, 512, 3, 1)
        net = self._maxpool_layer(net, 6, 2, 2)
        net = self._conv_layer(net, 7, 256, 1, 1)
        net = self._conv_layer(net, 8, 512, 3, 1)
        net = self._conv_layer(net, 9, 256, 1, 1)
        net = self._conv_layer(net, 10, 512, 3, 1)
        net = self._conv_layer(net, 11, 256, 1, 1)
        net = self._conv_layer(net, 12, 512, 3, 1)
        net = self._conv_layer(net, 13, 256, 1, 1)
        net = self._conv_layer(net, 14, 512, 3, 1)
        net = self._conv_layer(net, 15, 512, 1, 1)
        net = self._conv_layer(net, 16, 1024, 3, 1)
        net = self._maxpool_layer(net, 16, 2, 2)
        net = self._conv_layer(net, 17, 512, 1, 1)
        net = self._conv_layer(net, 18, 1024, 3, 1)
        net = self._conv_layer(net, 19, 512, 1, 1)
        net = self._conv_layer(net, 20, 1024, 3, 1)
        net = self._conv_layer(net, 21, 1024, 3, 1)
        net = self._conv_layer(net, 22, 1024, 3, 2)
        net = self._conv_layer(net, 23, 1024, 3, 1)
        net = self._conv_layer(net, 24, 1024, 3, 1)
        net = self._flatten(net)
        net = self._fc_layer(net, 25, 512, activation=leak_relu)
        net = self._fc_layer(net, 26, 4096, activation=leak_relu)
        net = self._fc_layer(net, 27, self.S*self.S*(self.C+5*self.B))
        return net

    def _conv_layer(self, x, id, num_filters, filter_size, stride):
        """Conv layer"""
		#上一层的输出
        in_channels = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size,
                                                  in_channels, num_filters], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_filters,]))
        # padding, note: not using padding="SAME"
        pad_size = filter_size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
		#1维 2 维 3维
        x_pad = tf.pad(x, pad_mat)
        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID")
		#strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
        output = leak_relu(tf.nn.bias_add(conv, bias))
        if self.verbose:
            print("    Layer %d: type=Conv, num_filter=%d, filter_size=%d, stride=%d, output_shape=%s" \
                  % (id, num_filters, filter_size, stride, str(output.get_shape())))
        return output

    def _fc_layer(self, x, id, num_out, activation=None):
        """fully connected layer"""
        num_in = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1))
        bias = tf.Variable(tf.zeros([num_out,]))  #逗号
        output = tf.nn.xw_plus_b(x, weight, bias)
        if activation:
            output = activation(output)
        if self.verbose:
            print("    Layer %d: type=Fc, num_out=%d, output_shape=%s" \
                  % (id, num_out, str(output.get_shape())))
        return output

    def _maxpool_layer(self, x, id, pool_size, stride):
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                                strides=[1, stride, stride, 1], padding="SAME")
        if self.verbose:
            print("    Layer %d: type=MaxPool, pool_size=%d, stride=%d, output_shape=%s" \
                  % (id, pool_size, stride, str(output.get_shape())))
        return output

    def _flatten(self, x):
        """flatten the x"""
        tran_x = tf.transpose(x, [0, 3, 1, 2])  # channle first mode
        nums = np.product(x.get_shape().as_list()[1:]) #乘积
        return tf.reshape(tran_x, [-1, nums])

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
    
    def loss_layer(self, predicts, labels):
        idx1 = self.S*self.S*self.C
        idx2 = idx1 + self.S*self.S*self.B
        predict_classes = tf.reshape(
                    predicts[:, :idx1],
                    [self.batch_size, self.S, self.S, self.C])
        predict_scales = tf.reshape(
                    predicts[:, idx1:idx2],
                    [self.batch_size, self.S, self.S, self.boxes_per_cell])
        predict_boxes = tf.reshape(
                    predicts[:, idx2:],
                    [self.batch_size, self.S, self.S, self.boxes_per_cell, 4])
    
        response = tf.reshape(
                    labels[..., 0],
                    [self.batch_size, self.S, self.S, 1])
        boxes = tf.reshape(
                    labels[..., 1:5],
                    [self.batch_size, self.S, self.S, 1, 4])
                # 扩张张量复制张量 并归一化
        boxes = tf.tile(
                    boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
        classes = labels[..., 5:]
        self.x_offset = tf.reshape(
                tf.constant(self.x_offset, dtype=tf.float32),
                [1, self.S, self.S, self.boxes_per_cell])
        self.x_offset = tf.tile(self.x_offset, [self.batch_size, 1, 1, 1])
        self.y_offset_tran = tf.transpose(self.x_offset, (0, 2, 1, 3))
        predict_boxes_tran = tf.stack(
                    [(predict_boxes[..., 0] + self.x_offset) / self.S,
                     (predict_boxes[..., 1] + self.y_offset) / self.S,
                     tf.square(predict_boxes[..., 2]),
                     tf.square(predict_boxes[..., 3])], axis=-1)
    
        iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)
    
                # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
                #获得指定维最大值
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast(
                    (iou_predict_truth >= object_mask), tf.float32) * response
    
                # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(
                    object_mask, dtype=tf.float32) - object_mask
                #boxes/（imagesize/cellsize）
        boxes_tran = tf.stack(
                    [boxes[..., 0] * self.S - self.x_offset,
                     boxes[..., 1] * self.S - self.y_offset,
                     tf.sqrt(boxes[..., 2]),
                     tf.sqrt(boxes[..., 3])], axis=-1)
    
                # class_loss
        class_delta = response * (predict_classes - classes)
        class_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                    name='class_loss') * self.class_scale
    
                # object_loss
        object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                    name='object_loss') * self.object_scale
    
                # noobject_loss
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                    name='noobject_loss') * self.noobject_scale
    
                # coord_loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                    name='coord_loss') * self.coord_scale
    
        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)
        
        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('noobject_loss', noobject_loss)
        tf.summary.scalar('coord_loss', coord_loss)
        
        tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
        tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
        tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
        tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
        tf.summary.histogram('iou', iou_predict_truth)
    
if __name__ == "__main__":
    yolo_net = Yolo()
    



