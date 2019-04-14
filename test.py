#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 17:46:07 2018

@author: zzq
"""

import numpy as np
import tensorflow as tf
import cv2
import os
from yolo import Yolo
from visualization import plt_bboxes

class test(object):
    def __init__(self, weights_file):
        self.verbose = True
        # detection params
        self.S = 7  # cell size
        self.B = 2  # boxes_per_cell
        self.classes = ["recoon","other","other","other","other","other","other","other",
                        "other","other","other","other","other","other","other","other","other","other","other","other"]
        self.weights_file =weights_file
        self.C = len(self.classes) # number of classes
        # offset for box center (top left point of each cell)
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(self.S)]*self.S*self.B),
                                              [self.B, self.S, self.S]), [1, 2, 0])
    		#将第0维放置第三维
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])

        self.threshold = 0.2  # confidence scores threshold
        self.iou_threshold = 0.5
        self.model_path = 'model'
        self.sess = tf.Session()
        self.net = Yolo()
        self.predicts = self.net.logits
        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.total_loss = 0.0
        self.total_TP=0;
        self.total_p = 0;
        
        self.ckpt_file = os.path.join(self.model_path,'yolo')
        gpuConfig = tf.ConfigProto(device_count={'gpu':0})
        #gpuConfig.gpu_options.allow_growth = True
        #gpu_options = tf.GPUOptions()
        #config = tf.ConfigProto(gpu_options=gpuConfig)
        self.sess = tf.Session(config=gpuConfig)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.model_path) #获取checkpoints对象
        if ckpt and ckpt.model_checkpoint_path:##判断ckpt是否为空，若不为空，才进行模型的加载，否则从头开始训练
            print('Restoring weights from: ' + ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)#恢复保存的神经网络结构，实现断点续训
        elif self.weights_file is not None:
           print('Restoring weights from: ' + self.weights_file)
           self.saver.restore(self.sess, self.weights_file) 
        

    def detect_from_file(self, image_file, imshow=True, deteted_boxes_file="boxes.txt",
                     detected_image_file="detected_image.jpg"):
        """Do detection given a image file"""
        # read image
        image = cv2.imread(image_file)
        
        img_h, img_w, _ = image.shape
        predicts = self._detect_from_image(image)
        predict_boxes = self._interpret_predicts(predicts, img_h, img_w)
        classes = np.array([i[0] for i in predict_boxes])
        boxes =  np.array([i[1:5] for i in predict_boxes])
        scores =  np.array([i[5] for i in predict_boxes])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #plt_bboxes(image,classes,scores,boxes )
        self.show_results(image, predict_boxes, imshow, deteted_boxes_file, detected_image_file)

    def _detect_from_image(self, image):
        """Do detection given a cv image"""
        img_resized = cv2.resize(image, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) #opencv习惯使用BGR，将其转换为RGB
        img_resized_np = np.asarray(img_RGB)  #将结构数据转化为ndarray numpy数组
        _images = np.zeros((1, 448, 448, 3), dtype=np.float32)
        _images[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        
        predicts = self.sess.run(self.predicts, feed_dict={self.net.images: _images})[0]
        return predicts

    def _interpret_predicts(self, predicts, img_h, img_w):
        """Interpret the predicts and get the detetction boxes"""
        idx1 = self.S*self.S*self.C
        idx2 = idx1 + self.S*self.S*self.B
        # class prediction
        class_probs = np.reshape(predicts[:idx1], [self.S, self.S, self.C])
        # confidence
        confs = np.reshape(predicts[idx1:idx2], [self.S, self.S, self.B])
        # boxes -> (x, y, w, h)
        boxes = np.reshape(predicts[idx2:], [self.S, self.S, self.B, 4])

        # convert the x, y to the coordinates relative to the top left point of the image
        boxes[:, :, :, 0] += self.x_offset
        boxes[:, :, :, 1] += self.y_offset
        boxes[:, :, :, :2] /= self.S      #将宽度除以S

        # the predictions of w, h are the square root
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        # multiply the width and height of image
        boxes[:, :, :, 0] *= img_w
        boxes[:, :, :, 1] *= img_h
        boxes[:, :, :, 2] *= img_w
        boxes[:, :, :, 3] *= img_h

        # class-specific confidence scores [S, S, B, C]
        scores = np.expand_dims(confs, -1) * np.expand_dims(class_probs, 2)
    		#在axis=second param 处添加数据
        scores = np.reshape(scores, [-1, self.C]) # [S*S*B, C]
        boxes = np.reshape(boxes, [-1, 4])        # [S*S*B, 4]

        # filter the boxes when score < threhold
        scores[scores < self.threshold] = 0.0

        # non max suppression
        self._non_max_suppression(scores, boxes)

        # report the boxes
        predict_boxes = [] # (class, x, y, w, h, scores)
        max_idxs = np.argmax(scores, axis=1) #找到每一列（每一种）最大值所在的行
        for i in range(len(scores)):
            max_idx = max_idxs[i]
            if scores[i, max_idx] > 0.0:
                predict_boxes.append((max_idx, boxes[i, 0], boxes[i, 1],
                                      boxes[i, 2], boxes[i, 3], scores[i, max_idx]))
        return predict_boxes

    def _non_max_suppression(self, scores, boxes):
        """Non max suppression"""
        # for each class
        for c in range(self.C):
            sorted_idxs = np.argsort(scores[:, c]) #将每此种类框排序
            last = len(sorted_idxs) - 1
            while last > 0:
                if scores[sorted_idxs[last], c] < 1e-6:
                    break
                for i in range(last):
                    if scores[sorted_idxs[i], c] < 1e-6:
                        continue
                    if self._iou(boxes[sorted_idxs[i]], boxes[sorted_idxs[last]]) > self.iou_threshold:
                        scores[sorted_idxs[i], c] = 0.0
                last -= 1

    def _iou(self, box1, box2):
        """Compute the iou of two boxes"""
        #与普通不同
        inter_w = np.minimum(box1[0]+0.5*box1[2], box2[0]+0.5*box2[2]) - \
                  np.maximum(box1[0]-0.5*box2[2], box2[0]-0.5*box2[2])
        inter_h = np.minimum(box1[1]+0.5*box1[3], box2[1]+0.5*box2[3]) - \
                  np.maximum(box1[1]-0.5*box2[3], box2[1]-0.5*box2[3])
        if inter_h < 0 or inter_w < 0:
            inter = 0
        else:
            inter = inter_w * inter_h
        union = box1[2]*box1[3] + box2[2]*box2[3] - inter
        return inter / union

    def show_results(self, image, results, imshow=True, deteted_boxes_file=None,
                     detected_image_file=None):
        """Show the detection boxes"""
        img_cp = image.copy()
        if deteted_boxes_file:
            f = open(deteted_boxes_file, "w")
        #  draw boxes
        for i in range(len(results)):
            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3]) // 2
            h = int(results[i][4]) // 2
            if self.verbose:
                print("   class: %s, [x, y, w, h]=[%d, %d, %d, %d], confidence=%f" % (results[i][0],
                            x, y, w, h, results[i][-1]))
 
                cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
				#第一个参数：img是原图第二个参数：（x，y）是矩阵的左上点坐标第三个参数：（x+w，y+h）是矩阵的右下点坐标
                #第四个参数：（0,255,0）是画线对应的rgb颜色 第五个参数：2是所画的线的宽度
                cv2.putText(img_cp, self.classes[results[i][0]] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if deteted_boxes_file:
                f.write(self.classes[results[i][0]] + ',' + str(x) + ',' + str(y) + ',' +
                        str(w) + ',' + str(h)+',' + str(results[i][5]) + '\n')
        if imshow:
            cv2.imshow('YOLO_small detection', img_cp)
            cv2.waitKey(0)
        if detected_image_file:
            cv2.imwrite(detected_image_file, img_cp)
        if deteted_boxes_file:
            f.close()

if __name__ == "__main__":
    tf.reset_default_graph()
    test = test("YOLO_small.ckpt")
    test.detect_from_file("cat.jpg")


