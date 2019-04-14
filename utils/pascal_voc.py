import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import config as cfg


class pascal_voc(object):
    def __init__(self, phase, rebuild=False):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes)))) #zip将两个列表 变成字典
        self.flipped = cfg.FLIPPED
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None    #
        self.prepare()

    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels
    '''从文件图片读取像素'''
    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image
    '''加载图片标注信息和像素信息并是否使用图片翻转'''  
    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] =\
                    gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 -\
                                gt_labels_cp[idx]['label'][i, j, 1]     #boxes的中点的X坐标
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)                                   #多维矩阵中，只对第一维（行
        self.gt_labels = gt_labels
        return gt_labels
    #如果文件。pkl存在，则直接导入 如果不存在则先写入在导出 gtlabels包括文件名 label label包括 label boxes
    def load_labels(self):
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl') 

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'train_label.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test_label.txt')
        with open(txtname, 'r') as f:
            self.image_info = f.readlines()

        gt_labels = []
        for info in self.image_info:
            info= info.split()
            label, num = self.load_pascal_annotation(info)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages',info[0][25:])
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels
    #一个label包括boxes 种类
    def load_pascal_annotation(self, info):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        imname = os.path.join(self.data_path, 'JPEGImages', info[0][25:])
        print('flag',info[0][18:])
        print(imname)
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 25))
        #filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        #tree = ET.parse(filename)
        #objs = tree.findall('object')
        n_box = len(info[1:]) // 5
        for i in range(n_box):
            #bbox = obj.find('bndbox')
            # Make pixel indexes 0-based  boxes为其实际坐标将原图缩小后
            x1 = max(min((float(info[1+i*5]) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(info[2+i*5]) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(info[3+i*5]) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(info[4+i*5]) - 1) * h_ratio, self.image_size - 1), 0)
            cls_ind = 0
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)  #如果已经有了目标 则跳过 否则加入目标
            if label[y_ind, x_ind, 0] == 1:
                continue                                   
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, n_box
