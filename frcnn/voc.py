# -*- coding:utf-8 -*-
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from os.path import dirname
from typing import Tuple

import cv2


class BaseData(ABC):

    @abstractmethod
    def load_data(self):
        raise NotImplementedError


class PascalVocData(BaseData):
    # VOC2007 uses test.txt as test data
    # VOC2017 uses val.txt as test data
    TEST_FILES = ('test.txt', 'val.txt')

    def __init__(self, voc_root_path: str, voc_names: tuple = ('VOC2007', 'VOC2012')):
        """
        :param voc_root_path: the path of "VOCdevkit" including VOC2007 or VOC2012 (i.e. '/data/VOCdevkit')
        :param voc_names: VOC challenge data (i.e. ('VOC2007', 'VOC20010', 'VOC2012'))

        voc_root_path path should look like this
        =======================================
        ├── VOC2007
        │   ├── Annotations
        │   ├── ImageSets
        │   │   ├── Layout
        │   │   ├── Main
        │   │   └── Segmentation
        │   ├── JPEGImages
        │   ├── SegmentationClass
        │   └── SegmentationObject
        ├── VOC2012
        │   ├── Annotations
        │   ├── ImageSets
        │   │   ├── Action
        │   │   ├── Layout
        │   │   ├── Main
        │   │   └── Segmentation
        │   ├── JPEGImages
        │   ├── SegmentationClass
        │   └── SegmentationObject
        =======================================
        """
        self.voc_root_path = voc_root_path
        self.voc_names = voc_names

    def load_data(self, limit_size: int = None, need_test: bool = True, add_bg: bool = True) -> Tuple[list, list, dict]:
        """
        Initialization method
        :param limit_size: limit the size of dataset. only use it for debug.
        :param need_test: if True it returns a list of test dataset. if not it returns an empty list.
        :param add_bg: if True Add 'background' class as 0

        :return
            - train, test: VOC data
            - classes: class mapping dictionary {class name: index}
                       You can use classes variable when training detector network
        """
        # VOC data absolute paths
        # i.e. ['/data/VOCdevkit/VOC2007', '/data/VOCdevkit/VOC2012']
        self.voc_paths = [os.path.join(self.voc_root_path, voc_name) for voc_name in self.voc_names]
        self.voc_paths = list(filter(lambda p: os.path.exists(p), self.voc_paths))

        # Read test file names from test.txt or val.txt
        # _test_file_names: it only has list of image file names; not as absolute paths
        _test_file_names = list()
        for voc_path in self.voc_paths:
            _test = [os.path.join(voc_path, 'ImageSets', 'Main', t) for t in self.TEST_FILES]
            _test = list(filter(lambda t: os.path.exists(t), _test))

            if len(_test) <= 0:
                continue
            _test = _test[0]

            with open(_test) as f:
                for line in f:
                    _test_file_names.append(line.strip() + '.jpg')

        # Make annotation data
        train = list()
        test = list()
        classes = dict()
        if add_bg:
            classes['bg'] = 0

        for voc_path in self.voc_paths:
            if limit_size is not None and len(train) >= limit_size and len(test) >= limit_size:
                break

            _annot_dir_path = os.path.join(voc_path, 'Annotations')
            _annotation_paths = [os.path.join(_annot_dir_path, a) for a in os.listdir(_annot_dir_path)]

            for annot_path in _annotation_paths:
                annot = self.parse_annotation(annot_path)

                if need_test and annot['filename'] in _test_file_names:
                    test.append(annot)
                else:
                    train.append(annot)

                for o in annot['objects']:
                    cls_name = o[0]
                    if cls_name not in classes:
                        classes[cls_name] = len(classes)

        return train, test, classes

    @staticmethod
    def parse_annotation(annot_path: str) -> dict:
        """
        It parses XML VOC annotation file.
        :param annot_path: full path of annotation file
        :param validation: it should be used only for data integrity test
        :return annotation information as dictionary
        """
        annot = dict()

        # Set absolute path
        _img_dir_path = os.path.join(dirname(dirname(annot_path)), 'JPEGImages')

        # Set default information
        et = ET.parse(annot_path)
        el = et.getroot()

        annot['filename'] = el.find('filename').text
        annot['width'] = int(el.find('size').find('width').text)
        annot['height'] = int(el.find('size').find('height').text)
        annot['image_path'] = os.path.join(_img_dir_path, annot['filename'])  # absolute image path

        annot['objects'] = list()
        object_els = el.findall('object')
        for el_object in object_els:
            class_name = el_object.find('name').text

            bbox = el_object.find('bndbox')
            x1 = int(float(bbox.find('xmin').text))
            y1 = int(float(bbox.find('ymin').text))
            x2 = int(float(bbox.find('xmax').text))
            y2 = int(float(bbox.find('ymax').text))

            _object = [class_name, x1, y1, x2, y2]
            annot['objects'].append(_object)

        return annot

    @staticmethod
    def visualize_img(annot: dict, file: str = None):
        """
        The method creates bounding boxes in the target image
        and saves the image to disk if `file` string argument is provided

        :param annot: An annotation dictionary data
        :param file: output file name; where to save the image file?
        :return:
        """
        img = cv2.imread(annot['image'])
        for bbox in annot['objects']:
            cv2.rectangle(img, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']), (0, 0, 255))
        if file:
            cv2.imwrite(file, img)
        return img
