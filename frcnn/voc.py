# -*- coding:utf-8 -*-
import os
import xml.etree.ElementTree as ET


class PascalVocData(object):
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
        self.voc_root_path: str = voc_root_path
        self.voc_names: list = voc_names

        self._init()

    def _init(self) -> None:
        """
        Initialization method
        """
        # VOC data absolute paths
        # i.e. ['/data/VOCdevkit/VOC2007', '/data/VOCdevkit/VOC2012']
        self.voc_paths = [os.path.join(self.voc_root_path, voc_name) for voc_name in self.voc_names]
        self.voc_paths = list(filter(lambda p: os.path.exists(p), self.voc_paths))

        # train and test image absolute paths
        train_files = list()
        test_files = list()
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
            #
            # img_root_path = os.path.join(voc_path, 'JPEGImages')
            # for img_name in os.listdir(img_root_path):
            #     img_path = os.path.join(img_root_path, img_name)
            #     if img_name in _test_file_names:
            #         test_files.append(img_path)
            #     else:
            #         train_files.append(img_path)
        print(_test_file_names)
        # Annotation absolute paths
        # i.e. ['/data/VOCdevkit/VOC2007/Annotations/004134.xml', ...]

        # JPEG Image absolute paths
        image_paths = list()
        for voc_path in self.voc_paths:
            _p = os.path.join(voc_path, 'JPEGImages')
            image_paths += [os.path.join(_p, n) for n in os.listdir(_p)]

        # Make annotation data
        annotation_paths = list()
        for voc_path in self.voc_paths:
            _annot_path = os.path.join(voc_path, 'Annotations')
            annotation_paths += [os.path.join(_annot_path, a) for a in os.listdir(_annot_path)]

            for annot_path in annotation_paths:
                annot = self.parse_annotation(annot_path)
                print(annot)
                break

    def parse_annotation(self, annot_path: str) -> dict:
        """
        It parses XML VOC annotation file.
        :param annot_path: full path of annotation file
        :return annotation information as dictionary
        """
        annot = dict()

        # Set default information
        et: ET.ElementTree = ET.parse(annot_path)
        el: ET.ElementTree = et.getroot()

        annot['filename'] = el.find('filename').text
        annot['width'] = int(el.find('size').find('width').text)
        annot['height'] = int(el.find('size').find('height').text)
        annot['objects'] = list()

        object_els = el.findall('object')
        for el_object in object_els:
            obj = dict()

            obj['name'] = el_object.find('name').text
            bbox = el_object.find('bndbox')
            obj['xmin'] = int(bbox.find('xmin').text)
            obj['ymin'] = int(bbox.find('ymin').text)
            obj['xmax'] = int(bbox.find('xmax').text)
            obj['ymax'] = int(bbox.find('ymax').text)
            annot['objects'].append(obj)

        return annot
