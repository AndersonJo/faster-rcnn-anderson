import cv2

from frcnn.augment import horizontal_flip, vertical_flip, rotate, invert, gaussian_blur
from frcnn.voc import PascalVocData
from tests import DATASET_ROOT_PATH


def test_augmentation():
    vocdata = PascalVocData(DATASET_ROOT_PATH)
    train, test, class_mapping = vocdata.load_data(limit_size=30, add_bg=True)

    # Test Horizontal Flip
    meta = train[0]
    image = cv2.imread(meta['image_path'])
    image, meta = horizontal_flip(image, meta)

    for _, x1, y1, x2, y2 in meta['objects']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/horizontal.png', image)

    # Test Vertical Flip
    meta = train[2]
    image = cv2.imread(meta['image_path'])
    image, meta = vertical_flip(image, meta)
    for _, x1, y1, x2, y2 in meta['objects']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/vertical.png', image)

    # Test Ratate 0
    meta = train[3]
    image = cv2.imread(meta['image_path'])
    image, meta = rotate(image, meta, 0)
    for _, x1, y1, x2, y2 in meta['objects']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/rotate_0.png', image)

    # Test Ratate 90
    meta = train[4]
    image = cv2.imread(meta['image_path'])
    image, meta = rotate(image, meta, 90)
    for _, x1, y1, x2, y2 in meta['objects']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/rotate_90.png', image)

    # Test Ratate 180
    meta = train[5]
    image = cv2.imread(meta['image_path'])
    image, meta = rotate(image, meta, 180)
    for _, x1, y1, x2, y2 in meta['objects']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/rotate_180.png', image)

    # Test Ratate 270
    meta = train[6]
    image = cv2.imread(meta['image_path'])
    image, meta = rotate(image, meta, 270)
    for _, x1, y1, x2, y2 in meta['objects']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/rotate_270.png', image)

    # Test Invert Red
    meta = train[7]
    image = cv2.imread(meta['image_path'])
    image = invert(image, 'r')
    for _, x1, y1, x2, y2 in meta['objects']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/invert_r.png', image)

    # Test Invert Green
    meta = train[8]
    image = cv2.imread(meta['image_path'])
    image = invert(image, 'g')
    for _, x1, y1, x2, y2 in meta['objects']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/invert_g.png', image)

    # Test Invert Blue
    meta = train[9]
    image = cv2.imread(meta['image_path'])
    image = invert(image, 'b')
    for _, x1, y1, x2, y2 in meta['objects']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/invert_b.png', image)

    # Test Invert
    meta = train[10]
    image = cv2.imread(meta['image_path'])
    image = invert(image, 'rgb')
    for _, x1, y1, x2, y2 in meta['objects']:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/invert.png', image)

    # Test Gaussian Blur
    for i in range(100):
        meta = train[i]
        image = cv2.imread(meta['image_path'])
        image = gaussian_blur(image)
        for _, x1, y1, x2, y2 in meta['objects']:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imwrite('temp/gaussian_blur.png', image)
