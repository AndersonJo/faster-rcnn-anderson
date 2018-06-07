import cv2
import numpy as np
import copy


def horizontal_flip(image, meta):
    rows, cols = image.shape[:2]

    image = cv2.flip(image, 1)

    for i, (_, x1, y1, x2, y2) in enumerate(meta['objects']):
        meta['objects'][i][3] = cols - x1
        meta['objects'][i][1] = cols - x2

    height, width, _ = image.shape
    meta['width'] = width
    meta['height'] = height
    return image, meta


def vertical_flip(image, meta):
    rows, cols = image.shape[:2]

    image = cv2.flip(image, 0)

    for i, (_, x1, y1, x2, y2) in enumerate(meta['objects']):
        meta['objects'][i][4] = rows - y1
        meta['objects'][i][2] = rows - y2

    height, width, _ = image.shape
    meta['width'] = width
    meta['height'] = height
    return image, meta


def rotate(image, meta, angle):
    rows, cols = image.shape[:2]

    if angle == 270:
        image = np.transpose(image, (1, 0, 2))
        image = cv2.flip(image, 0)
    elif angle == 180:
        image = cv2.flip(image, -1)
    elif angle == 90:
        image = np.transpose(image, (1, 0, 2))
        image = cv2.flip(image, 1)
    elif angle == 0:
        pass

    for i, (_, x1, y1, x2, y2) in enumerate(meta['objects']):

        if angle == 270:
            meta['objects'][i][1] = y1
            meta['objects'][i][3] = y2
            meta['objects'][i][2] = cols - x2
            meta['objects'][i][4] = cols - x1
        elif angle == 180:
            meta['objects'][i][3] = cols - x1
            meta['objects'][i][1] = cols - x2
            meta['objects'][i][4] = rows - y1
            meta['objects'][i][2] = rows - y2
        elif angle == 90:
            meta['objects'][i][1] = rows - y2
            meta['objects'][i][3] = rows - y1
            meta['objects'][i][2] = x1
            meta['objects'][i][4] = x2
        elif angle == 0:
            pass

    height, width, _ = image.shape
    meta['width'] = width
    meta['height'] = height
    return image, meta


def invert(image, mode=None):
    if mode == 'r':
        image[:, :, 2] = 255 - image[:, :, 2]
    elif mode == 'g':
        image[:, :, 1] = 255 - image[:, :, 1]
    elif mode == 'b':
        image[:, :, 0] = 255 - image[:, :, 0]
    elif mode == 'rgb':
        image = 255 - image

    return image


def gaussian_blur(image, sigma=0):
    try:
        image = cv2.GaussianBlur(image, (7, 7), sigma)
    except Exception as e:
        pass

    return image


def augment(image, meta):
    assert 'image_path' in meta
    assert 'objects' in meta
    assert 'width' in meta
    assert 'height' in meta
    choice = np.random.choice([0, 1, 2, 3], p=[0.2, 0.2, 0.3, 0.3])

    if choice == 0:
        image, meta = horizontal_flip(image, meta)

    elif choice == 1:
        image, meta = vertical_flip(image, meta)

    elif choice == 2:
        angle = np.random.choice([0, 90, 180, 270], 1)[0]
        image, meta = rotate(image, meta, angle)

    choice = np.random.choice([0, 1, 2], p=[0.05, 0.05, 0.9])
    if choice == 0:
        mode = np.random.choice(['r', 'g', 'b', 'rgb'], 1)[0]
        image = invert(image, mode)
    elif choice == 1:
        sigma = np.random.randint(0, 300)
        image = gaussian_blur(image, sigma)

    return image, meta
