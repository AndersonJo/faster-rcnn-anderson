import keras
from keras_resnet import models as resnet_models
import numpy as np
from keras import Input, Model
from keras.applications import ResNet50

from frcnn.tools import calculate_resnet50_output


def btest_fen_output():
    # config = singleton_config()
    # config.net_name = 'resnet50'
    # fen = FeatureExtractionNetwork(config)

    image_input = Input(shape=(None, None, 3), name='fen_input_image')
    model = ResNet50(include_top=False, input_tensor=image_input)
    model = Model(image_input, model.layers[-6].output)
    for i in range(100):
        width = np.random.randint(600, 1800)
        height = np.random.randint(600, 1800)

        image = np.random.rand(height, width, 3)
        image = np.expand_dims(image, 0)

        print('[{0}] {1} x {2} = '.format(i, height, width), model.predict(image).shape)

    """
    VGG16
        600 x 600: (1, 37, 37, 512)
    Resnet
        600 x 600, -1: (1, 19, 19, 2048)
        600 x 600, -2: (1, 19, 19, 2048)
        600 x 600, -3: (1, 19, 19, 2048)
        600 x 600, -4: (1, 19, 19, 2048)
        600 x 600, -8: (1, 19, 19, 2048)
        600 x 600, -10: (1, 19, 19, 2048)
        
        
        [0] 1787 x 791 =  (1, 56, 25, 512)
        [1] 1379 x 1437 =  (1, 43, 45, 512)
        [2] 1534 x 1795 =  (1, 48, 56, 512)
        [3] 626 x 1557 =  (1, 20, 49, 512)
        [4] 1132 x 800 =  (1, 36, 25, 512)
        [5] 1248 x 1644 =  (1, 39, 52, 512)
        [6] 1559 x 670 =  (1, 49, 21, 512)
        [7] 1716 x 949 =  (1, 54, 30, 512)
        [8] 964 x 1612 =  (1, 30, 51, 512)
        [9] 1503 x 1004 =  (1, 47, 32, 512)
        [10] 1027 x 1299 =  (1, 32, 41, 512)
        [11] 1504 x 916 =  (1, 47, 29, 512)
        [12] 1559 x 1499 =  (1, 49, 47, 512)
        [13] 1092 x 1758 =  (1, 34, 55, 512)
        [14] 1723 x 1164 =  (1, 54, 37, 512)
        [15] 1334 x 1631 =  (1, 42, 51, 512)
        [16] 735 x 1125 =  (1, 23, 36, 512)
        [17] 1739 x 1034 =  (1, 55, 33, 512)
        [18] 619 x 783 =  (1, 20, 25, 512)
        [19] 887 x 608 =  (1, 28, 19, 512)
        [20] 1480 x 667 =  (1, 47, 21, 512)
        [21] 1696 x 755 =  (1, 53, 24, 512)
        [22] 1186 x 1507 =  (1, 37, 47, 512)
        [23] 969 x 969 =  (1, 31, 31, 512)
        [24] 1200 x 952 =  (1, 38, 30, 512)
        [25] 649 x 839 =  (1, 21, 27, 512)
        [26] 884 x 1550 =  (1, 28, 49, 512)
        [27] 718 x 1415 =  (1, 23, 45, 512)
        [28] 664 x 698 =  (1, 21, 22, 512)
        [29] 830 x 795 =  (1, 26, 25, 512)
        [30] 977 x 994 =  (1, 31, 31, 512)
        [31] 1138 x 1103 =  (1, 36, 35, 512)
        [32] 1324 x 1027 =  (1, 42, 32, 512)
        [33] 1752 x 1314 =  (1, 55, 41, 512)
        [34] 1143 x 1690 =  (1, 36, 53, 512)
        [35] 674 x 662 =  (1, 21, 21, 512)
        [36] 1267 x 956 =  (1, 40, 30, 512)
        [37] 1730 x 1017 =  (1, 54, 32, 512)
        [38] 609 x 1284 =  (1, 19, 40, 512)
        [39] 1731 x 943 =  (1, 54, 30, 512)
        [40] 947 x 945 =  (1, 30, 30, 512)
        [41] 1415 x 1216 =  (1, 45, 38, 512)
        [42] 640 x 1058 =  (1, 20, 33, 512)
        [43] 919 x 1233 =  (1, 29, 39, 512)
        [44] 1588 x 1408 =  (1, 50, 44, 512)
        [45] 907 x 694 =  (1, 29, 22, 512)
        [46] 641 x 1051 =  (1, 20, 33, 512)
        [47] 969 x 798 =  (1, 31, 25, 512)
        [48] 828 x 1778 =  (1, 26, 56, 512)
        [49] 1536 x 729 =  (1, 48, 23, 512)
        [50] 1053 x 1276 =  (1, 33, 40, 512)
        [51] 1245 x 928 =  (1, 39, 29, 512)
        [52] 1735 x 779 =  (1, 55, 25, 512)
        [53] 1564 x 1212 =  (1, 49, 38, 512)
        [54] 1248 x 1292 =  (1, 39, 41, 512)
        [55] 1117 x 1323 =  (1, 35, 42, 512)
        [56] 1168 x 1136 =  (1, 37, 36, 512)
        [57] 1408 x 1490 =  (1, 44, 47, 512)
        [58] 626 x 1668 =  (1, 20, 52, 512)
        [59] 977 x 1242 =  (1, 31, 39, 512)
        [60] 1586 x 1337 =  (1, 50, 42, 512)
        [61] 677 x 711 =  (1, 22, 23, 512)
        [62] 1149 x 701 =  (1, 36, 22, 512)
        [63] 957 x 966 =  (1, 30, 31, 512)
        [64] 1519 x 860 =  (1, 48, 27, 512)
        [65] 1221 x 1274 =  (1, 39, 40, 512)
        [66] 1752 x 1666 =  (1, 55, 52, 512)
        [67] 1537 x 619 =  (1, 48, 20, 512)
        [68] 1391 x 1548 =  (1, 44, 49, 512)
        [69] 1318 x 1676 =  (1, 42, 53, 512)
        [70] 1460 x 1611 =  (1, 46, 51, 512)
        [71] 1549 x 717 =  (1, 49, 23, 512)
        [72] 1659 x 1378 =  (1, 52, 43, 512)
        [73] 1511 x 1650 =  (1, 48, 52, 512)
        [74] 1592 x 1299 =  (1, 50, 41, 512)
        [75] 784 x 1308 =  (1, 25, 41, 512)
        [76] 1392 x 1387 =  (1, 44, 44, 512)
        [77] 1451 x 1599 =  (1, 46, 50, 512)
        [78] 1011 x 1059 =  (1, 32, 33, 512)
        [79] 968 x 607 =  (1, 31, 19, 512)
        [80] 1399 x 1448 =  (1, 44, 46, 512)
        [81] 1451 x 1407 =  (1, 46, 44, 512)
        [82] 893 x 1763 =  (1, 28, 55, 512)

    """


def btest_restnet_output_size():
    """
         600 x 600, -4: (1, 19, 19, 2048)
    """
    print(calculate_resnet50_output(600, 600), '(1, 19, 19, 2048)', calculate_resnet50_output(600, 600) == (19, 19))

    print(calculate_resnet50_output(1449, 1459), '(1, 46, 46, 512)', calculate_resnet50_output(1449, 1459) == (46, 46))
    print(calculate_resnet50_output(1435, 1393), '(1, 45, 44, 512)', calculate_resnet50_output(1435, 1393) == (45, 44))
    print(calculate_resnet50_output(767, 1342), '(1, 24, 42, 512)', calculate_resnet50_output(767, 1342) == (24, 42))
    print(calculate_resnet50_output(968, 1288), '(1, 31, 41, 512)', calculate_resnet50_output(968, 1288) == (31, 41))
    print(calculate_resnet50_output(1018, 782), '(1, 32, 25, 512)', calculate_resnet50_output(1018, 782) == (32, 25))

    output = calculate_resnet50_output(1702, 1615)
    print(output, '(1, 54, 51, 512)', output == (54, 51))
    assert output == (54, 51)

    output = calculate_resnet50_output(1450, 1120)
    print(output, '(1, 46, 35, 512)', output == (46, 35))
    assert output == (46, 35)

    output = calculate_resnet50_output(608, 1031)
    print(output, '(1, 19, 33, 512)', output == (19, 33))
    assert output == (19, 33)

    output = calculate_resnet50_output(1762, 1784)
    print(output, '(1, 55, 56, 512)', output == (55, 56))
    assert output == (55, 56)

    output = calculate_resnet50_output(727, 1536)
    print(output, '(1, 23, 48, 512)', output == (23, 48))
    assert output == (23, 48)

    output = calculate_resnet50_output(865, 1409)
    print(output, '(1, 27, 44, 512)', output == (27, 44))
    assert output == (27, 44)


def test_resnet():
    inputs = keras.layers.Input(shape=(None, None, 3))

    model = resnet_models.ResNet50(inputs, include_top=False, freeze_bn=True)

    image = np.random.rand(600, 600, 3)
    image = np.expand_dims(image, 0)

    print(model.predict(image))
    Model(inputs, outputs=model.outputs[-1])
    # print('[{0}] {1} x {2} = '.format(i, height, width), model.predict(image).shape)
    print()
