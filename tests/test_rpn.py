from frcnn.rpn import create_rpn_regression_target


def test_regression_target():
    gt_box = [4, 8, 14, 18]
    a_box = [3, 2, 14, 18]
    create_rpn_regression_target(gt_box, a_box)
