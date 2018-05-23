import numpy as np

from frcnn.anchor import to_relative_coord_np, to_absolute_coord, apply_regression_to_rois


def test_relative_and_absolute_anchor():
    gtas = np.array([[0, 0, 1, 1],
                     [0, 0, 1, 1],
                     [0, 0, 2, 5],
                     [5, 0, 7, 7],
                     [0, 10, 8, 100],
                     [10, 11, 15, 20]], dtype=np.float64)

    ancs = np.array([[0, 0, 1, 1],
                     [2, 1, 3, 3],
                     [4, 2, 10, 8],
                     [3, 3, 10, 15],
                     [2, 8, 5, 120],
                     [1, 1, 2, 2]], dtype=np.float64)
    txtytwth = to_relative_coord_np(gtas, ancs)

    # Test to_absolute_coord
    gtas_pred = list()
    for anc, regr in zip(ancs, txtytwth):
        xywh = to_absolute_coord(anc, regr)
        xywh = list(xywh)
        xywh[0] -= xywh[2] / 2.
        xywh[1] -= xywh[3] / 2.
        xywh[2] += xywh[0]
        xywh[3] += xywh[1]
        gtas_pred.append(xywh)

    gtas_pred = np.array(gtas_pred)
    assert (gtas_pred == gtas).all()

    # Test apply_regression_to_roi
    mxmywh = ancs.copy()
    mxmywh[:, 2] = mxmywh[:, 2] - mxmywh[:, 0]  # to width
    mxmywh[:, 3] = mxmywh[:, 3] - mxmywh[:, 1]  # to height
    cxcywh = apply_regression_to_rois(txtytwth, mxmywh).astype(np.float64)
    anchors = cxcywh
    anchors[:, 0] -= anchors[:, 2] / 2.
    anchors[:, 1] -= anchors[:, 3] / 2.
    anchors[:, 2] += anchors[:, 0]
    anchors[:, 3] += anchors[:, 1]
    assert (anchors == gtas).all()
