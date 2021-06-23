#
# multiclass_nms paddle model generator
#
import os
import numpy as np
import copy  # deepcopy
import sys

from nms import NMS


def main():  # multiclass_nms
    test_case = [None] * 20

    # case  multiclass_nms_by_class_id
    test_case[0] = {  # N 1, C 2, M 6
        'name':
        'multiclass_nms_by_class_id',
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype(np.float32),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                   [0.95, 0.75, 0.6, 0.80, 0.5, 0.3]]]).astype(np.float32),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,
            'nms_top_k': 3,
            'nms_threshold': 0.5,
            'keep_top_k': -1,
            'normalized': True,
            'nms_eta': 1.0,
            'return_index': True
        }
    }

    # case multiclass_nms_two_batches_two_classes_by_class_id
    test_case[1] = {  # N 2, C 2, M 3
        'name':
        'multiclass_nms_two_batches_two_classes_by_class_id',
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0, 101.0]],
                  [[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype('float32'),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                   [0.95, 0.75, 0.6, 0.80, 0.5, 0.3]],
                  [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                   [0.95, 0.75, 0.6, 0.80, 0.5, 0.3]]]).astype('float32'),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,  # the less, the more bbox kept.
            'nms_top_k': 3,  # max_output_box_per_class
            'nms_threshold': 0.5,  # the bigger, the more bbox kept.
            'keep_top_k': -1,  # -1, keep all
            'normalized': True,
            'nms_eta': 1.0,
            'return_index': True
        }
    }

    # case  multiclass_nms_identical_boxes
    test_case[2] = {  # N 1, C 1, M 10
        'name':
        'multiclass_nms_identical_boxes',
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0,
                                          1.0]]]).astype(np.float32),
        'scores':
        np.array([[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                    0.9]]]).astype(np.float32),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,
            'nms_top_k': 3,
            'nms_threshold': 0.5,
            'keep_top_k': -1,
            'normalized': True,
            'nms_eta': 1.0,
            'return_index': True
        }
    }

    # case  multiclass_nms_flipped_coordinates
    test_case[3] = {  # N 1, C 1, M 6
        'name':
        'multiclass_nms_flipped_coordinates',
        'boxes':
        np.array([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, 0.9, 1.0, -0.1], [0.0, 10.0, 1.0, 11.0],
                   [1.0, 10.1, 0.0, 11.1], [1.0, 101.0, 0.0,
                                            100.0]]]).astype(np.float32),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,
            'nms_top_k': 3,
            'nms_threshold': 0.5,
            'keep_top_k': -1,
            'normalized': True,
            'nms_eta': 1.0,
            'return_index': True
        }
    }

    # case  multiclass_nms_by_nms_top_k
    test_case[4] = {  # N 1, C 1, M 6
        'name':
        'multiclass_nms_by_nms_top_k',
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype(np.float32),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,
            'nms_top_k': 2,
            'nms_threshold': 0.5,
            'keep_top_k': -1,
            'normalized': True,
            'nms_eta': 1.0,
            'return_index': True
        }
    }

    # case  multiclass_nms_single_box
    test_case[5] = {  # N 1, C 1, M 1
        'name': 'multiclass_nms_single_box',
        'boxes': np.array([[[0.0, 0.0, 1.0, 1.0]]]).astype(np.float32),
        'scores': np.array([[[0.9]]]).astype(np.float32),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,
            'nms_top_k': 3,
            'nms_threshold': 0.5,
            'keep_top_k': -1,
            'normalized': True,
            'nms_eta': 1.0,
            'return_index': True
        }
    }

    # case  multiclass_nms_by_IOU
    test_case[6] = {  # N 1, C 1, M 6
        'name':
        'multiclass_nms_by_IOU',
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype(np.float32),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,
            'nms_top_k': 3,
            'nms_threshold': 0.5,
            'keep_top_k': -1,
            'normalized': True,
            'nms_eta': 1.0,
            'return_index': True
        }
    }

    # case  multiclass_nms_by_IOU_and_scores
    test_case[7] = {  # N 1, C 1, M 6
        'name':
        'multiclass_nms_by_IOU_and_scores',
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype(np.float32),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': -1,
            'score_threshold': 0.93,
            'nms_top_k': 3,
            'nms_threshold': 0.5,
            'keep_top_k': -1,
            'normalized': True,
            'nms_eta': 1.0,
            'return_index': True
        }
    }

    # case multiclass_nms_by_background
    test_case[8] = {  # N 2, C 2, M 3
        'name':
        'multiclass_nms_by_background',
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0, 101.0]],
                  [[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype('float32'),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                   [0.95, 0.75, 0.6, 0.80, 0.5, 0.3]],
                  [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                   [0.95, 0.75, 0.6, 0.80, 0.5, 0.3]]]).astype('float32'),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': 0,
            'score_threshold': 0.0,  # the less, the more bbox kept.
            'nms_top_k': 3,  # max_output_box_per_class
            'nms_threshold': 0.5,  # the bigger, the more bbox kept.
            'keep_top_k': -1,  # -1, keep all
            'normalized': True,
            'nms_eta': 1.0,
            'return_index': True
        }
    }

    # case multiclass_nms_by_keep_top_k
    test_case[9] = {  # N 2, C 2, M 3
        'name':
        'multiclass_nms_by_keep_top_k',
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0, 101.0]],
                  [[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype('float32'),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                   [0.95, 0.75, 0.6, 0.80, 0.5, 0.3]],
                  [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                   [0.95, 0.75, 0.6, 0.80, 0.5, 0.3]]]).astype('float32'),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,  # the less, the more bbox kept.
            'nms_top_k': 3,  # max_output_box_per_class
            'nms_threshold': 0.5,  # the bigger, the more bbox kept.
            'keep_top_k': 3,  # -1, keep all
            'normalized': True,
            'nms_eta': 1.0,
            'return_index': True
        }
    }

    # case multiclass_nms_by_nms_eta
    test_case[10] = {  # N 2, C 2, M 3
        'name':
        'multiclass_nms_by_nms_eta',
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0, 101.0]],
                  [[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype('float32'),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                   [0.95, 0.75, 0.6, 0.80, 0.5, 0.3]],
                  [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                   [0.95, 0.75, 0.6, 0.80, 0.5, 0.3]]]).astype('float32'),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  # PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,  # the less, the more bbox kept.
            'nms_top_k': -1,  # max_output_box_per_class
            'nms_threshold': 1.0,  # the bigger, the more bbox kept.
            'keep_top_k': -1,  # -1, keep all
            'normalized': True,
            'nms_eta': 0.1,
            'return_index': True
        }
    }

    # case multiclass_nms_not_normalized
    test_case[11] = copy.deepcopy(test_case[1])
    test_case[11]['name'] = 'multiclass_nms_not_normalized'
    test_case[11]['pdpd_attrs']['normalized'] = False

    # case multiclass_nms_not_return_indexed
    test_case[12] = copy.deepcopy(test_case[1])
    test_case[12]['name'] = 'multiclass_nms_not_return_indexed'
    test_case[12]['pdpd_attrs']['return_index'] = False

    # bboxes shape (N, M, 4)
    # scores shape (N, C, M)
    for i, t in enumerate(test_case):
        if t is not None:
            print('\033[95m' +
                  '\n\Generating multiclass_nms test case: {} {} ......'.format(i, t['name']) +
                  '\033[0m')

            data_bboxes = t['boxes']
            data_scores = t['scores']
            pdpd_attrs = t['pdpd_attrs']

            pred_pdpd = NMS(t['name'], data_bboxes, data_scores,
                            pdpd_attrs)


if __name__ == "__main__":
    main()
