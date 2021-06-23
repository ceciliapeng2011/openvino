#
# multiclass_nms paddle model generator
#
import os
import numpy as np
import copy  # deepcopy
import sys

from save_model import saveModel

# print numpy array like vector array
# this is to faciliate some unit test, e.g. ngraph op unit test.


def print_alike(arr, seperator_begin='{', seperator_end='}'):
    shape = arr.shape
    rank = len(shape)

    #print("shape: ", shape, "rank: %d" %(rank))

    # for idx, value in np.ndenumerate(arr):
    #    print(idx, value)

    def print_array(arr, end=' '):
        shape = arr.shape
        rank = len(arr.shape)
        if rank > 1:
            line = seperator_begin
            for i in range(arr.shape[0]):
                line += print_array(
                    arr[i, :],
                    end=seperator_end +
                    ",\n" if i < arr.shape[0] - 1 else seperator_end)
            line += end
            return line
        else:
            line = seperator_begin
            for i in range(arr.shape[0]):
                line += "{:.2f}".format(arr[i])  # str(arr[i])
                line += ", " if i < shape[0] - 1 else ' '
            line += end
            # print(line)
            return line

    print(print_array(arr, seperator_end))


# bboxes shape (N, M, 4)
# scores shape (N, C, M)
def multiclass_nms(name: str, bboxes, scores, attrs: dict, quite=True):
    import paddle as pdpd
    from ppdet.modeling import ops
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(),
                                   pdpd.static.Program()):
        node_boxes = pdpd.static.data(name='bboxes',
                                      shape=bboxes.shape,
                                      dtype=bboxes.dtype,
                                      lod_level=1)
        node_scores = pdpd.static.data(name='scores',
                                       shape=scores.shape,
                                       dtype=scores.dtype,
                                       lod_level=1)

        output = ops.multiclass_nms(bboxes=node_boxes,
                                    scores=node_scores,
                                    background_label=attrs['background_label'],
                                    score_threshold=attrs['score_threshold'],
                                    nms_top_k=attrs['nms_top_k'],
                                    nms_threshold=attrs['nms_threshold'],
                                    keep_top_k=attrs['keep_top_k'],
                                    normalized=attrs['normalized'],
                                    nms_eta=attrs['nms_eta'],
                                    return_index=attrs['return_index'])

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        fetch_vars = [x for x in output if x is not None]
        output_lod = exe.run(feed={'bboxes': bboxes, 'scores': scores},
                             fetch_list=fetch_vars,
                             return_numpy=False)

        # There is a bug in paddledet that dtype of model var mismatch its output LodTensor.
        # Specifically, it is 'Index' is 'int64', while its LodTensor of 'int32'.
        # This will lead to a failure in ngraph frontend op fuzzy test.
        # So here is an workaround to align the dtypes.
        out = np.array(output_lod.pop(0))
        nms_rois_num = np.array(
            output_lod.pop(0)) if output[1] is not None else None
        index = np.array(output_lod.pop(0)).astype(pdpd.fluid.data_feeder.convert_dtype(
            output[2].dtype)) if output[2] is not None else None

        # Save inputs in order of ngraph function, to facilite Fuzzy test,
        # which accepts inputs and outputs in this order as well.
        output_np = [out, nms_rois_num, index]
        saveModel(name,
                  exe,
                  feedkeys=['bboxes', 'scores'],
                  fetchlist=fetch_vars,
                  inputs=[bboxes, scores],
                  outputs=[x for x in output_np if x is not None],
                  target_dir=sys.argv[1])

    if quite is False:
        # input
        print('\033[94m' + 'bboxes: {}'.format(bboxes.shape) + '\033[0m')
        print_alike(bboxes, seperator_begin='', seperator_end='')
        print('\033[94m' + 'scores: {}'.format(scores.shape) + '\033[0m')
        print_alike(scores, seperator_begin='', seperator_end='')

        # output
        print('\033[91m' + 'out_np: {}'.format(out.shape) + '\033[0m')
        print_alike(out, seperator_begin='', seperator_end='')
        print('\033[91m' + 'nms_rois_num_np: {}'.format(nms_rois_num.shape) +
              '\033[0m')
        print_alike(nms_rois_num, seperator_begin='', seperator_end='')
        if index is not None:
            print('\033[91m' + 'index_np: {}'.format(index.shape) + '\033[0m')
            print_alike(index, seperator_begin='', seperator_end='')

    return


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

            pred_pdpd = multiclass_nms(t['name'], data_bboxes, data_scores,
                                       pdpd_attrs)


if __name__ == "__main__":
    main()
