#
# multiclass_nms paddle model generator
#
import os
import numpy as np
import copy  #deepcopy

#from save_model import saveModel


#print numpy array like vector array
def print_alike(arr, seperator_begin='{', seperator_end='}'):
    shape = arr.shape
    rank = len(shape)

    #print("shape: ", shape, "rank: %d" %(rank))

    #for idx, value in np.ndenumerate(arr):
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
                line += "{:.2f}".format(arr[i])  #str(arr[i])
                line += ", " if i < shape[0] - 1 else ' '
            line += end
            #print(line)
            return line

    print(print_array(arr, seperator_end))


# bboxes shape (N, M, 4)
# scores shape (N, C, M)
def multiclass_nms(name: str, bboxes, scores, attrs: dict):
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
                                    return_index=True)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        out_np, nms_rois_num_np, index_np = exe.run(feed={
            'bboxes': bboxes,
            'scores': scores
        },
                                                    fetch_list=output,
                                                    return_numpy=False)

        out = np.array(out_np)
        index = np.array(index_np)
        nms_rois_num = np.array(nms_rois_num_np)

        # Save inputs in order of ngraph function, to facilite Fuzzy test,
        # which accepts inputs and outputs in this order as well.
        #saveModel(name, exe, feedkeys=['bboxes', 'scores'], fetchlist=output, inputs=[bboxes, scores], outputs=[out, index, nms_rois_num])

    # input
    print('\033[94m' + 'bboxes: {}'.format(bboxes.shape) + '\033[0m')
    print_alike(bboxes, seperator_begin='', seperator_end='')
    print('\033[94m' + 'scores: {}'.format(scores.shape) + '\033[0m')
    print_alike(scores, seperator_begin='', seperator_end='')

    # output
    print('\033[91m' + 'out_np: {}'.format(out.shape) + '\033[0m')
    print_alike(out, seperator_begin='', seperator_end='')
    print('\033[91m' + 'index_np: {}'.format(index.shape) + '\033[0m')
    print_alike(index, seperator_begin='', seperator_end='')
    print('\033[91m' + 'nms_rois_num_np: {}'.format(nms_rois_num.shape) +
          '\033[0m')
    print_alike(nms_rois_num, seperator_begin='', seperator_end='')

    return [index, nms_rois_num, out]  #the same order of pred_ngraph dict


def main():  # multiclass_nms
    random_testcase = [None] * 10
    test_case = [None] * 10

    # case PASS
    test_case[0] = {  # N 1, C 1, M 6
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype(np.float32),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  #PDPD Op type
            'background_label': -1,
            'score_threshold': 0.4,
            'nms_top_k': 200,
            'nms_threshold': 0.5,
            'keep_top_k': 200,
            'normalized': False,
            'nms_eta': 1.0
        },
        'hack_nonzero': [np.array([1., 1., 0., 0., 0., 0.])]
    }

    # case PASS
    test_case[1] = {  # N 1, C 2, M 6
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype(np.float32),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                   [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  #PDPD Op type
            'background_label': 0,
            'score_threshold': 0.0,
            'nms_top_k': 2,
            'nms_threshold': 0.5,
            'keep_top_k': -1,  #keep all
            'normalized': False,
            'nms_eta': 1.0
        },
        'hack_nonzero': [np.array([0., 1., 0., 1.]),
                         np.array([1., 1.])]
    }

    # all are background.. so no output.
    # TODO
    test_case[2] = {  # N 2, C 1, M 6
        'boxes':
        np.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0, 101.0]],
                  [[0.0, 0.0, 1.0, 1.0], [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9], [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1], [0.0, 100.0, 1.0,
                                            101.0]]]).astype(np.float32),
        'scores':
        np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
                  [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  #PDPD Op type
            'background_label': 0,
            'score_threshold': 0.0,
            'nms_top_k': 2,
            'nms_threshold': 0.5,
            'keep_top_k': -1,  #keep all
            'normalized': False,
            'nms_eta': 1.0
        },
        'hack_nonzero': [
            np.array([0., 0., 0., 0.]),  # ALL ARE BACKGROUND. EARLY DROP.
            np.array([1., 1.])
        ]
    }

    test_case[3] = {  # N 2, C 5, M 3 PASS
        'boxes':
        np.array([[[0.5, 0.06, 0.79, 0.76], [0.34, 0.13, 0.52, 0.96],
                   [0.05, 0.35, 0.95, 0.96]],
                  [[0.21, 0.01, 0.85, 0.95], [0.4, 0.09, 0.51, 0.84],
                   [0.37, 0.14, 0.61, 0.95]]]).astype('float32'),
        'scores':
        np.array([[[0.12, 0.12, 0.4], [0.54, 0.14, 0.29], [0.22, 0.22, 0.16],
                   [0.18, 0.13, 0.12], [0.2, 0.45, 0.22]],
                  [[0.12, 0.2, 0.10], [0.91, 0.99, 0.95], [0.99, 0.99, 0.96],
                   [0.95, 0.90, 0.91], [0.93, 0.95, 0.87]]]).astype('float32'),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  #PDPD Op type
            'background_label': 1,
            'score_threshold': 0.0,  # the less, the more bbox kept.
            'nms_top_k': 2,  # max_output_box_per_class
            'nms_threshold': 0.02,  # the bigger, the more bbox kept.
            'keep_top_k': 2,  #-1, keep all
            'normalized': True,
            'nms_eta': 1.0
        },
        'hack_nonzero': [
            np.array([
                0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                1., 1., 1., 1.
            ]),  #elimiate background
            np.array([
                0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0.
            ]),  #image 0
            np.array([
                1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0.
            ])  #image 1
        ]
    }

    # test case 4: data type float64
    # FAILED, as
    # FP64 is not supported,
    # inference-engine/src/mkldnn_plugin/mkldnn_plugin.cpp:399 [ NOT_IMPLEMENTED ] Input image format FP64 is not supported yet...
    test_case[4] = copy.deepcopy(test_case[3])
    test_case[4]['boxes'] = test_case[4]['boxes'].astype('float64')
    test_case[4]['scores'] = test_case[4]['scores'].astype('float64')

    # test case 5: normalize
    # PASS
    test_case[5] = copy.deepcopy(test_case[3])
    test_case[5]['normalized'] = False

    # case 6: no detection in the first image
    test_case[6] = {  # N 2, C 5, M 3
        'boxes':
        np.array([[[0.5, 0.06, 0.79, 0.76], [0.34, 0.13, 0.52, 0.96],
                   [0.05, 0.35, 0.95, 0.96]],
                  [[0.21, 0.01, 0.85, 0.95], [0.4, 0.09, 0.51, 0.84],
                   [0.37, 0.14, 0.61, 0.95]]]).astype('float32'),
        'scores':
        np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                  [[0.12, 0.2, 0.10], [0.91, 0.99, 0.95], [0.99, 0.99, 0.96],
                   [0.95, 0.90, 0.91], [0.93, 0.95, 0.87]]]).astype('float32'),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  #PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,  # the less, the more bbox kept.
            'nms_top_k': 2,  # max_output_box_per_class
            'nms_threshold': 0.02,  # the bigger, the more bbox kept.
            'keep_top_k': 2,  #-1, keep all
            'normalized': True,
            'nms_eta': 1.0
        },
        'hack_nonzero': [
            np.array([0.] * 20),  #image 0
            np.array([
                1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0.
            ])  #image 1
        ]
    }

    # case 7: no detection in the second image
    test_case[7] = {  # N 2, C 5, M 3
        'boxes':
        np.array([[[0.5, 0.06, 0.79, 0.76], [0.34, 0.13, 0.52, 0.96],
                   [0.05, 0.35, 0.95, 0.96]],
                  [[0.21, 0.01, 0.85, 0.95], [0.4, 0.09, 0.51, 0.84],
                   [0.37, 0.14, 0.61, 0.95]]]).astype('float32'),
        'scores':
        np.array([[[0.12, 0.12, 0.4], [0.54, 0.14, 0.29], [0.22, 0.22, 0.16],
                   [0.18, 0.13, 0.12], [0.2, 0.45, 0.22]],
                  [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]).astype('float32'),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  #PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,  # the less, the more bbox kept.
            'nms_top_k': 2,  # max_output_box_per_class
            'nms_threshold': 0.02,  # the bigger, the more bbox kept.
            'keep_top_k': 2,  #-1, keep all
            'normalized': True,
            'nms_eta': 1.0
        },
        'hack_nonzero': [
            np.array([
                1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0.
            ]),  #image 0
            np.array([0.] * 20)  #image 1
        ]
    }

    # case 8: no detection in this batch
    test_case[8] = {  # N 2, C 5, M 3
        'boxes':
        np.array([[[0.5, 0.06, 0.79, 0.76], [0.34, 0.13, 0.52, 0.96],
                   [0.05, 0.35, 0.95, 0.96]],
                  [[0.21, 0.01, 0.85, 0.95], [0.4, 0.09, 0.51, 0.84],
                   [0.37, 0.14, 0.61, 0.95]]]).astype('float32'),
        'scores':
        np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                  [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]).astype('float32'),
        'pdpd_attrs': {
            'nms_type': 'multiclass_nms3',  #PDPD Op type
            'background_label': -1,
            'score_threshold': 0.0,  # the less, the more bbox kept.
            'nms_top_k': 2,  # max_output_box_per_class
            'nms_threshold': 0.02,  # the bigger, the more bbox kept.
            'keep_top_k': 2,  #-1, keep all
            'normalized': True,
            'nms_eta': 1.0
        },
        'hack_nonzero': [
            np.array([0.] * 20),  #image 0
            np.array([0.] * 20)  #image 1
        ]
    }

    # test case 9: no output
    # TODO
    # Here set 2.0 to test the case there is no outputs.
    # In practical use, 0.0 < score_threshold < 1.0
    test_case[9] = copy.deepcopy(test_case[3])
    test_case[9]['pdpd_attrs']['score_threshold'] = 2.0

    # bboxes shape (N, M, 4)
    # scores shape (N, C, M)
    T = 3
    data_bboxes = test_case[T]['boxes']
    data_scores = test_case[T]['scores']
    pdpd_attrs = test_case[T]['pdpd_attrs']
    hack_nonzero = test_case[T]['hack_nonzero']

    # For any change to pdpd_attrs, do -
    # step 1. generate paddle model
    pred_pdpd = multiclass_nms('multiclass_nms_test1', data_bboxes,
                               data_scores, pdpd_attrs)

    #from multiclass_nms3_ngraph import ngraph_multiclass_nms3
    #pred_ngraph = ngraph_multiclass_nms3(data_bboxes, data_scores, pdpd_attrs, hack_nonzero, static_shape=True, static_type=True)

    # step 2. generate onnx model
    # !paddle2onnx --model_dir=../models/yolo_box_test1/ --save_file=../models/yolo_box_test1/yolo_box_test1.onnx --opset_version=10
    #import subprocess
    #subprocess.run(["paddle2onnx", "--model_dir=../models/multiclass_nms_test1/", "--save_file=../models/multiclass_nms_test1/multiclass_nms_test1.onnx", "--opset_version=11", "--enable_onnx_checker=True"])
    #pred_onx = onnx_run(data_bboxes, data_scores)

    # step 3. run from frontend API
    #pred_ie = OV_frontend_run(data_bboxes, data_scores)

    # step 4. compare
    # Try different tolerence
    #validate(pred_pdpd, pred_ngraph)
    #validate(pred_pdpd, pred_ngraph, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    main()
