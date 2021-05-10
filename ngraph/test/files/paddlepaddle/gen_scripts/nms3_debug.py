from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os

def ngraph_multiclass_nms3(input_boxes, input_scores, pdpd_attrs):
    from openvino.inference_engine import IECore
    import ngraph as ng
    from ngraph import opset6

    '''
    # function graph
    '''
    # inputs
    node_bboxes = ng.parameter(shape=input_boxes.shape, name='boxes', dtype=np.float32)
    node_scores = ng.parameter(shape=input_scores.shape, name='scores', dtype=np.float32)
    # nms
    node_score_threshold = ng.constant(pdpd_attrs['score_threshold'], name='score_threshold', dtype=np.float32)
    node_iou_threshold = ng.constant(iou_threshold, name='iou_threshold', dtype=np.float32)
    node_max_output_boxes_per_class = ng.constant(pdpd_attrs['nms_top_k'], name='max_output_boxes_per_class', dtype=np.int64)

    node_nms = ng.non_max_suppression(node_bboxes, node_scores,
                                        max_output_boxes_per_class=node_max_output_boxes_per_class,
                                        iou_threshold=node_iou_threshold,
                                        output_type='i32',
                                        score_threshold=node_score_threshold,
                                        name='non_max_suppression')

    select_bbox_indices, select_scores, valid_outputs = node_nms.outputs()                                        
    # create some const value to use
    const_values = [] 
    for value in [0, 1, 2, -1]:
        const_value = ng.constant([value], name='const_'+str(value), dtype=np.int64)                
        const_values.append(const_value)

    # remove background class
    background = pdpd_attrs['background_label']

    class_id = ng.gather(select_bbox_indices, indices=const_values[1], axis=const_values[1], name='gather_class_id')
    squeezed_class_id = ng.squeeze(class_id, axes=const_values[1], name='squeezed_class_id')

    node_background = ng.constant(np.array([background]), dtype=np.int32, name='node_background')
    notequal_background = ng.not_equal(squeezed_class_id, node_background, name='notequal_background')
    nonzero_background = ng.non_zero(notequal_background, output_type="i32", name='nonzero')                              

    ### problem #1.
    # notequal_background is of type float32. But according to OV opset doc, it should be bool.
    #### problem #2.
    # shape infer failed, with error log:
    '''
        executable_network = ie.load_network(ie_network, 'CPU')
        File "ie_api.pyx", line 325, in openvino.inference_engine.ie_api.IECore.load_network
        File "ie_api.pyx", line 337, in openvino.inference_engine.ie_api.IECore.load_network
        RuntimeError: 
        /home/cecilia/explore/openvino/inference-engine/src/legacy_api/src/convert_function_to_cnn_network.cpp:1904 
        Unsupported dynamic ops: 
        v3::NonZero nonzero (notequal_background[0]:u8{4}) -> (i32{1,[0, 4]})
        v0::Result Result_15 (nonzero[0]:i32{1,[0, 4]}) -> (i32{1,[0, 4]})
    '''
    graph = [nonzero_background, notequal_background, squeezed_class_id]

    assert isinstance(graph, list)
    result_node0 = ng.result(graph[0], 'debug0')
    result_node1 = ng.result(graph[1], 'debug1')
    result_node2 = ng.result(graph[2], 'debug2')     


    '''
    # runtime
    '''
    function = ng.Function([result_node0, result_node1, result_node2], [node_bboxes, node_scores], "nms")
    ie_network = ng.function_to_cnn(function)

    ie = IECore()
    executable_network = ie.load_network(ie_network, 'CPU')
    output = executable_network.infer(inputs={'boxes': input_boxes,
                                              'scores': input_scores})


    print('\033[92m' + "executable_network output {} ".format(output) + '\033[0m')
    
    for key, value in output.items():
        print("# Ngraph result:\n", key, value.shape, value.dtype)

    return output #dict


if __name__ == "__main__":
    # https://github.com/onnx/onnx/blob/a64c2ed232392e2586dc8320710841eee62874df/onnx/backend/test/case/node/nonmaxsuppression.py#L180
    sample_boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]]).astype(np.float32)
    sample_scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                            [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    score_threshold = 0.0
    background = 0
    iou_threshold = 0.5
    max_output_boxes_per_class = 2

    pdpd_attrs = {
        'nms_type': 'multiclass_nms3', #PDPD Op type
        'background_label': background,
        'score_threshold': score_threshold,
        'nms_top_k': max_output_boxes_per_class,
        'nms_threshold': iou_threshold,
        'keep_top_k': -1,  #keep all
        'normalized': False,
        'nms_eta': 1.0
    }    

    ng_result = ngraph_multiclass_nms3(sample_boxes, sample_scores, pdpd_attrs)