from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os

def ngraph_multiclass_nms3(input_boxes, input_scores, score_threshold, iou_threshold, max_output_boxes_per_class, output_type):
    from openvino.inference_engine import IECore
    import ngraph as ng
    from ngraph import opset7

    node_input = ng.parameter(shape=input_boxes.shape, name='boxes', dtype=np.float32)
    node_score = ng.parameter(shape=input_scores.shape, name='scores', dtype=np.float32)
    node_max_output_boxes_per_class = ng.constant(max_output_boxes_per_class, name='max_output_boxes_per_class', dtype=np.int64)
    node_iou_threshold = ng.constant(iou_threshold, name='iou_threshold', dtype=np.float32)
    node_score_threshold = ng.constant(score_threshold, name='score_threshold', dtype=np.float32)

    graph = ng.opset5.non_max_suppression(node_input, node_score,
                                          max_output_boxes_per_class=node_max_output_boxes_per_class,
                                          iou_threshold=node_iou_threshold,
                                          output_type='i32',
                                          score_threshold=node_score_threshold)
    result_node0 = ng.result(graph.output(0), 'selected_indices')
    result_node1 = ng.result(graph.output(1), 'selected_scores')
    result_node2 = ng.result(graph.output(2), 'valid_outputs')
    function = ng.Function([result_node0, result_node1, result_node2], [node_input, node_score], "nms")
    ie_network = ng.function_to_cnn(function)

    ie = IECore()
    executable_network = ie.load_network(ie_network, 'CPU')
    output = executable_network.infer(inputs={'boxes': input_boxes,
                                              'scores': input_scores})

    selected_indices = None
    selected_scores = None
    valid_outputs = None
    for key, value in output.items():
        print("# Ngraph result:\n", key, value.shape, value.dtype)
        
        if len(value.shape) == 1:
            valid_outputs = value
        elif isinstance(value[0][0], np.float32):
            selected_scores = value
        else:
            selected_indices = value
    
    pred_ngraph_indice = []
    for i in range(valid_outputs[0]):
        print("index: {}, scores: {}".format(selected_indices[i], selected_scores[i]))
        pred_ngraph_indice.append(selected_indices[i])

    return pred_ngraph_indice


def onnx_multiclass_nms3(input_boxes, input_scores, score_threshold, iou_threshold, max_output_boxes_per_class, output_type):
    import onnx
    from onnx import helper
    from onnx import AttributeProto, TensorProto, GraphProto
    import onnx
    from onnx import helper, shape_inference
    from onnx import TensorProto

    # Create one input (ValueInfoProto)
    _bboxes = helper.make_tensor_value_info('bboxes', TensorProto.FLOAT, input_boxes.shape)
    print(type(_bboxes))
    _scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, input_scores.shape)
    # Create one output (ValueInfoProto)
    _selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [6,3])  

    def make_onnx_const_node(dtype, value):
        if isinstance(value, list):
            dims = (len(value), )
        elif value is None:
            dims = ()
            value = []
        else:
            dims = ()
            value = [value]
        return helper.make_tensor(
                name='Constant_0', data_type=dtype, dims=dims, vals=value)
    node_max_output_boxes_per_class = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['max_output_boxes_per_class'],
                value=make_onnx_const_node(TensorProto.INT64, [np.int64(max_output_boxes_per_class)])
            )
    print(type(max_output_boxes_per_class))
    node_iou_threshold = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['iou_threshold'],
                value=make_onnx_const_node(TensorProto.FLOAT, [float(iou_threshold)])
            )
    node_score_threshold = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['score_threshold'],
                value=make_onnx_const_node(TensorProto.FLOAT, [float(score_threshold)])
            )                        
    
    node_select_bbox_indices = helper.make_node(
        'NonMaxSuppression',        
        inputs=[
            'bboxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'
        ],
        outputs=['selected_indices']
        )    

    graph = helper.make_graph(
        [node_max_output_boxes_per_class, node_iou_threshold, node_score_threshold, node_select_bbox_indices],
        'multiclass_nms',
        [_bboxes, _scores],  #inputs
        [_selected_indices] #outputs
    )

    original_model = helper.make_model(graph, producer_name='onnx-examples')
    #print('The model is:\n{}'.format(original_model))
    # Save the ONNX model
    model_path = os.path.join('onnx', 'original_model.onnx')
    onnx.save(original_model, model_path)    

    # Check the model and print Y's shape information
    onnx.checker.check_model(original_model)
    print('Before shape inference, the shape info of Y is:\n{}'.format(original_model.graph.value_info))

    # Apply shape inference on the model
    inferred_model = shape_inference.infer_shapes(original_model)

    # Check the model and print Y's shape information
    onnx.checker.check_model(inferred_model)
    print('After shape inference, the shape info of Y is:\n{}'.format(inferred_model.graph.value_info))

    # Save the ONNX model
    new_model_path = os.path.join('onnx', 'inferred_model.onnx')
    onnx.save(inferred_model, new_model_path)    

    import onnxruntime as rt

    sess = rt.InferenceSession(new_model_path)
    for input in sess.get_inputs():
        print(input.name)
    for output in sess.get_outputs():
        print(output.name)        
    pred_onx = sess.run(None, {"bboxes": input_boxes, "scores": input_scores})
    print(type(pred_onx), len(pred_onx))
    print(pred_onx)
    return pred_onx    

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
    background = -1
    iou_threshold = 0.5
    max_output_boxes_per_class = 2

    ng_result = ngraph_multiclass_nms3(sample_boxes, sample_scores, score_threshold=score_threshold, iou_threshold=iou_threshold,
                           max_output_boxes_per_class=max_output_boxes_per_class, output_type='i64')

    onnx_result = onnx_multiclass_nms3(sample_boxes, sample_scores, score_threshold=score_threshold, iou_threshold=iou_threshold,
                           max_output_boxes_per_class=max_output_boxes_per_class, output_type='i64')
