from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os

def ngraph_multiclass_nms3(input_boxes, input_scores, pdpd_attrs):
    from openvino.inference_engine import IECore
    import ngraph as ng
    from ngraph import opset7

    # nms
    def nms(node_bboxes, node_scores, pdpd_attrs, class_id=None):
        normalized = pdpd_attrs['normalized']
        nms_top_k = pdpd_attrs['nms_top_k']
        nms_type = pdpd_attrs['nms_type']
        if nms_type == 'matrix_nms':
            iou_threshold = 0.5
            print('\033[91m' + "{} not implemented yet! ".format(nms_type) + '\033[0m')
        else:
            iou_threshold = pdpd_attrs['nms_threshold']
        if nms_top_k == -1:
            nms_top_k = 100000

        #convert the paddle attribute to ngraph constant node
        node_score_threshold = ng.constant(pdpd_attrs['score_threshold'], name='score_threshold', dtype=np.float32)
        node_iou_threshold = ng.constant(iou_threshold, name='iou_threshold', dtype=np.float32)
        node_max_output_boxes_per_class = ng.constant(nms_top_k, name='max_output_boxes_per_class', dtype=np.int64)
        
        # the paddle data format is x1,y1,x2,y2
        kwargs = {'center_point_box': 0}

        if normalized:     
            node_nms = ng.non_max_suppression(node_bboxes, node_scores,
                                                max_output_boxes_per_class=node_max_output_boxes_per_class,
                                                iou_threshold=node_iou_threshold,
                                                output_type='i32',
                                                score_threshold=node_score_threshold,
                                                name='non_max_suppression')
        elif not normalized:
            node_value_one = ng.constant([1.0], name='one', dtype=np.float32)
            node_new_bboxes = ng.split(node_bboxes, axis=2, name="split_bboxes", num_splits=4)
            node_new_bboxes = node_new_bboxes.outputs()   
            node_new_xmax = ng.add(node_new_bboxes[2], node_value_one)                                            
            node_new_ymax = ng.add(node_new_bboxes[3], node_value_one)
            node_new_bboxes = ng.concat([node_new_bboxes[0], node_new_bboxes[1], node_new_xmax, node_new_ymax], axis=2)
            node_nms = ng.non_max_suppression(node_new_bboxes, node_scores,
                                                max_output_boxes_per_class=node_max_output_boxes_per_class,
                                                iou_threshold=node_iou_threshold,
                                                output_type='i32',
                                                score_threshold=node_score_threshold,
                                                name='non_max_suppression')
        
        if class_id is not None and class_id != 0:
            print('\033[91m' + "class_id {} not implemented yet! ".format(class_id) + '\033[0m')        

        return node_nms.outputs()

    #keep_topK
    def keep_top_k(select_bbox_indices, select_scores, valid_outputs, scores, bboxes, pdpd_attrs, is_lod_input=False):
        outputs = [None, None, None] # store outputs of this graph

        # create some const value to use
        background = pdpd_attrs['background_label']
        const_values = []
        for value in [0, 1, 2, -1]:
            const_value = ng.constant([value], name='const_'+str(value), dtype=np.int64)                
            const_values.append(const_value)

        class_id = ng.gather(select_bbox_indices, indices=const_values[1], axis=const_values[1], name='gather_class_id')
        squeezed_class_id = ng.squeeze(class_id, axes=const_values[1], name='squeezed_class_id')
        bbox_id = ng.gather(select_bbox_indices, indices=const_values[2], axis=const_values[1], name='gather_bbox_id')

        # Phase 1: remove background class     
        node_background = ng.constant(np.array([background]), dtype=np.int32, name='node_background')
        notequal_background = ng.not_equal(squeezed_class_id, node_background, name='notequal_background')
        # TODO: have to hardcode here to make non_zero happy!!
        notequal_background = ng.constant(np.array([0., 1., 0., 1.]), dtype=np.int32)          
        nonzero = ng.non_zero(notequal_background, output_type="i32", name='nonzero')    

        # non-background's
        class_id = ng.gather(class_id, indices=nonzero, axis=const_values[0], name='non-bg-class_id') # class_id:shape (4,1) -> (1,4,1)
        bbox_id = ng.gather(bbox_id, indices=nonzero, axis=const_values[0], name='nong-bg-bbox_id')

        select_scores = ng.gather(select_scores, indices=nonzero, axis=const_values[0], name='nong-bg-select_scores') # dim increase
        gather_scores = ng.gather(select_scores, indices=const_values[2], axis=const_values[2], name='gather_scores') #axis=0 true scores

        # Squeeze the indices to 1 dim
        squeeze_axes = ng.constant([0 ,2], dtype=np.int64, name='squeeze_axes')            
        gather_scores = ng.squeeze(gather_scores, axes=squeeze_axes, name='squeeze_gather_scores') #(1,2,1) -> (2,)

        # Phase 2: topK 
        keep_top_k = pdpd_attrs['keep_top_k']
        if keep_top_k == -1:
            keep_top_k = 100000    # K must be positive, must be a scaler     
        keep_top_k = ng.constant(np.array([keep_top_k]).reshape(1,1), dtype=np.int64, name='keep_top_k') #dims=[1, 1]

        # get min(topK, num_select)
        shape_select_num = ng.shape_of(gather_scores, name='shape_select_num')

        gather_select_num = ng.gather(shape_select_num, const_values[0], axis=0, name='gather_select_num')

        unsqueeze_select_num = ng.unsqueeze(gather_select_num, axes=[0], name='unsqueeze_select_num')

        concat_topK_select_num = ng.concat([unsqueeze_select_num, keep_top_k], axis=0, name='concat_topK_select_num')

        cast_concat_topK_select_num = ng.convert(concat_topK_select_num, destination_type=np.int32, name='cast_concat_topK_select_num') #6

        keep_top_k = ng.reduce_min(cast_concat_topK_select_num, reduction_axes=[0], keep_dims=False, name='reduce_min/keep_top_k')   

        # select topk scores indices
        keep_top_k = ng.squeeze(keep_top_k, axes=const_values[0]) # K must be positive, must be a scaler        
        node_topk = ng.topk(gather_scores, keep_top_k, axis=0, mode='max', sort='value', index_element_type='i64', name='topK')
        keep_topk_scores, keep_topk_indices = node_topk.outputs()

        # gather topk label, scores, boxes
        gather_topk_scores = ng.gather(gather_scores, indices=keep_topk_indices, axis=const_values[0], name='topk_scores')
        gather_topk_class = ng.gather(class_id, indices=keep_topk_indices, axis=const_values[1], name='topk_label')

        # gather the boxes need to gather the boxes id, then get boxes
        if is_lod_input:
            gather_topk_boxes_id = ng.gather(add_class_indices, keep_topk_indices, axis=const_values[1], name='topk_boxes_id')
        else:
            gather_topk_boxes_id = ng.gather(bbox_id, keep_topk_indices, axis=const_values[1])

        # squeeze the gather_topk_boxes_id to 1 dim
        const_axes_1dim = ng.constant([0,2], dtype=np.int64)
        squeeze_topk_boxes_id = ng.squeeze(gather_topk_boxes_id, axes=const_axes_1dim)

        gather_select_boxes = ng.gather(bboxes, squeeze_topk_boxes_id, axis=const_values[1], name='gather_select_boxes')

        # concat the final result
        # before concat need to cast the class to float
        cast_topk_class = ng.convert(gather_topk_class, destination_type=np.float, name='cast_topk_class')

        unsqueeze_topk_scores = ng.unsqueeze(gather_topk_scores, axes=const_axes_1dim, name='unsqueeze_topk_scores')

        inputs_concat_final_results = [
            cast_topk_class, unsqueeze_topk_scores, gather_select_boxes
        ]

        sort_by_score_results = ng.concat(inputs_concat_final_results, axis=2, name='sort_by_score_results')

        # Phase 3: sort by class_id
        squeeze_cast_topk_class = ng.squeeze(cast_topk_class, axes=const_axes_1dim)

        node_topk = ng.topk(squeeze_cast_topk_class, keep_top_k, axis=0, mode='min', sort='value')
        data, indices = node_topk.outputs()       

        concat_final_results = ng.gather(sort_by_score_results, indices, axis=const_values[1], name='concat_final_results')

        # output node['Out']
        outputs[0] = ng.squeeze(concat_final_results, axes=const_values[0], name='Out')

        nms_type = pdpd_attrs['nms_type']
        if nms_type in ['multiclass_nms2', 'matrix_nms', 'multiclass_nms3']:
            # output node['Index']
            final_indices = ng.gather(bbox_id, indices, axis=const_values[1], name='final_indices')
            final_indices = ng.squeeze(final_indices, axes=const_values[0], name='Index')

            outputs[1] = final_indices

            # output node['NmsRoisNum']
            if nms_type in ['matrix_nms', 'multiclass_nms3']:
                select_bboxes_shape = ng.shape_of(final_indices)
                indices = ng.constant([0], dtype=np.int64)
                rois_num = ng.gather(select_bboxes_shape, indices=indices, axis=const_values[0], name='NmsRoisNum')
                outputs[2] = rois_num
        
        return outputs

    '''
    # Op conversion
    '''
    #inputs
    node_bboxes = ng.parameter(shape=input_boxes.shape, name='boxes', dtype=np.float32)
    node_scores = ng.parameter(shape=input_scores.shape, name='scores', dtype=np.float32)

    if len(input_scores.shape) == 2:
        # inputs: scores & bboxes is lod tensor
        print('\033[91m' + "{} not implemented yet! ".format('lod tensor inputs') + '\033[0m')
    else:
        select_bbox_indices, select_scores, valid_outputs = nms(node_bboxes, node_scores, pdpd_attrs)
        #graph = [select_bbox_indices, select_scores, valid_outputs]
        graph = keep_top_k(select_bbox_indices, select_scores, valid_outputs, node_scores, node_bboxes, pdpd_attrs)    

    assert isinstance(graph, list)
    print('\033[94m' + "executable_network graph {} {}".format(type(graph), graph[0]) + '\033[0m')
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


def onnx_multiclass_nms3(input_boxes, input_scores, pdpd_attrs):
    import onnx
    from onnx import helper
    from onnx import AttributeProto, TensorProto, GraphProto
    import onnx
    from onnx import helper, shape_inference
    from onnx import TensorProto

    # debug         
    print('\033[94m' + "@@#############################debugging {}".format(int(TensorProto.INT32))+ '\033[0m')       
    print('\033[94m' + "@@#############################debugging {}".format(np.array([False]).astype(np.bool).dtype)+ '\033[0m')          

    # The protobuf definition can be found here:
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

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

    #onnx_result = onnx_multiclass_nms3(sample_boxes, sample_scores, pdpd_attrs)
