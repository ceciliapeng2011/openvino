from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os

def ngraph_multiclass_nms3(input_boxes, input_scores, pdpd_attrs, hack_nonzero=None):
    from openvino.inference_engine import IECore
    import ngraph as ng
    from ngraph import opset7

    # nms
    # node_bboxes: shape (N, M, 4)
    # node_scores: shape (N, C, M)
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
                                                sort_result_descending=True, #onnx: sort_result_descending=false
                                                name='non_max_suppression')
        
        if class_id is not None and class_id != 0:
            print('\033[91m' + "class_id {} not implemented yet! ".format(class_id) + '\033[0m')        

        return node_nms.outputs()

    # keep_topK
    # select_bbox_indices: shape [num_selected_boxes, 3], num_selected_boxes = min(M, max_ouput_boxes_per_class) * N * C
    # max_ouput_boxes_per_class equals nms_top_k in pdpd_attrs.
    # select_scores: shape [num_selected_boxes, 3]
    def keep_top_k(select_bbox_indices, select_scores, valid_outputs, input_scores, input_bboxes, pdpd_attrs, is_lod_input=False, hack_nonzero=None):
        selected_out, selected_indices, selected_num = [], [], [] # store outputs of this graph

        # preapre: create some const value to use        
        const_values = []
        for value in [0, 1, 2, -1]:
            const_value = ng.constant([value], name='const_'+str(value), dtype=np.int64)                
            const_values.append(const_value)

        # Phase 0: 
        # yield invalid outputs
        '''
        slice_begin = ng.constant(np.array([0]), dtype=np.int32)
        slice_end  =  ng.constant(np.array([10]), dtype=np.int32)
        slice_strides = ng.constant(np.array([1]),dtype=np.int32)
        select_bbox_indices = ng.strided_slice(select_bbox_indices, begin=slice_begin, end=valid_outputs, strides=slice_strides, begin_mask=[], end_mask=[], name='strideslice_select_bbox_indices') #Unsupported dynamic ops
        select_scores = ng.strided_slice(select_scores, begin=slice_begin, end=valid_outputs, strides=slice_strides, begin_mask=[], end_mask=[], name='strideslice_select_scores') #Unsupported dynamic ops
        return [select_bbox_indices, select_scores, valid_outputs]
        '''    

        # Phase 1: 
        # eliminate background class
        background = pdpd_attrs['background_label']
        if background >= 0:
            select_class_id = ng.gather(select_scores, indices=const_values[1], axis=const_values[1], name='gather_select_class_id') # shape (num_selected_boxes, 1)
            select_class_id = ng.squeeze(select_class_id, axes=const_values[1], name='squeezed_select_class_id') # shape (num_selected_boxes,)  
                    
            node_background = ng.constant(np.array([background]), dtype=np.float, name='node_background')
            notequal_background = ng.not_equal(select_class_id, node_background, name='notequal_background')

            notequal_background = ng.constant(np.array([0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1.]), dtype=np.float) #HARDCODE   
            nonzero = ng.non_zero(notequal_background, output_type="i32", name='nonzero_background')  # shape (1, S1) #Unsupported dynamic ops            

            # non-background's
            select_scores = ng.gather(select_scores, indices=nonzero, axis=const_values[0], name='nonbg_select_scores') # shape (1, S1, 3)
            select_scores = ng.squeeze(select_scores, axes=const_values[0], name='select_scores')
             
            select_bbox_indices = ng.gather(select_bbox_indices, indices=nonzero, axis=const_values[0], name='nonbg_select_bbox_indices') # shape (1, S1, 3)
            select_bbox_indices = ng.squeeze(select_bbox_indices, axes=const_values[0], name='select_bbox_indices')

        # Phase 2: 
        # topK 
        # keep_top_k per batch, according to 
        # https://github.com/PaddlePaddle/Paddle/blob/c6713bc00e881b281a6ad4cf20daf1088334dbea/python/paddle/fluid/tests/unittests/test_multiclass_nms_op.py#L119
        keep_top_k = pdpd_attrs['keep_top_k'] 
        if keep_top_k < 0:
            keep_top_k = 100000

        N = input_scores.shape[0]
        for i in range(N):
            print("~~~~ loop start for image {}/{} ~~~~~".format(i, N))

            #
            # gather the bboxes and scores for this image.
            #
            image_id = ng.gather(select_scores, indices=const_values[0], axis=const_values[1], name='gather_image_id') # shape (S1, 1)
            squeezed_image_id = ng.squeeze(image_id, axes=const_values[1], name='squeezed_image_id') # shape (S1,)

            const_imageid = ng.constant(np.array([i]), dtype=np.float, name='const_image_id'+str(i))
            equal_imageid = ng.equal(squeezed_image_id, const_imageid, name='equal_imageid')
            #return [select_scores, image_id, equal_imageid]
            
            equal_imageid = ng.constant(np.array([0., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), dtype=np.int32) #HARDCODE
            nonzero_imageid = ng.non_zero(equal_imageid, output_type="i32", name='nonzero_imageid')  # shape (1, S2) #Unsupported dynamic ops

            cur_select_bbox_indices = ng.gather(select_bbox_indices, indices=nonzero_imageid, axis=const_values[0]) # shape (1, S2, 3)
            cur_select_bbox_indices = ng.squeeze(cur_select_bbox_indices, axes=const_values[0], name='select_bbox_indices')

            cur_select_scores = ng.gather(select_scores, indices=nonzero_imageid, axis=const_values[0]) # shape (1, S2, 3)
            cur_select_scores = ng.squeeze(cur_select_scores, axes=const_values[0], name='cur_select_scores')

            # 
            # select_topk
            #
            shape_select_scores = ng.shape_of(cur_select_scores, name='shape_select_scores') # shape (2,) with value [S2, 3]
            gather_num_select = ng.gather(shape_select_scores, const_values[0], axis=0, name='gather_num_select') # shape (1,)
            
            # min(keep_top_k, num_select)
            node_keep_top_k = ng.constant(np.array([keep_top_k]))
            concat_topK_select_num = ng.concat([gather_num_select, node_keep_top_k], axis=0, name='concat_topK_select_num')                
            num_select = ng.reduce_min(concat_topK_select_num, reduction_axes=[0], keep_dims=False, name='reduce_min_keep_top_k') # shape () # note: num_select is a scaler here now.
            
            # 
            # select topk scores indices
            #
            # Since sort_result_descending=True, scores are ordered, we can just use Slice to get topK either. 
            # Unluckily Runtime StrideSlice has no dynamic shape support so far.
            gather_scores = ng.gather(cur_select_scores, indices=const_values[2], axis=const_values[1], name='gather_select_scores') # shape (S2, 1)
            gather_scores = ng.squeeze(gather_scores, axes=const_values[1], name='gather_scores') # shape (S2,)  

            node_topk = ng.topk(gather_scores, num_select, axis=0, mode='max', sort='value', index_element_type='i64', name='topK') # K must be positive, must be a scaler
            topk_scores, topk_indices = node_topk.outputs()  # shape (S3,)
            topk_scores = ng.unsqueeze(topk_scores, axes=const_values[1], name='topk_scores')            

            # 
            # gather corresponding  class, boxes_id, then bboxes
            #
            topk_bboxes_indices = ng.gather(cur_select_bbox_indices, indices=topk_indices, axis=const_values[0], name='topk_bboxes_indices') # shape (S3, 3)

            topk_class_id = ng.gather(topk_bboxes_indices, indices=const_values[1], axis=const_values[1], name='gather_class_id') # shape (S3, 1)
            topk_class_id = ng.convert(topk_class_id, destination_type=np.float, name='topk_class_id')

            topk_box_id = ng.gather(topk_bboxes_indices, indices=const_values[2], axis=const_values[1], name='gather_box_id') # shape (S3, 1)
            topk_box_id = ng.convert(topk_box_id, destination_type=np.float, name='topk_box_id')            

            const_02 = ng.constant([0,2], dtype=np.int64)
            gather_bbox_indices = ng.gather(topk_bboxes_indices, indices=const_02, axis=1, name='gather_bbox_indices') # shape (S3, 2) containing triplets (batch_index, box_index)            
            gather_bboxes = ng.gather_nd(input_bboxes, indices=gather_bbox_indices, batch_dims=0, name='gather_bboxes') # shape (S3, 4) containg triplets of bbox coords.
            
            # concat the final result for current image
            sort_by_score_results = ng.concat([topk_class_id, topk_scores, gather_bboxes], axis=1, name='sort_by_score_results'+str(i))            

            # Phase 3: 
            # sort by class_id
            node_topk = ng.topk(topk_class_id, num_select, axis=0, mode='min', sort='value')
            data, indices = node_topk.outputs() # shape (S3,1)
            indices = ng.squeeze(indices, axes=const_values[1], name='topk_class_indices')  # shape (S3,)    

            # output['Out']
            sort_by_class_results = ng.gather(sort_by_score_results, indices, axis=const_values[0], name='sort_by_class_results'+str(i)) # shape (S3, 6)           
            selected_out.append(sort_by_class_results) 

            # output['Index']
            final_indices = ng.gather(topk_box_id, indices, axis=const_values[0], name='final_indices'+str(i)) # shape (S3, 1)
            selected_indices.append(final_indices)

            # output['NmsRoisNum']
            select_bboxes_shape = ng.shape_of(final_indices)
            indices = ng.constant([0], dtype=np.int64)
            rois_num = ng.gather(select_bboxes_shape, indices=indices, axis=const_values[0], name='NmsRoisNum'+str(i))
            selected_num.append(rois_num)

            #return [sort_by_class_results, final_indices, rois_num]

            print("~~~~ loop end for image {}/{} ~~~~~".format(i, N))

        # concat each output of image
        selected_out = ng.concat(selected_out, axis=0, name='concat_selected_out')  # output['Out']
        selected_indices = ng.concat(selected_indices, axis=0, name='concat_selected_indices')
        selected_num = ng.concat(selected_num, axis=0, name='concat_selected_num')

        return [selected_out, selected_indices, selected_num]

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
        graph = keep_top_k(select_bbox_indices, select_scores, valid_outputs, node_scores, node_bboxes, pdpd_attrs, hack_nonzero=hack_nonzero)    

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

    orig_model_name = "../models/multiclass_nms_test1"
    ie_network.serialize(orig_model_name + ".xml", orig_model_name + ".bin")

    ie = IECore()
    executable_network = ie.load_network(ie_network, 'CPU')
    output = executable_network.infer(inputs={'boxes': input_boxes,
                                              'scores': input_scores})


    print('\033[92m' + "executable_network output {} ".format(output) + '\033[0m')
    
    for key, value in output.items():
        print("# Ngraph result:\n", key, value.shape, value.dtype)
        from generate_multiclass_nms import print_2Dlike
        print_2Dlike(value, "ie_{}".format(key)) 

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
