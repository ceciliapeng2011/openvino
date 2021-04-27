#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel

# bboxes shape (N, M, 4)
# scores shape (N, C, M) 
def multiclass_nms(name : str, bboxes, scores, attrs : dict):
    import paddle as pdpd
    from ppdet.modeling import ops
    pdpd.enable_static()
   
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_boxes = pdpd.static.data(name='bboxes', shape=bboxes.shape,
                                dtype=bboxes.dtype, lod_level=1)
        node_scores = pdpd.static.data(name='scores', shape=scores.shape,
                                dtype=scores.dtype, lod_level=1)

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
       
        out_np, nms_rois_num_np, index_np = exe.run(
            feed={'bboxes': bboxes, 'scores': scores},
            fetch_list=output, return_numpy=False)

        out = np.array(out_np)
        index = np.array(index_np)
        nms_rois_num = np.array(nms_rois_num_np)

        # Save inputs in order of ngraph function, to facilite Fuzzy test, 
        # which accepts inputs and outputs in this order as well. 
        saveModel(name, exe, feedkeys=['bboxes', 'scores'], fetchlist=output, inputs=[bboxes, scores], outputs=[out, index, nms_rois_num])

    #
    print('out_np: ', out.shape)
    print('index_np: ', index.shape)
    print('nms_rois_num_np: ', nms_rois_num.shape, nms_rois_num)
    return [out, index, nms_rois_num]

def onnx_run(bboxes, scores, onnx_model="../models/multiclass_nms_test1/multiclass_nms_test1.onnx"):
    import onnxruntime as rt

    sess = rt.InferenceSession(onnx_model)
    for input in sess.get_inputs():
        print(input.name)
    for output in sess.get_outputs():
        print(output.name)        
    pred_onx = sess.run(None, {"bboxes": bboxes, "scores": scores})
    #print(type(pred_onx), len(pred_onx))
    return pred_onx

def OV_frontend_run(bboxes, scores, path_to_pdpd_model="../models/multiclass_nms_test1/", user_shapes=[]):
    from ngraph import FrontEndManager
    from ngraph import function_to_cnn
    from ngraph import PartialShape

    fem = FrontEndManager()
    print('fem.availableFrontEnds: ' + str(fem.availableFrontEnds()))
    print('Initializing new FE for framework {}'.format("pdpd"))
    fe = fem.loadByFramework("pdpd")
    print(fe)
    inputModel = fe.loadFromFile(path_to_pdpd_model)
    try:
        place = inputModel.getPlaceByTensorName('x')
        print(place)
        print(place.isEqual(None))
    except Exception:
        print('Failed to call model API with hardcoded input name "x"')

    if len(user_shapes) > 0:
        for user_shape in user_shapes:
            inputModel.setPartialShape(user_shape['node'], PartialShape(user_shape['shape']))
    nGraphModel = fe.convert(inputModel)

    net = function_to_cnn(nGraphModel)

    from openvino.inference_engine import IECore

    #IE inference
    ie = IECore()
    exec_net = ie.load_network(net, "CPU")
    pred_ie = exec_net.infer({"bboxes": bboxes, "scores": scores})
    print(type(pred_ie))
    return pred_ie    

def checker(outs, name, dump=False):
    if type(outs) is list:
        print("\n###{} list".format(name))
        for i in range(len(outs)):      
            print("output{} shape {}, type {} ".format(i, outs[i].shape, outs[i].dtype))
            if(dump):
                print(outs[i])
    if type(outs) is dict:
        print("\n###{} dict".format(name))
        for i in outs:
            print("output{} shape {}, type {} ".format(i, outs[i].shape, outs[i].dtype))
            if(dump):
                print(outs[i]) 
    if type(outs) is np.ndarray:
        print("\n###{} numpy.ndarray".format(name))
        print("output shape {}, type {} ".format(outs.shape, outs.dtype))


def print_2Dlike(data, filename):
    with open(filename+".txt", 'w') as f:
        print(data.shape, file=f)
        _data = data.copy()
        _data = np.squeeze(_data)
        _data.reshape(-1, data.shape[len(data.shape)-1])
        print(_data.shape, file=f)
        print("\n[", file=f)
        for j in range(_data.shape[0]):
            line=""
            for i in range(_data.shape[1]):
                line+="{:.2f}".format(_data[j, i]) + "  "
            print(line, file=f)
        print("]\n", file=f)
    f.close()            

def validate(pred_pdpd: list, pred_onx: list, pred_ie: dict, rtol=1e-05, atol=1e-08):
    checker(pred_pdpd, "paddle")
    #checker(pred_onx, "onnx")
    checker(pred_ie, "openvino")
    '''
    comp1 = np.all(np.isclose(pred_pdpd, pred_onx, rtol=rtol, atol=atol, equal_nan=True))
    if not comp1: 
        print('\033[91m' + "PDPD and ONNX results are different "+ '\033[0m')
    else:
        print('\033[92m' + "PDPD, ONNX and IE results are identical"+ '\033[0m')

    # compare results: IE vs PDPD vs ONNX
    idx = 0
    for key in pred_ie:
        comp1 = np.all(np.isclose(pred_pdpd[idx], pred_onx[idx], rtol=rtol, atol=atol, equal_nan=True))
        #comp1 = np.all(np.isclose([1,2.1], [1,2], rtol=1e-05, atol=1e-08, equal_nan=True))
        #np.all(np.isclose(res_pdpd[0], list(res_ie.values())[0], rtol=1e-4, atol=1e-5))
        if not comp1: 
            print('\033[91m' + "PDPD and ONNX results are different at {} ".format(idx) + '\033[0m')

        comp2 = np.all(np.isclose(pred_pdpd[idx], pred_ie[key], rtol=rtol, atol=atol, equal_nan=True))
        #np.all(np.isclose(res_pdpd[0], list(res_ie.values())[0], rtol=1e-4, atol=1e-5))
        if not comp2:
            print('\033[91m' + "PDPD and IE results are different at {} ".format(idx) + '\033[0m')

        comp3 = np.all(np.isclose(pred_ie[key], pred_onx[idx], rtol=rtol, atol=atol, equal_nan=True))
        #np.all(np.isclose(res_pdpd[0], list(res_ie.values())[0], rtol=1e-4, atol=1e-5))
        if not comp3:
            print('\033[91m' + "ONNX and IE results are different at {} ".format(idx) + '\033[0m')            

        print_2Dlike(pred_pdpd[idx], "pdpd{}".format(idx))
        print_2Dlike(pred_onx[idx], "onnx{}".format(idx))            
        print_2Dlike(pred_ie[key], "ie{}".format(idx))

        if comp1 and comp2 and comp3:
            print('\033[92m' + "PDPD, ONNX and IE results are identical at {} ".format(idx) + '\033[0m')

        idx += 1   
    '''         

'''
ref: paddle/fluid/tests/unittests/test_multiclass_nms_op.py
OpTest
||				
TestMulticlassNMSLoDInput            /                        TestMulticlassNMSOp
||								                                        ||						
TestMulticlassNMS2LoDInput /TestMulticlassNMSNoBox	          TestMulticlassNMS2Op  /  TestMulticlassNMSOpNoOutput
||								                                        ||
TestMulticlassNMS3LoDInput				                      TestMulticlassNMS3Op
||								                                        ||
TestMulticlassNMS3LoDNoOutput			                    TestMulticlassNMS3OpNoOutput
'''
def main():
    # multiclass_nms

    '''
    # case1
    boxes = np.array([[
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.1, 1.0, 1.1],
        [0.0, -0.1, 1.0, 0.9],
        [0.0, 10.0, 1.0, 11.0],
        [0.0, 10.1, 1.0, 11.1],
        [0.0, 100.0, 1.0, 101.0]
    ]]).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)

    pdpd_attrs = {
        'background_label': -1,
        'score_threshold': 0.4,
        'nms_top_k': 200,
        'nms_threshold': 0.5,
        'keep_top_k': 200,
        'normalized': False,
        'nms_eta': 1.0
    }
    '''

    def softmax(x):
        # clip to shiftx, otherwise, when calc loss with
        # log(exp(shiftx)), may get log(0)=INF
        shiftx = (x - np.max(x)).clip(-64.)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    # case 2
    N = 1  #onnx multiclass_nms only supports input[batch_size] == 1.
    M = 1200
    C = 21
    BOX_SIZE = 4
    background = -1
    nms_threshold = 0.3
    nms_top_k = 400  #max_output_boxes_per_class
    keep_top_k = -1
    score_threshold = 0.02

    scores = np.random.random((N * M, C)).astype('float32')

    scores = np.apply_along_axis(softmax, 1, scores)
    scores = np.reshape(scores, (N, M, C))
    scores = np.transpose(scores, (0, 2, 1))

    boxes = np.random.random((N, M, BOX_SIZE)).astype('float32')
    boxes[:, :, 0:2] = boxes[:, :, 0:2] * 0.5
    boxes[:, :, 2:4] = boxes[:, :, 2:4] * 0.5 + 0.5

    pdpd_attrs = {
        'background_label': background,
        'score_threshold': score_threshold,
        'nms_top_k': nms_top_k,
        'nms_threshold': nms_threshold,
        'keep_top_k': keep_top_k,
        'normalized': False,
        'nms_eta': 1.0
    }

    '''
    M = 1200
    N = 7
    C = 21
    BOX_SIZE = 4

    boxes_np = np.random.random((M, C, BOX_SIZE)).astype('float32')
    scores = np.random.random((N * M, C)).astype('float32')
    scores = np.apply_along_axis(softmax, 1, scores)
    scores = np.reshape(scores, (N, M, C))
    scores_np = np.transpose(scores, (0, 2, 1))

    boxes_data = fluid.data(
        name='bboxes', shape=[M, C, BOX_SIZE], dtype='float32')
    scores_data = fluid.data(
        name='scores', shape=[N, C, M], dtype='float32') 
    '''   

    # bboxes shape (N, M, 4) 
    # scores shape (N, C, M)  
    data_bboxes = boxes
    data_scores = scores 

    # For any change to pdpd_attrs, do -
    # step 1. generate paddle model
    pred_pdpd = multiclass_nms('multiclass_nms_test1', data_bboxes, data_scores, pdpd_attrs)

    from multiclass_nms3_ngraph import ngraph_multiclass_nms3
    pred_ngraph = ngraph_multiclass_nms3(data_bboxes, data_scores, score_threshold=score_threshold, iou_threshold=nms_threshold,
                           max_output_boxes_per_class=nms_top_k, output_type='i32')

    # step 2. generate onnx model
    # !paddle2onnx --model_dir=../models/yolo_box_test1/ --save_file=../models/yolo_box_test1/yolo_box_test1.onnx --opset_version=10
    #import subprocess
    #subprocess.run(["paddle2onnx", "--model_dir=../models/multiclass_nms_test1/", "--save_file=../models/multiclass_nms_test1/multiclass_nms_test1.onnx", "--opset_version=11", "--enable_onnx_checker=True"])
    #pred_onx = onnx_run(data_bboxes, data_scores)

    # step 3. run from frontend API
    #pred_ie = OV_frontend_run(data_bboxes, data_scores)

    # step 4. compare 
    # Try different tolerence
    #validate(pred_pdpd, pred_onx, pred_ie)
    #validate(pred_pdpd, [], pred_ie, rtol=1e-4, atol=1e-5) 


if __name__ == "__main__":
    main()     