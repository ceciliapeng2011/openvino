#
# pool2d paddle model generator
#
import numpy as np
#from save_model import saveModel

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
                                        normalized=attrs['normalized'])                                                

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
       
        outs = exe.run(
            feed={'bboxes': bboxes, 'scores': scores},
            fetch_list=[output], return_numpy=False)

        # Save inputs in order of ngraph function, to facilite Fuzzy test, 
        # which accepts inputs and outputs in this order as well. 
        #saveModel(name, exe, feedkeys=['bboxes', 'scores'], fetchlist=[output], inputs=[bboxes, scores], outputs=outs)
        #saveModel(name, exe, feedkeys=['bboxes', 'scores'], fetchlist=[output], inputs=[], outputs=[])

    #
    lod_num = len(outs)
    w, h = outs[0].shape()
    if w == 1 and h == 1:
        return None
    lod_shape = [lod_num, w, h]
    result = np.zeros(lod_shape)

    for lod in outs:
        rows, length = lod.shape()
        for i in range(0, rows):
            for index in range(0, length):
                result[0][i][index] = lod._get_float_element(i * 6 + index)
    
    print("multiclass_nms: ", type(result), result.shape)
    return result

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
    background = 0
    nms_threshold = 0.3
    nms_top_k = 400
    keep_top_k = 200
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


if __name__ == "__main__":
    main()     