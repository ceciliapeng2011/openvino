def onnx_run(x, img_size, onnx_model):
    import onnxruntime as rt

    sess = rt.InferenceSession(onnx_model)
    for input in sess.get_inputs():
        print(input.name)
    for output in sess.get_outputs():
        print(output.name)        
    pred_onx = sess.run(None, {"img_size": img_size, "x": x})
    print(pred_onx)
    return pred_onx

def OV_frontend_run(x, img_size, path_to_model, user_shapes=[]):
    from ngraph import FrontEndManager # pylint: disable=import-error
    from ngraph import function_to_cnn # pylint: disable=import-error
    from ngraph import PartialShape    # pylint: disable=import-error

    fem = FrontEndManager()
    print('fem.availableFrontEnds: ' + str(fem.availableFrontEnds()))
    print('Initializing new FE for framework {}'.format("pdpd"))
    fe = fem.loadByFramework("pdpd")
    print(fe)
    inputModel = fe.loadFromFile(path_to_model)
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

    import os
    orig_model_name = "ir"
    net.serialize(os.path.join(path_to_model, orig_model_name + ".xml"), 
                    os.path.join(path_to_model, orig_model_name + ".bin"))

    from openvino.inference_engine import IECore

    #IE inference
    ie = IECore()
    exec_net = ie.load_network(net, "CPU")
    pred_ie = exec_net.infer({"img_size": img_size, "x": x})
    print((pred_ie))

    wanted = {k:v for k, v in pred_ie.items() if k.startswith('save_infer_model')}
    return wanted