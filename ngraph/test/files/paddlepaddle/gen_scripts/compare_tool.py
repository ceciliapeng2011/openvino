import numpy as np
import os

def checker(outs, name, dump=False):
    if type(outs) is list:
        print("\n###{} list".format(name))
        for i in range(len(outs)):      
            print("output {} shape {}, type {} ".format(i, outs[i].shape, outs[i].dtype))
            if(dump):
                print(outs[i])
    if type(outs) is dict:
        print("\n###{} dict".format(name))
        for i in outs:
            print("output {} shape {}, type {} ".format(i, outs[i].shape, outs[i].dtype))
            if(dump):
                print(outs[i]) 
    if type(outs) is np.ndarray:
        print("\n###{} numpy.ndarray".format(name))
        print("output shape {}, type {} ".format(outs.shape, outs.dtype))


def print_2Dlike(data, filename):
    with open(filename+".txt", 'w') as f:
        print(data.shape, file=f)
        _data = data.copy()
        if len(_data.shape)==1:
            _data = np.expand_dims(_data, axis=1)
        if len(_data.shape)>2:
            _data = _data.reshape((_data.shape[0],-1))
        print(_data.shape, file=f)
        
        print("\n[", file=f)
        for j in range(_data.shape[0]):
            line=""
            for i in range(_data.shape[1]):
                line+="{:.2f}".format(_data[j, i]) + "  "
            print(line, file=f)
        print("]\n", file=f)
    f.close()            

def validate(pred_ref: list, pred_ie: dict, rtol=1e-05, atol=1e-08):
    checker(pred_ref, "paddle")
    checker(pred_ie, "openvino")

    if isinstance(pred_ie, list) and isinstance(pred_ref, list):
        comp1 = np.all(np.isclose(pred_ref, pred_ie, rtol=rtol, atol=atol, equal_nan=True))
        if not comp1: 
            print('\033[91m' + "PDPD and ONNX results are different "+ '\033[0m')
        else:
            print('\033[92m' + "PDPD, ONNX and IE results are identical"+ '\033[0m')

    if isinstance(pred_ie, dict) and isinstance(pred_ref, list):
        idx = 0
        for key in pred_ie:
            comp2 = np.all(np.isclose(pred_ref[idx], pred_ie[key], rtol=rtol, atol=atol, equal_nan=True))
            comp2_type = pred_ref[idx].dtype == pred_ie[key].dtype
            dtype_compare = "data type {}".format("identical" if comp2_type else "different")
            if not comp2:
                print('\033[91m' + "PDPD and IE results are different at {}, {} ".format(idx, dtype_compare) + '\033[0m')
                print_2Dlike(pred_ref[idx], "pdpd{}".format(idx))        
                print_2Dlike(pred_ie[key], "ie{}".format(idx))                  
            else:
                print('\033[92m' + "PDPD and IE results are identical at {}, {} ".format(idx, dtype_compare) + '\033[0m') 
            idx += 1  