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
        if np.ndim(data) == 0:
            print(data.shape, file=f)
            print(data, file=f)
        else:
            data_shape = np.array(data.shape)
            idx = np.where(data_shape != 1)[0][0]
            
            _data = data.copy()
            _data.flatten()
            _data = _data.reshape(data_shape[idx],-1)

            print(data_shape, data_shape != 1, np.where(data_shape != 1), idx, data_shape[idx], _data.shape)

            print(data.shape, file=f)
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
    retval = True

    if isinstance(pred_ie, list) and isinstance(pred_ref, list):
        comp1 = np.all(np.isclose(pred_ref, pred_ie, rtol=rtol, atol=atol, equal_nan=True))
        if not comp1: 
            print('\033[91m' + "PDPD and ONNX results are different "+ '\033[0m')
            retval = False
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
                retval = False              
            else:
                print('\033[92m' + "PDPD and IE results are identical at {}, {} ".format(idx, dtype_compare) + '\033[0m') 
            idx += 1 

    return retval



if __name__ == "__main__":
    print_2Dlike(np.ones(5), "shape5")
    print_2Dlike(np.ones((5,)), "shape5_0")
    print_2Dlike(np.ones((5, 1)), "shape5_1")
    print_2Dlike(np.ones((1, 5)), "shape1_5")
    print_2Dlike(np.ones((1, 1, 5)), "shape1_1_5")
    print_2Dlike(np.ones((1, 5, 1)), "shape1_5_1")
    print_2Dlike(np.ones((5, 1, 1)), "shape5_1_1")
    print_2Dlike(np.ones((5, 6, 1)), "shape5_6_1")
    print_2Dlike(np.ones((5, 6, 7)), "shape5_6_7")