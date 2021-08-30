#
# paddle model generator
# 
# reference impl:
# https://github.com/PaddlePaddle/Paddle/blob/e04b66f2d272d68f77dcd94cb2956938475411d8/python/paddle/fluid/tests/unittests/test_transformer_api.py#L184
#
import numpy as np
from save_model import saveModel
import sys


def layer_norm(name : str, x, scale=True, shift=True, begin_norm_axis=1, epsilon=1e-05, weight_array=None, bias_array=None, act=None):
    import paddle as pdpd
    pdpd.enable_static()

    node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
    scale_attr = pdpd.ParamAttr(name="scale1", initializer=pdpd.nn.initializer.Assign(weight_array)) if weight_array is not None else None
    bias_attr = pdpd.ParamAttr(name="bias1", initializer=pdpd.nn.initializer.Assign(bias_array)) if bias_array is not None else None

    out = pdpd.static.nn.layer_norm(node_x, scale, shift, begin_norm_axis, epsilon, param_attr=scale_attr, bias_attr=bias_attr, act=act)

    cpu = pdpd.static.cpu_places(1)
    exe = pdpd.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(pdpd.static.default_startup_program())

    outs = exe.run(
        feed={'x': x},
        fetch_list=[out])             

    saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    batch_size, sequence_length, d_model = 7, 9, 81
    data = np.random.rand(batch_size, sequence_length, d_model).astype("float32")    
    # data layout is NCHW
    weight = np.random.rand(sequence_length*d_model,).astype("float32") 
    bias = np.random.rand(sequence_length*d_model,).astype("float32")
    print(weight.shape, bias.shape, data.shape)
    layer_norm("layer_norm", data, True, True, 1, 1e-05, weight, bias, 'relu')

if __name__ == "__main__":
    main()     