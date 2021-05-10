#
# range paddle model generator
#
import numpy as np
from save_model import saveModel


def pdpd_range(name : str, x, start, end, step):
    import paddle as pdpd
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
        # Range op only support fill_constant input, since dynamic op is not supported in ov
        out = pdpd.fluid.layers.range(start, end, step, "float32")
        out = pdpd.equal(node_x, out)
        out = pdpd.cast(out, np.float32)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
                feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]])

    return outs[0]

def main():
    start = 1.5
    end = 10.5
    step = 2
    data = np.random.random([1, 5]).astype("float32")
    pdpd_range("range", data, start, end, step)

if __name__ == "__main__":
    main()     
