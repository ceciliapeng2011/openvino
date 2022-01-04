# ref: https://www.paddlepaddle.org.cn/tutorials/projectdetail/1998893#anchor-2

import os
import sys

import numpy as np
import paddle

from save_model import exportModel, saveModel


def loop():
    paddle.enable_static()
    x = np.full(shape=[1], fill_value=0, dtype='int64')

    def cond(i, ten):
        return ten >= i

    def body(i, dummy):
        i = i + 1
        return i, dummy

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_i = paddle.full(shape=[1], fill_value=0, dtype='int64', name='i')
        node_i = paddle.fluid.layers.nn.elementwise_add(node_i, node_x)
        node_ten = paddle.full(shape=[1], fill_value=10, dtype='int64', name='ten')

        out, dummy = paddle.static.nn.while_loop(cond, body, [node_i, node_ten], name='while_loop')

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        res = exe.run(paddle.static.default_main_program(), feed={'x':x}, fetch_list=out)

        saveModel('loop', exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[res[0]], target_dir=sys.argv[1])

    return res

def loop_x():
    paddle.enable_static()
    x = np.full(shape=[1], fill_value=1, dtype='int64')

    def cond(i, ten):
        return ten >= i

    def body(i, t):
        i = i + x
        return i, t

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_i = paddle.full(shape=[1], fill_value=0, dtype='int64', name='i')
        node_i = paddle.fluid.layers.nn.elementwise_add(node_i, node_x)
        node_ten = paddle.full(shape=[1], fill_value=10, dtype='int64', name='ten')

        out, dummy = paddle.static.nn.while_loop(cond, body, [node_i, node_ten], name='while_loop')

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        res = exe.run(paddle.static.default_main_program(), feed={'x':x}, fetch_list=out)

        saveModel('loop_x', exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[res[0]], target_dir=sys.argv[1])

    return res

def loop_t():
    paddle.enable_static()
    x = np.full(shape=[1], fill_value=0, dtype='int64')

    def cond(i, ten):
        return ten >= i

    def body(i, t):
        i = i + 1
        t = t - 1
        return i, t

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_i = paddle.full(shape=[1], fill_value=0, dtype='int64', name='i')
        node_i = paddle.fluid.layers.nn.elementwise_add(node_i, node_x)
        node_ten = paddle.full(shape=[1], fill_value=10, dtype='int64', name='ten')

        out_i,out_t = paddle.static.nn.while_loop(cond, body, [node_i, node_ten], name='while_loop')

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        res = exe.run(paddle.static.default_main_program(), feed={'x':x}, fetch_list=[out_i,out_t])

        saveModel('loop_t', exe, feedkeys=['x'], fetchlist=[out_i,out_t], inputs=[x], outputs=res, target_dir=sys.argv[1])

    return res

def loop_dyn():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x):
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        
        i = i + x
        t = paddle.full(shape=[1], fill_value=10, dtype='int64')
        j = i + 1

        while t >= i:
            i = i + 1

        return i, j

    x = np.full(shape=[1], fill_value=0, dtype='int64')
    return exportModel('loop_dyn', test_model, [x], target_dir=sys.argv[1])


def loop_dyn_x():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x):
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        
        i = i + x
        t = paddle.full(shape=[1], fill_value=10, dtype='int64')

        while t >= i:
            i = i + x

        return i

    x = np.full(shape=[1], fill_value=1, dtype='int64')
    return exportModel('loop_dyn_x', test_model, [x], target_dir=sys.argv[1])

def loop_if():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x):
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        
        i = i + x
        t = paddle.full(shape=[1], fill_value=10, dtype='int64')

        while t >= i:
            if i < 5:
                i = i + x
            else:
                i = i + 2 * x

        return i

    x = np.full(shape=[1], fill_value=1, dtype='int64')
    return exportModel('loop_if', test_model, [x], target_dir=sys.argv[1])

if __name__ == "__main__":
    print(loop())
    print(loop_dyn())

    print(loop_t())
    print(loop_x())

    print(loop_dyn_x().numpy())
    print(loop_if().numpy())


