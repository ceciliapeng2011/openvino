from __future__ import print_function
import unittest

import contextlib
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import Program
from paddle.fluid import core
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.dygraph import base

import ppdet.modeling.ops as ops

class LayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 111

    @classmethod
    def tearDownClass(cls):
        pass

    def _get_place(self, force_to_use_cpu=False):
        # this option for ops that only have cpu kernel
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    @contextlib.contextmanager
    def static_graph(self):
        paddle.enable_static()
        scope = fluid.core.Scope()
        program = Program()
        with fluid.scope_guard(scope):
            with fluid.program_guard(program):
                paddle.seed(self.seed)
                paddle.framework.random._manual_program_seed(self.seed)
                yield

    def get_static_graph_result(self,
                                feed,
                                fetch_list,
                                with_lod=False,
                                force_to_use_cpu=False):
        exe = fluid.Executor(self._get_place(force_to_use_cpu))
        exe.run(fluid.default_startup_program())
        return exe.run(fluid.default_main_program(),
                       feed=feed,
                       fetch_list=fetch_list,
                       return_numpy=(not with_lod))

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        paddle.disable_static()
        with fluid.dygraph.guard(
                self._get_place(force_to_use_cpu=force_to_use_cpu)):
            paddle.seed(self.seed)
            paddle.framework.random._manual_program_seed(self.seed)
            yield


class TestMulticlassNms(LayerTest):
    def test_multiclass_nms(self):
        boxes_np = np.random.rand(10, 50, 4).astype('float32')
        scores_np = np.random.rand(10, 81, 50).astype('float32')
        with self.static_graph():
            boxes = paddle.static.data(
                name='bboxes', shape=[10, 50, 4], dtype='float32')
            scores = paddle.static.data(
                name='scores', shape=[10, 81, 50], dtype='float32')

            output = ops.multiclass_nms(
                bboxes=boxes,
                scores=scores,
                background_label=0,
                score_threshold=0.5,
                nms_top_k=400,
                nms_threshold=0.3,
                keep_top_k=200,
                normalized=False,
                return_index=True)
            out_np, index_np, nms_rois_num_np = self.get_static_graph_result(
                feed={
                    'bboxes': boxes_np,
                    'scores': scores_np,
                },
                fetch_list=output,
                with_lod=True)
            print('out_np: ', np.array(out_np))
            print('index_np: ', np.array(index_np))
            print('nms_rois_num_np: ', np.array(nms_rois_num_np))

if __name__ == '__main__':
    unittest.main()
