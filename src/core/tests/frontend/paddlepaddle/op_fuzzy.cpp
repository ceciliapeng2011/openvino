// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_fuzzy.hpp"

#include <cnpy.h>

#include <fstream>

#include "engines_util/test_engines.hpp"
#include "ngraph/ngraph.hpp"
#include "paddle_utils.hpp"
#include "util/test_control.hpp"

using namespace InferenceEngine;
using namespace ov::frontend;

using PDPDFuzzyOpTest = FrontEndFuzzyOpTest;

static const std::vector<std::string> models{
    std::string("argmax"),
    std::string("argmax1"),
    std::string("assign_none/assign_none.pdmodel"),
    std::string("assign_output/assign_output.pdmodel"),
    std::string("assign_value_boolean"),
    std::string("assign_value_fp32"),
    std::string("assign_value_int32"),
    std::string("assign_value_int64"),
    std::string("avgAdaptivePool2D_test1"),
    std::string("avgPool_test1"),
    std::string("avgPool_test10"),
    std::string("avgPool_test11"),
    std::string("avgPool_test2"),
    std::string("avgPool_test3"),
    std::string("avgPool_test4"),
    std::string("avgPool_test5"),
    // avgPool_test6<nchw support is disabled now>,
    std::string("avgPool_test7"),
    std::string("avgPool_test8"),
    std::string("avgPool_test9"),
    std::string("batch_norm_nchw"),
    std::string("batch_norm_nhwc"),
    std::string("bicubic_downsample_false_0"),
    std::string("bicubic_downsample_false_1"),
    std::string("bicubic_downsample_true_0"),
    std::string("bicubic_upsample_false_0"),
    std::string("bicubic_upsample_false_1"),
    std::string("bicubic_upsample_scales"),
    std::string("bicubic_upsample_scales2"),
    std::string("bicubic_upsample_true_0"),
    std::string("bilinear_downsample_false_0"),
    std::string("bilinear_downsample_false_1"),
    std::string("bilinear_downsample_true_0"),
    std::string("bilinear_upsample_false_0"),
    std::string("bilinear_upsample_false_1"),
    std::string("bilinear_upsample_scales"),
    std::string("bilinear_upsample_scales2"),
    std::string("bilinear_upsample_true_0"),
    std::string("bmm"),
    std::string("clip"),
    std::string("conv2d_dilation_assymetric_pads_strides"),
    std::string("conv2d_SAME_padding"),
    std::string("conv2d_strides_assymetric_padding"),
    std::string("conv2d_strides_no_padding"),
    std::string("conv2d_strides_padding"),
    std::string("conv2d_transpose_dilation_assymetric_pads_strides"),
    // conv2d_transpose_SAME_padding(PDPD outputs wrong results),
    std::string("conv2d_transpose_strides_assymetric_padding"),
    std::string("conv2d_transpose_strides_no_padding"),
    std::string("conv2d_transpose_strides_padding"),
    std::string("conv2d_transpose_VALID_padding"),
    std::string("conv2d_VALID_padding"),
    std::string("cumsum"),
    std::string("cumsum_i32"),
    std::string("cumsum_i64"),
    std::string("cumsum_f32"),
    std::string("cumsum_f64"),
    std::string("deformable_conv_default"),
    std::string("deformable_conv_with_pad"),
    std::string("deformable_conv_with_pad_tuple"),
    std::string("deformable_conv_with_pad_list"),
    std::string("deformable_conv_with_stride"),
    std::string("deformable_conv_with_stride_tuple"),
    std::string("deformable_conv_with_stride_list"),
    std::string("deformable_conv_with_dilation"),
    std::string("deformable_conv_with_dilation_tuple"),
    std::string("deformable_conv_with_dilation_list"),
    std::string("deformable_conv_with_pad_stride_dilation"),
    std::string("deformable_conv_with_groups"),
    std::string("deformable_conv_with_deformable_groups"),
    std::string("deformable_conv_with_groups_and_deformable_groups"),
    std::string("deformable_conv_with_mask"),
    std::string("deformable_conv_with_bias"),
    std::string("deformable_conv_with_mask_bias"),
    std::string("deformable_conv_full"),
    std::string("depthwise_conv2d_convolution"),
    std::string("depthwise_conv2d_transpose_convolution"),
    std::string("dropout"),
    std::string("dropout_upscale_in_train"),
    std::string("elementwise_add1"),
    std::string("elementwise_div1"),
    std::string("elementwise_max1"),
    std::string("elementwise_min1"),
    std::string("elementwise_mul1"),
    std::string("elementwise_pow1"),
    std::string("elementwise_sub1"),
    std::string("embedding_0"),
    std::string("embedding_sparse"),
    std::string("embedding_none_weight"),
    std::string("embedding_paddings"),
    std::string("embedding_paddings_neg1"),
    std::string("embedding_tensorIds"),
    std::string("embedding_tensorIds_paddings"),
    std::string("equal"),
    std::string("expand_v2"),
    std::string("expand_v2_tensor"),
    std::string("expand_v2_tensor_list"),
    std::string("exp_test_float32"),
    std::string("fill_any_like"),
    std::string("fill_any_like_f16"),
    std::string("fill_any_like_f32"),
    std::string("fill_any_like_f64"),
    std::string("fill_any_like_i32"),
    std::string("fill_any_like_i64"),
    std::string("fill_constant"),
    std::string("fill_constant_batch_size_like"),
    std::string("fill_constant_int32"),
    std::string("fill_constant_int64"),
    std::string("fill_constant_tensor"),
    std::string("fill_constant_shape_tensor"),
    std::string("fill_constant_shape_tensor_list"),
    std::string("flatten_contiguous_range_test1"),
    std::string("gelu_erf"),
    std::string("gelu_tanh"),
    // greater_equal_big_int64(failure due to CPU inference),
    std::string("greater_equal_float32"),
    std::string("greater_equal_int32"),
    std::string("greater_equal_int64"),
    std::string("hard_sigmoid"),
    std::string("hard_swish"),
    std::string("layer_norm"),
    std::string("layer_norm_noall"),
    std::string("layer_norm_noscale"),
    std::string("layer_norm_noshift"),
    std::string("leaky_relu"),
    std::string("linear_downsample_false_0"),
    std::string("linear_downsample_false_1"),
    std::string("linear_downsample_true_0"),
    std::string("linear_upsample_false_0"),
    std::string("linear_upsample_false_1"),
    std::string("linear_upsample_scales"),
    std::string("linear_upsample_scales2"),
    std::string("linear_upsample_true_0"),
    std::string("log"),
    std::string("logical_not"),
    std::string("loop/loop.pdmodel"),
    std::string("loop_dyn/loop_dyn.pdmodel"),
    std::string("loop_dyn_x/loop_dyn_x.pdmodel"),
    std::string("loop_if/loop_if.pdmodel"),
    std::string("loop_if_loop/loop_if_loop.pdmodel"),
    std::string("loop_if_loop_if/loop_if_loop_if.pdmodel"),
    std::string("loop_if_loop_complex/loop_if_loop_complex.pdmodel"),
    std::string("loop_if_tensor_array/loop_if_tensor_array.pdmodel"),
    std::string("loop_t/loop_t.pdmodel"),
    std::string("loop_tensor_array/loop_tensor_array.pdmodel"),
    std::string("loop_x/loop_x.pdmodel"),
    std::string("conditional_block_const/conditional_block_const.pdmodel"),
    std::string("conditional_block_const_2outputs/conditional_block_const_2outputs.pdmodel"),
    std::string("conditional_block_2inputs/conditional_block_2inputs.pdmodel"),
    std::string("conditional_block_2inputs_2outputs/conditional_block_2inputs_2outputs.pdmodel"),
    std::string("conditional_block_2inputs_dyn/conditional_block_2inputs_dyn.pdmodel"),
    std::string("conditional_block_2inputs_dyn_2outputs/conditional_block_2inputs_dyn_2outputs.pdmodel"),
    std::string("conditional_block_dyn_multiple_consumers/conditional_block_dyn_multiple_consumers.pdmodel"),
    std::string("conditional_block_dyn_multiple_blocks/conditional_block_dyn_multiple_blocks.pdmodel"),
    std::string("conditional_block_dyn_multiple_blocks2/conditional_block_dyn_multiple_blocks2.pdmodel"),
    std::string("conditional_block_dyn_conditionalblock_only/conditional_block_dyn_conditionalblock_only.pdmodel"),
    std::string("matmul_xt"),
    std::string("matmul_xt_yt"),
    std::string("matmul_yt"),
    std::string("matmul_v2_1dx1d"),
    std::string("matmul_v2_1dx2d"),
    std::string("matmul_v2_2dx1d"),
    std::string("matmul_v2_ndxmd"),
    std::string("matmul_v2_xt"),
    std::string("matmul_v2_xt_yt"),
    std::string("matmul_v2_yt"),
    std::string("matrix_nms_by_background"),
    std::string("matrix_nms_by_keep_top_k"),
    std::string("matrix_nms_by_nms_top_k"),
    std::string("matrix_nms_by_post_threshold"),
    std::string("matrix_nms_flipped_coordinates"),
    std::string("matrix_nms_gaussian"),
    std::string("matrix_nms_gaussian_sigma"),
    std::string("matrix_nms_identical_boxes"),
    std::string("matrix_nms_not_normalized"),
    std::string("matrix_nms_not_return_indexed"),
    std::string("matrix_nms_not_return_rois_num"),
    std::string("matrix_nms_not_return_rois_num_neither_index"),
    std::string("matrix_nms_one_batch"),
    std::string("matrix_nms_single_box"),
    std::string("matrix_nms_two_batches_two_classes"),
    std::string("matrix_nms_normalized_random"),
    std::string("matrix_nms_not_normalized_random"),
    std::string("maxAdaptivePool2D_test1"),
    std::string("maxPool_test1"),
    std::string("maxPool_test10"),
    std::string("maxPool_test11"),
    std::string("maxPool_test2"),
    std::string("maxPool_test3"),
    std::string("maxPool_test4"),
    std::string("maxPool_test5"),
    // maxPool_test6(nchw support is disabled now),
    std::string("maxPool_test7"),
    std::string("maxPool_test8"),
    std::string("maxPool_test9"),
    std::string("multiclass_nms_by_background"),
    std::string("multiclass_nms_by_class_id"),
    std::string("multiclass_nms_by_IOU"),
    std::string("multiclass_nms_by_IOU_and_scores"),
    std::string("multiclass_nms_by_keep_top_k"),
    std::string("multiclass_nms_by_nms_eta"),
    std::string("multiclass_nms_by_nms_top_k"),
    std::string("multiclass_nms_flipped_coordinates"),
    std::string("multiclass_nms_identical_boxes"),
    std::string("multiclass_nms_not_normalized"),
    std::string("multiclass_nms_not_return_indexed"),
    std::string("multiclass_nms_single_box"),
    std::string("multiclass_nms_two_batches_two_classes_by_class_id"),
    std::string("multiclass_nms_normalized_random"),
    std::string("multiclass_nms_not_normalized_random"),
    std::string("nearest_downsample_false_0"),
    std::string("nearest_downsample_false_1"),
    std::string("nearest_upsample_false_0"),
    std::string("nearest_upsample_false_1"),
    std::string("pad3d_test1"),
    std::string("pad3d_test2"),
    std::string("pad3d_test3"),
    // pad3d_test4,
    std::string("pow_float32"),
    std::string("pow_int32"),
    std::string("pow_int64"),
    // pow_int64_out_of_range(out of range of OV int64),
    std::string("pow_y_tensor"),
    std::string("prior_box_attrs_mmar_order_true"),
    std::string("prior_box_default"),
    std::string("prior_box_flip_clip_false"),
    std::string("prior_box_max_sizes_none"),
    std::string("range0"),
    std::string("range1"),
    std::string("range2"),
    std::string("relu"),
    std::string("relu6"),
    std::string("relu6_1"),
    std::string("reshape"),
    std::string("reshape_tensor"),
    std::string("reshape_tensor_list"),
    std::string("rnn_lstm_layer_1_bidirectional"),
    std::string("rnn_lstm_layer_1_forward"),
    std::string("rnn_lstm_layer_2_bidirectional"),
    std::string("rnn_lstm_layer_2_forward"),
    std::string("rnn_lstm_layer_1_forward_seq_len_4"),
    std::string("rnn_lstm_layer_2_bidirectional_seq_len_4"),
    std::string("scale_bias_after_float32"),
    std::string("scale_bias_after_int32"),
    std::string("scale_bias_after_int64"),
    std::string("scale_bias_before_float32"),
    std::string("scale_bias_before_int32"),
    std::string("scale_bias_before_int64"),
    std::string("scale_tensor_bias_after"),
    std::string("scale_tensor_bias_before"),
    std::string("shape"),
    std::string("sigmoid"),
    std::string("slice"),
    std::string("slice_1d"),
    std::string("slice_decrease_axis/slice_decrease_axis.pdmodel"),
    std::string("slice_decrease_axis_all/slice_decrease_axis_all.pdmodel"),
    std::string("slice_reshape/slice_reshape.pdmodel"),
    std::string("softmax"),
    std::string("softmax_minus"),
    std::string("softplus_default_params"),
    std::string("split_test1"),
    std::string("split_test2"),
    std::string("split_test3"),
    std::string("split_test4"),
    std::string("split_test5"),
    std::string("split_test6"),
    std::string("split_test_dim_int32"),
    std::string("split_test_dim_int64"),
    std::string("split_test_list"),
    std::string("split_test_list_tensor"),
    std::string("squeeze"),
    std::string("squeeze_null_axes"),
    std::string("stack_test_float32"),
    std::string("stack_test_int32"),
    std::string("stack_test_neg_axis"),
    std::string("stack_test_none_axis"),
    std::string("tanh"),
    std::string("trilinear_downsample_false_0"),
    std::string("trilinear_downsample_false_1"),
    std::string("trilinear_downsample_true_0"),
    std::string("trilinear_upsample_false_0"),
    std::string("trilinear_upsample_false_1"),
    std::string("trilinear_upsample_scales"),
    std::string("trilinear_upsample_scales2"),
    std::string("trilinear_upsample_true_0"),
    std::string("unsqueeze"),
    std::string("yolo_box_clip_box"),
    std::string("yolo_box_default"),
    std::string("yolo_box_scale_xy"),
    std::string("yolo_box_uneven_wh")};

INSTANTIATE_TEST_SUITE_P(PDPDFuzzyOpTest,
                         FrontEndFuzzyOpTest,
                         ::testing::Combine(::testing::Values(PADDLE_FE),
                                            ::testing::Values(std::string(TEST_PADDLE_MODELS_DIRNAME)),
                                            ::testing::ValuesIn(models)),
                         PDPDFuzzyOpTest::getTestCaseName);
