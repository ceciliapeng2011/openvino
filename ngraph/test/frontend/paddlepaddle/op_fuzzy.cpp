// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_fuzzy.hpp"

static const std::string PDPD = "pdpd";
using PDPDFuzzyOpTest = FrontEndFuzzyOpTest;

static const std::vector<std::string> models{
    std::string("argmax"),
    std::string("argmax1"),
    std::string("assign_value_boolean"),
    std::string("assign_value_fp32"),
    std::string("assign_value_int32"),
    std::string("assign_value_int64"),
    std::string("batch_norm_nchw"),
    std::string("batch_norm_nhwc"),
    std::string("clip"),
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
    std::string("relu"),
};

INSTANTIATE_TEST_SUITE_P(PDPDFuzzyOpTest,
                         FrontEndFuzzyOpTest,
                         ::testing::Combine(::testing::Values(PDPD),
                                            ::testing::Values(std::string(TEST_PDPD_MODELS)),
                                            ::testing::ValuesIn(models)),
                         PDPDFuzzyOpTest::getTestCaseName);
