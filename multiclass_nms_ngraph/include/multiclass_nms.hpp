// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <ngraph/runtime/host_tensor.hpp>
#include <vector>
#include "ngraph/node.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void multiclass_nms(const float* boxes_data,
                                const Shape& boxes_data_shape,
                                const float* scores_data,
                                const Shape& scores_data_shape,
                                float iou_threshold,
                                float score_threshold,
                                int64_t nms_top_k,
                                int64_t keep_top_k,
                                int64_t background_class,                                
                                float nms_eta,
                                int64_t* selected_indices,
                                const Shape& selected_indices_shape,
                                float* selected_scores,
                                const Shape& selected_scores_shape,
                                int64_t* valid_outputs,
                                const std::string &sort_result);

            void multiclass_nms_postprocessing(const HostTensorVector& outputs,
                                               const ngraph::element::Type output_type,
                                               const std::vector<int64_t>& selected_indices,
                                               const std::vector<float>& selected_scores,
                                               int64_t valid_outputs,
                                               const ngraph::element::Type selected_scores_type);
        }
    }
}
