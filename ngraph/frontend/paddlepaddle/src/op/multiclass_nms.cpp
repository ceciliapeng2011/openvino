// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multiclass_nms.hpp"
#include <ngraph/opsets/opset8.hpp>
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs multiclass_nms(const NodeContext& node)
                {
                    using namespace ngraph;
                    using namespace opset8;
                    using namespace element;

                    auto bboxes = node.get_ng_input("BBoxes");
                    auto scores = node.get_ng_input("Scores");

                    auto score_threshold = node.get_attribute<float>("score_threshold");
                    auto iou_threshold = node.get_attribute<float>("nms_threshold");
                    auto nms_top_k = node.get_attribute<int>("nms_top_k");
                    auto keep_top_k = node.get_attribute<int>("keep_top_k");
                    auto background_class = node.get_attribute<int>("background_label");
                    auto nms_eta = node.get_attribute<float>("nms_eta");

                    auto normalized = node.get_attribute<bool>("normalized");

                    if( !normalized )
                    {
                        auto node_value_one = Constant::create<float>(f32, {1}, {1.0});

                        auto node_axis2 = Constant::create<int64_t>(i64, {}, {2});
                        auto node_new_bboxes = std::make_shared<Split>(bboxes, node_axis2, 4)->outputs();
                        auto node_new_xmax = std::make_shared<Add>(node_new_bboxes[2], node_value_one);
                        auto node_new_ymax = std::make_shared<Add>(node_new_bboxes[3], node_value_one);

                        bboxes = std::make_shared<Concat>(OutputVector{node_new_bboxes[0], node_new_bboxes[1], node_new_xmax, node_new_ymax}, 2);
                    }

                    NamedOutputs named_outputs;
                    std::vector<Output<Node>> nms_outputs;
                    nms_outputs = std::make_shared<MulticlassNms>(
                            bboxes,
                            scores,
                            MulticlassNms::SortResultType::CLASSID,
                            false,
                            i64, // TODO
                            iou_threshold,
                            score_threshold,
                            nms_top_k,
                            keep_top_k,
                            background_class,
                            nms_eta)->outputs();

                    auto out_names = node.get_output_names();
                    PDPD_ASSERT(out_names.size() == 3, "Unexpected number of outputs of MulticlassNMS");

                    named_outputs["Out"] = {nms_outputs[0]};
                    named_outputs["Index"] = {nms_outputs[1]};
                    named_outputs["NmsRoisNum"] = {nms_outputs[2]};
                    return named_outputs;
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
