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

                    auto type_index = node.get_out_port_type("Index");
                    auto type_num = node.get_out_port_type("NmsRoisNum");
                    PDPD_ASSERT((type_index == i32 || type_index == i64) &&
                                    (type_num == i32 || type_num == i64),
                                "Unexpected data type of outputs of MulticlassNMS");

                    // auto normalized = node.get_attribute<bool>("normalized");

                    NamedOutputs named_outputs;
                    std::vector<Output<Node>> nms_outputs;
                    nms_outputs =
                        std::make_shared<MulticlassNms>(bboxes,
                                                        scores,
                                                        MulticlassNms::SortResultType::CLASSID,
                                                        false,
                                                        type_index,
                                                        iou_threshold,
                                                        score_threshold,
                                                        nms_top_k,
                                                        keep_top_k,
                                                        background_class,
                                                        nms_eta)
                            ->outputs();

                    auto out_names = node.get_output_names();
                    PDPD_ASSERT(out_names.size() == 3,
                                "Unexpected number of outputs of MulticlassNMS");

                    named_outputs["Out"] = {nms_outputs[0]};
                    named_outputs["Index"] = {nms_outputs[1]};
                    named_outputs["NmsRoisNum"] = {nms_outputs[2]};

                    if (type_num != type_index)
                    {
                        // adapter
                        auto node_convert = std::make_shared<Convert>(nms_outputs[2], type_num);
                        named_outputs["NmsRoisNum"] = {node_convert};
                    }

                    return named_outputs;
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
