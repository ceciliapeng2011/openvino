// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

class MKLDNNMatrixNmsNode : public MKLDNNNode {
public:
    MKLDNNMatrixNmsNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

    struct filteredBoxes {
        float score;
        int batch_index;
        int class_index;
        int box_index;
        filteredBoxes() = default;
        filteredBoxes(float _score, int _batch_index, int _class_index, int _box_index) :
                score(_score), batch_index(_batch_index), class_index(_class_index), box_index(_box_index) {}
    };

    struct boxInfo {
        float score;
        int idx;
        int suppress_begin_index;
    };

private:
    // input
    const size_t NMS_BOXES = 0;
    const size_t NMS_SCORES = 1;

    // output
    const size_t NMS_SELECTED_OUTPUTS = 0;
    const size_t NMS_SELECTED_INDICES = 1;
    const size_t NMS_VALID_OUTPUTS = 2;

    enum class boxEncoding {
        CORNER,
        CENTER
    };

    bool sort_result_descending = true;

    size_t num_batches;
    size_t num_boxes;
    size_t num_classes;

    ngraph::op::util::NmsBase::SortResultType m_sort_result_type;
    bool m_sort_result_across_batch;
    ngraph::element::Type m_output_type;
    float m_score_threshold;
    int m_nms_top_k;
    int m_keep_top_k;
    int m_background_class;
    ngraph::op::v8::MatrixNms::DecayFunction m_decay_function;
    float m_gaussian_sigma;
    float m_post_threshold;


    SizeVector outputShape_SELECTED_OUTPUTS;
    SizeVector outputShape_SELECTED_INDICES;
    SizeVector outputShape_VALID_OUTPUTS;

    std::string errorPrefix;
    const std::string inType = "input", outType = "output";

    void checkPrecision(const Precision prec, const std::vector<Precision> precList, const std::string name, const std::string type);
    void checkOutput(const SizeVector& dims, const std::vector<Precision> precList, const std::string name, const size_t port);
};

}  // namespace MKLDNNPlugin
