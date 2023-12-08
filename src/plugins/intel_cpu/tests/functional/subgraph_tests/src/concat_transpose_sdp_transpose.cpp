// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset13.hpp>
#include <transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"

using namespace ov::test;
using namespace ngraph;
using namespace CPUTestUtils;
using namespace InferenceEngine;

namespace SubgraphTestsDefinitions {

using InputShapeAndTransposeOrder = std::pair<std::vector<InputShape>, std::vector<size_t>>;
using ConcatSDPTransposeTestParams = std::tuple<ElementType,
                                       InputShapeAndTransposeOrder,
                                       std::vector<size_t>,         // post transpose order
                                       bool                         // has ShapeOf
                                       >;
// Subgraph:
/*                              Parameter
 *                                  |
 *       Parameter    ReadValue     |           ReadValue  Parameter
 *           \           /          |               \          /
 *            \         /           |                \        /
 *               Concat         Transpose              Concat
 *                / \               |                 /     \
 *               /   \              |                /       \
 *              /   Transpose       |          Transpose      \
 *             /       \            |            /             \
 *          Assign      ScaledDotProductAttention              Assign
 *                                  |
 *                               Tranpose
 *                                  |
 *                                Reshape
 *                                  |
 *                                 Add
 *                                  |
 *                                Result
 */

class ConcatSDPTransposeTest : public testing::WithParamInterface<ConcatSDPTransposeTestParams>, virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatSDPTransposeTestParams>& obj) {
        ElementType inType;
        InputShapeAndTransposeOrder inputShapeAndOrders;
        std::vector<size_t> postSDPATransposeOrders;
        bool hasShapeof;
        std::tie(inType, inputShapeAndOrders, postSDPATransposeOrders, hasShapeof) = obj.param;
        std::ostringstream result;
        std::vector<InputShape>& inputShapes = inputShapeAndOrders.first;
        std::vector<size_t>& transposeOrder = inputShapeAndOrders.second;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")_";
        }
        result << "Prc=" << inType << "_";
        result << "HasShapeOf=" << hasShapeof;
        result << "TransposeOrder=";
        result << "(";
        for (const auto& itr : transposeOrder) {
            result << itr << ",";
        }
        result << ")";
        result << "postSDPATransposeOrders=";
        result << "(";
        for (const auto& itr : postSDPATransposeOrders) {
            result << itr << ",";
        }
        result << ")";

        return result.str();
    }

    void SetUp() override {
        ElementType inType;
        InputShapeAndTransposeOrder inputShapeAndOrders;
        std::vector<size_t> postSDPATransposeOrders;
        bool hasShapeOf;
        std::tie(inType, inputShapeAndOrders, postSDPATransposeOrders, hasShapeOf) = this->GetParam();
        std::vector<InputShape>& inputShapes = inputShapeAndOrders.first;
        std::vector<size_t>& transposeOrder = inputShapeAndOrders.second;
        targetDevice = ov::test::utils::DEVICE_CPU;
        rel_threshold = 1e-2f;
        if (inType == ElementType::bf16) {
            configuration.insert({"ENFORCE_BF16", "YES"});
            rel_threshold = 0.01f;
        }
        init_input_shapes(inputShapes);
        ov::ParameterVector inputParams;
        // q,k,v
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]));
        inputParams[0]->set_friendly_name("q");
        inputParams[1]->set_friendly_name("k");
        inputParams[2]->set_friendly_name("v");
        // pastkv init_cost
        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]));
        auto var_k = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputDynamicShapes[1], inType, "pastk"});
        auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_k);
        pastk->set_friendly_name("pastk_r");
        auto var_v = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputDynamicShapes[1], inType, "pastv"});
        auto pastv = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_v);
        pastv->set_friendly_name("pastv_r");
        std::shared_ptr<Node> pastk_shapeof, pastv_shapeof;
        if (hasShapeOf) {
            pastk_shapeof = std::make_shared<ov::op::v0::ShapeOf>(pastk);
            pastv_shapeof = std::make_shared<ov::op::v0::ShapeOf>(pastv);
        }

        // pre SDPA transpose
        auto preOrder = op::v0::Constant::create(ov::element::i32, {4}, transposeOrder);  // TODO dtype
        auto transposeQ = std::make_shared<ov::op::v1::Transpose>(inputParams[0], preOrder);

        auto concat_axis = transposeOrder[2];
        auto concatK = std::make_shared<ov::op::v0::Concat>(OutputVector{pastk, inputParams[1]}, concat_axis);
        auto concatV = std::make_shared<ov::op::v0::Concat>(OutputVector{pastv, inputParams[2]}, concat_axis);
        auto transposeK = std::make_shared<ov::op::v1::Transpose>(concatK, preOrder);
        auto transposeV = std::make_shared<ov::op::v1::Transpose>(concatV, preOrder);

        auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(transposeQ, transposeK, transposeV, false);
        sdp->set_friendly_name("mha");

        // post SDPA transpose + reshape
        auto get_reshape_order = [] (const ov::PartialShape& qkv_shape, const std::vector<size_t>& transposeOrder, const std::vector<size_t>& postSDPATransposeOrders) -> std::vector<size_t> {
            assert(transposeOrder.size()==4);
            assert(postSDPATransposeOrders.size()==4);
            auto H = qkv_shape[transposeOrder[1]].get_length();
            auto S = qkv_shape[transposeOrder[3]].get_length();

            auto indexH = std::distance(postSDPATransposeOrders.begin(), std::find(postSDPATransposeOrders.begin(), postSDPATransposeOrders.end(), 1));
            auto indexS = std::distance(postSDPATransposeOrders.begin(), std::find(postSDPATransposeOrders.begin(), postSDPATransposeOrders.end(), 3));
            assert(std::abs(indexH - indexS) == 1); // HxS

            std::vector<size_t> reshape_order(3, 0);
            reshape_order[std::min(indexH, indexS)] = static_cast<size_t>(H*S);
            
            return reshape_order;
        };
        const auto reshapeOrder = get_reshape_order(inputDynamicShapes[0], transposeOrder, postSDPATransposeOrders);
        std::cout << "============ qkv_shape " << inputDynamicShapes[0] << " transpose " << ov::Shape(transposeOrder) << "postSDPATransposeOrders " << ov::Shape(postSDPATransposeOrders) << " -> reshape " << ov::Shape(reshapeOrder) << std::endl;

        auto postOrder = op::v0::Constant::create(ov::element::i32, {4}, postSDPATransposeOrders);  // BHLS -> BLHS
        auto transposeSDP = std::make_shared<ov::op::v1::Transpose>(sdp, postOrder);

        auto constReshape = op::v0::Constant::create(ov::element::i32, {3}, reshapeOrder);
        auto reshapeSDP = std::make_shared<ov::op::v1::Reshape>(transposeSDP, constReshape, true); // BLHS -> B,L,HxS        

        // auto add = std::make_shared<op::v1::Add>(reshapeSDP, op::v0::Constant::create(inType, {1}, {1.0f}));
        auto pastk_assign = std::make_shared<op::v6::Assign>(concatK, var_k);
        auto pastv_assign = std::make_shared<op::v6::Assign>(concatV, var_v);
        pastk_assign->set_friendly_name("pastk_w");
        pastv_assign->set_friendly_name("pastv_w");

        ResultVector results{std::make_shared<ov::op::v0::Result>(reshapeSDP)};
        if (hasShapeOf) {
            results.push_back(std::make_shared<ov::op::v0::Result>(pastk_shapeof));
            results.push_back(std::make_shared<ov::op::v0::Result>(pastv_shapeof));
        }
        SinkVector sinks{pastk_assign, pastv_assign};
        function = std::make_shared<Function>(results, sinks, inputParams, "ConcatTranposeSDP");
        targetDevice = ov::test::utils::DEVICE_CPU;

        {
            ov::pass::Serialize serializer("ConcatTranposeSDP.xml", "ConcatTranposeSDP.bin");
            serializer.run_on_model(function);
        }

        functionRefs = function->clone();
        pass::Manager manager;
        // decompose ScaledDotProductAttention
        manager.register_pass<ov::pass::ScaledDotProductAttentionDecomposition>();
        manager.run_passes(functionRefs);
        {
            ov::pass::Serialize serializer("ConcatTranposeSDP_ref.xml", "ConcatTranposeSDP_ref.bin");
            serializer.run_on_model(function);
        }
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        std::vector<ov::Shape> shapes(4);
        shapes[0] = targetInputStaticShapes[0];
        shapes[1] = targetInputStaticShapes[0];
        shapes[2] = targetInputStaticShapes[0];
        shapes[3] = targetInputStaticShapes[1];
        SubgraphBaseTest::generate_inputs(shapes);
    }
    template<typename IT, typename T>
    void strided_iota(IT first, size_t n, T value, T stride) {
        for (size_t i = 0; i < n; i++) {
            *first++ = value;
            value += stride;
        }
    }
    void generate(int idx, const std::vector<ov::Shape>& targetInputStaticShapes) {
        inputs.clear();
        auto create_input = [this] (std::shared_ptr<op::v0::Parameter> param, ov::Shape shape, float val) {
            if (param->get_element_type() == element::f32) {
                ov::Tensor t{ov::element::f32, shape};
                strided_iota(static_cast<float*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            } else {
                ov::Tensor t{ov::element::bf16, shape};
                strided_iota(static_cast<ov::bfloat16*>(t.data()), t.get_size(), val, 0.1f);
                inputs.insert({param, t});
            }
        };
        // q, k, v
        create_input(function->get_parameters()[0], targetInputStaticShapes[0], idx + 1.0f);
        create_input(function->get_parameters()[1], targetInputStaticShapes[0], idx + 2.0f);
        create_input(function->get_parameters()[2], targetInputStaticShapes[0], idx + 3.0f);
        create_input(function->get_parameters()[3], targetInputStaticShapes[1], idx + 4.0f);
    }
    void prepare() {
        compile_model();
        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);
    }
    void reset() {
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }
    }
    std::vector<ov::Tensor> run_test(std::shared_ptr<ov::Model> model) {
        function = model;
        prepare();
        std::vector<ov::Tensor> outputs;
        int idx = 0;
        for (auto&& shapes : targetStaticShapes) {
            generate(idx++, shapes);
            for (const auto& input : inputs) {
                inferRequest.set_tensor(input.first, input.second);
            }
            inferRequest.infer();
            auto outputTensor = inferRequest.get_output_tensor(0);
            ov::Tensor copy{outputTensor.get_element_type(), outputTensor.get_shape()};
            outputTensor.copy_to(copy);
            outputs.push_back(copy);
        }
        reset();

        return outputs;
    }
};

TEST_P(ConcatSDPTransposeTest, CompareWithRefs) {
    auto actualOutputs = run_test(function);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 1);
    CheckNumberOfNodesWithType(compiledModel, "Concatenation", 0);
    CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
    CheckNumberOfNodesWithType(compiledModel, "Transpose", 0);  // TODO
    auto expectedOutputs = run_test(functionRefs);
    CheckNumberOfNodesWithType(compiledModel, "ScaledDotProductAttention", 0);
    for (size_t i = 0; i < actualOutputs.size(); i++) {
        ov::test::utils::compare(expectedOutputs[i], actualOutputs[i], abs_threshold, rel_threshold);
    }
}

namespace {
const std::vector<InputShapeAndTransposeOrder> inputShapeAndReorders = {
    {   // TODO: dynamic batch
        // inputShapes LLama
        {
            // B, H, L1, S
            {{1, 8, -1, 64}, {{1, 8, 10, 64}, {1, 8, 1, 64}, {1, 8, 1, 64}, {1, 8, 20, 64}, {1, 8, 1, 64}}},
            // B, H, L0, S
            {{1, 8, -1, 64}, {{1, 8, 0, 64}, {1, 8, 10, 64}, {1, 8, 11, 64}, {1, 8, 12, 64}, {1, 8, 32, 64}}},
        },
        // transposeOrder
        {0, 1, 2, 3}
    },
    {
        // inputShapes QWen
        {
            // B, L1, H, S
            {{1, -1, 8, 64}, {{1, 10, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {1, 20, 8, 64}, {1, 1, 8, 64}}},
            // B, L0, H, S
            {{1, -1, 8, 64}, {{1, 0, 8, 64}, {1, 10, 8, 64}, {1, 11, 8, 64}, {1, 12, 8, 64}, {1, 32, 8, 64}}},
        },
        // transposeOrder
        {0, 2, 1, 3}
    },
    {
        // inputShapes ChatGLM
        {
            // L1, B, H, S
            {{-1, 1, 8, 64}, {{10, 1, 8, 64}, {1, 1, 8, 64}, {1, 1, 8, 64}, {20, 1, 8, 64}, {1, 1, 8, 64}}},
            // L0, B, H, S
            {{-1, 1, 8, 64}, {{0, 1, 8, 64}, {10, 1, 8, 64}, {11, 1, 8, 64}, {12, 1, 8, 64}, {32, 1, 8, 64}}},
        },
        // transposeOrder
        {1, 2, 0, 3}
    },
};

const std::vector<std::vector<size_t>> postTransposeReorders = {
    {0, 2, 1, 3},       // SDPA BHLS -> BLHS (llama, QWen)
    {2, 0, 1, 3},       // SDPA BHLS -> LBHS (chatglm)
};

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPTransposeTest,
                         ConcatSDPTransposeTest,
                         ::testing::Combine(::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(inputShapeAndReorders),
                                            ::testing::ValuesIn(postTransposeReorders),
                                            ::testing::Values(true, false)),
                         ConcatSDPTransposeTest::getTestCaseName);

}  // namespace
}  // namespace SubgraphTestsDefinitions
