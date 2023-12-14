// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

/*This test runs the following subgraph:

            const     param0    param1
          (condition)  (xa)    (kvcache)
               \        |        /
                \       |       /
                 \      |      /
   |-------------------------------------------|
   |           port:xa     port:kvcache        |
   |                                           |
   |                    If                     |
   |                                           |
   |  |----------------|                       |   
   |        param            |--------------|  |
          (from xa)               param
              |               (from kvcache)
              |                     |
         inplacedInput              |
              |                     |
              |                     |
             Add                    |
              |                  result
              |                (to kvcache)
           result            |--------------|
        (to kvcache)     
   |  |-----------------|                      |
   |                                           |
   |                port: kvcahe               |
   |-------------------------------------------|
               /                  \
              /                    \
          result0                   \
          (kvcache)                 |
                                   Add
                                    |
                                  Result1
  
  The main purpose of this test is checking zero copy of portmaps within If node.
  The pattern is similar to the If node of openai-whisper model.
*/

namespace ov {
namespace test {

using IfSubgraphTestParams = std::tuple<ElementType,
                                       std::vector<InputShape>
                                       >;

class IfSubgraphTest : public testing::WithParamInterface<IfSubgraphTestParams>,
                                        virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<IfSubgraphTestParams> obj) {
        ElementType inType;
        std::vector<InputShape> inputShapes;
        std::tie(inType, inputShapes) = obj.param;

        std::ostringstream result;
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
        result << "Prc=" << inType;
        return result.str();
    }

    IfSubgraphTest() {
        targetDevice = ov::test::utils::DEVICE_CPU;
        rel_threshold = 0.01f;
    }

    std::shared_ptr<ov::Model> get_then_body() {
        auto Xt = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
        Xt->set_friendly_name("then/xa");

        // k
        ov::Shape fcWeightsShape{8, 8};
        auto tensor = ov::test::utils::create_and_fill_tensor(inType, fcWeightsShape);
        auto fc1secondInput = std::make_shared<ov::op::v0::Constant>(tensor);
        const auto fc1 = std::make_shared<ov::op::v0::MatMul>(Xt, fc1secondInput, false, true);
        fc1->set_friendly_name("then/fc1");

        auto fc1_bias = std::make_shared<ov::op::v1::Add>(fc1, op::v0::Constant::create(inType, {8}, {1.0f}));
        fc1_bias->set_friendly_name("then/fc1/bias");

        auto then_result_k = std::make_shared<ov::op::v0::Result>(fc1_bias);
        then_result_k->set_friendly_name("then/result/k");

        // v
        ov::Shape fcWeightsShape_v{8, 8};
        auto tensor_v = ov::test::utils::create_and_fill_tensor(inType, fcWeightsShape_v);
        auto fc2secondInput = std::make_shared<ov::op::v0::Constant>(tensor_v);
        const auto fc2 = std::make_shared<ov::op::v0::MatMul>(Xt, fc2secondInput, false, true);
        fc2->set_friendly_name("then/fc2");

        auto then_result_v = std::make_shared<ov::op::v0::Result>(fc2);
        then_result_v->set_friendly_name("then/result/v");

        auto then_body = std::make_shared<ov::Model>(ov::OutputVector{then_result_k, then_result_v}, ov::ParameterVector{Xt});
        return then_body;
    }

    std::shared_ptr<ov::Model> get_else_body() {
        auto Ke = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);
        Ke->set_friendly_name("else/kcache");
        auto else_result_k = std::make_shared<ov::op::v0::Result>(Ke);
        else_result_k->set_friendly_name("else/result/k");

        auto Ve = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);
        Ve->set_friendly_name("else/vcache");
        auto else_result_v = std::make_shared<ov::op::v0::Result>(Ve);
        else_result_v->set_friendly_name("else/result/v");

        auto else_body = std::make_shared<ov::Model>(ov::OutputVector{else_result_k, else_result_v}, ov::ParameterVector{Ke, Ve});
        return else_body;
    }

    std::shared_ptr<ov::op::v8::If> create_if_node(ov::ParameterVector input_params) {
        auto shapeof_k = std::make_shared<ov::op::v0::ShapeOf>(input_params[1]);
        auto indicesNode = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({}), {1});
        auto axisNode = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({}), {0});
        auto gather = std::make_shared<ov::op::v1::Gather>(shapeof_k, indicesNode, axisNode);
        auto const_zero = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({1}), {0});
        auto condition = std::make_shared<ov::op::v1::Equal>(gather, const_zero);
        condition->set_friendly_name("cond");

        auto if_op = std::make_shared<ov::op::v8::If>(condition);
        if_op->set_friendly_name("if_op");
        const auto& then_body = get_then_body();
        const auto& else_body = get_else_body();

        ov::pass::Serialize serializer1("then_body.xml", "then_body.bin");
        serializer1.run_on_model(then_body);
        ov::pass::Serialize serializer2("else_body.xml", "else_body.bin");
        serializer2.run_on_model(else_body);

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto then_p = then_body->get_parameters();
        auto else_p = else_body->get_parameters();
        if_op->set_input(input_params[0], then_p[0], nullptr);
        if_op->set_input(input_params[1], nullptr, else_p[0]);
        if_op->set_input(input_params[2], nullptr, else_p[1]);
        if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
        if_op->set_output(then_body->get_results()[1], else_body->get_results()[1]);

        return if_op;
    }

    void SetUp() override {
        ElementType inType;
        std::vector<InputShape> inputShapes;
        std::tie(inType, inputShapes) = this->GetParam();

        init_input_shapes(inputShapes);
        ov::ParameterVector input_params;
        for (auto&& shape : inputDynamicShapes) { // xa, k
            input_params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        input_params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1])); // v

        auto if_node = create_if_node(input_params);

        // result kvcache
        auto result0_k = std::make_shared<ov::op::v0::Result>(if_node->output(0));
        result0_k->set_friendly_name("k/result0");
        auto result0_v = std::make_shared<ov::op::v0::Result>(if_node->output(1));
        result0_v->set_friendly_name("v/result0");

        // mha
        auto add_op = std::make_shared<ov::op::v1::Add>(if_node->output(0), if_node->output(1));
        add_op->set_friendly_name("add_kv");
        auto result1 = std::make_shared<ov::op::v0::Result>(add_op);
        result1->set_friendly_name("k/result1");

        function = std::make_shared<ov::Model>(ov::NodeVector{result0_k, result0_v, result1}, input_params, "IfSubgraphPattern");
        ov::pass::Serialize serializer("IfSubgraphPattern.xml", "IfSubgraphPattern.bin");
        serializer.run_on_model(function);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        std::vector<ov::Shape> shapes(3);
        shapes[0] = targetInputStaticShapes[0]; // xa
        shapes[1] = targetInputStaticShapes[1]; // k
        shapes[2] = targetInputStaticShapes[1]; // v
        SubgraphBaseTest::generate_inputs(shapes);
    }
    // void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
    //     inputs.clear();
    //     const auto& funcInputs = function->inputs();

    //     // xa
    //     const std::vector<float> xa(ov::shape_size(targetInputStaticShapes[0]), 0.f);
    //     auto tensor = ov::test::utils::create_tensor<float>(funcInputs[0].get_element_type(), targetInputStaticShapes[0], xa, xa.size());
    //     inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});

    //     // k, v
    //     const std::vector<float> kv(ov::shape_size(targetInputStaticShapes[1]), 0.f);
    //     auto tensor_kv = ov::test::utils::create_tensor<float>(funcInputs[1].get_element_type(), targetInputStaticShapes[1], kv, kv.size());
    //     inputs.insert({funcInputs[1].get_node_shared_ptr(), tensor_kv});
    //     inputs.insert({funcInputs[2].get_node_shared_ptr(), tensor_kv});
    // }
private:
        const ov::element::Type inType = ov::element::f32;
};

TEST_P(IfSubgraphTest, CompareWithRefs) {
    run();
}

namespace {
const std::vector<std::vector<InputShape>> inputShapes = {
    // dynamic batch
    {
        // xa B,L1,HS
        {{-1, -1, -1}, {{1, 10, 8}, {1, 10, 8}, {1, 10, 8}}},
        // kvcache B,L0,HS
        {{-1, -1, 8}, {{1, 0, 8}, {1, 10, 8}, {1, 10, 8}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_IfSubgraphTest, IfSubgraphTest,
                        ::testing::Combine(::testing::Values(ElementType::f32),
                                           ::testing::ValuesIn(inputShapes)),
                        IfSubgraphTest::getTestCaseName);
} // namespace
}  // namespace test
}  // namespace ov