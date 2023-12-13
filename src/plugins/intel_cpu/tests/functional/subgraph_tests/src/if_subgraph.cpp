// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

using IfSubgraphTestParams = std::tuple<bool,          // condition
                                        bool,          // has inplaced-input
                                        bool           // has inplaced-output
                                       >;

class IfSubgraphTest : public testing::WithParamInterface<IfSubgraphTestParams>,
                                        virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<IfSubgraphTestParams> obj) {
        bool condition, hasInplacedInput, hasInplacedOutput;
        std::tie(condition, hasInplacedInput, hasInplacedOutput) = obj.param;

        std::ostringstream result;
        result << "condition=" << condition << "_";
        result << "hasInplacedInput=" << hasInplacedInput << "_";
        result << "hasInplacedOutput=" << hasInplacedOutput;
        return result.str();
    }

    IfSubgraphTest() {
        targetDevice = ov::test::utils::DEVICE_CPU;
        rel_threshold = 1e-4;
    }

    std::shared_ptr<ov::Model> get_then_body() {
        auto Xt = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[0]);
        Xt->set_friendly_name("then/xa");

        auto add_op = std::make_shared<ov::op::v1::Add>(Xt, op::v0::Constant::create(inType, {1, 8}, {1.0f}));
        add_op->set_friendly_name("then/add_op");

        auto then_op_result = std::make_shared<ov::op::v0::Result>(add_op);
        then_op_result->set_friendly_name("then_op_result");

        auto then_body = std::make_shared<ov::Model>(ov::OutputVector{then_op_result}, ov::ParameterVector{Xt});
        return then_body;
    }

    std::shared_ptr<ov::Model> get_else_body() {
        auto Xe = std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes[1]);
        Xe->set_friendly_name("else/kvcache");

        auto else_op_result = std::make_shared<ov::op::v0::Result>(Xe);
        else_op_result->set_friendly_name("else_op_result");

        auto else_body = std::make_shared<ov::Model>(ov::OutputVector{else_op_result}, ov::ParameterVector{Xe});
        return else_body;
    }

    std::shared_ptr<ov::op::v8::If> create_if_node(bool condition, std::shared_ptr<ov::op::v0::Parameter> X, std::shared_ptr<ov::op::v0::Parameter> Y) {
        auto cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, condition);
        cond->set_friendly_name("cond");

        auto if_op = std::make_shared<ov::op::v8::If>(cond);
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
        if_op->set_input(X, then_p[0], nullptr);
        if_op->set_input(Y, nullptr, else_p[0]);
        if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);

        return if_op;
    }

    void SetUp() override {
        std::vector<InputShape> inputShapes({
                                            // xa
                                            {{-1, -1}, {{1, 8}}},
                                            // kvache
                                            {{-1, 8}, {{1, 8}}},
                                            });
        bool condition, hasInplacedInput, hasInplacedOutput;
        std::tie(condition, hasInplacedInput, hasInplacedOutput) = this->GetParam();

        init_input_shapes(inputShapes);
        ov::ParameterVector input_params;
        for (auto&& shape : inputDynamicShapes) {
            input_params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }

        auto if_node = create_if_node(condition, input_params[0], input_params[1]);

        auto result0 = std::make_shared<ov::op::v0::Result>(if_node);
        result0->set_friendly_name("result0");

        auto add_op = std::make_shared<ov::op::v1::Add>(if_node, op::v0::Constant::create(inType, {1, 8}, {1.0f}));
        add_op->set_friendly_name("add");
        auto result1 = std::make_shared<ov::op::v0::Result>(add_op);
        result1->set_friendly_name("result1");

        function = std::make_shared<ov::Model>(ov::NodeVector{result0, result1}, input_params, "IfSubgraphPattern");
        ov::pass::Serialize serializer("IfSubgraphPattern.xml", "IfSubgraphPattern.bin");
        serializer.run_on_model(function);
    }

private:
        const ov::element::Type inType = ov::element::f32;
};

TEST_P(IfSubgraphTest, CompareWithRefs) {
    run();
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_IfSubgraphTest, IfSubgraphTest,
                        ::testing::Combine(::testing::Values(true, false),   // condition
                                           ::testing::Values(true, false),
                                           ::testing::Values(true, false)),
                        IfSubgraphTest::getTestCaseName);
} // namespace
}  // namespace test
}  // namespace ov