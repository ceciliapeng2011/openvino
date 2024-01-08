// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_subgraph.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

// This test case is for stateful model with an init_state graph,
// and Assign is straight from ReadValue.
// What's more, during each iteration (except the first iteration),
// the state should keep unchanged untill it's resetted.
//
//                      ┌─────────┐
//                      │ Param0  │
//                      └────┬────┘
//                           |
//                        ┌──┴──┐
//                        │ Add │
//                        └──┬──┘
//                           |
//      ┌─────────┐    ┌─────┴─────┐
//      │ Param1  │    | ReadValue │...........
//      └────┬────┘    └─────┬─────┘          .
//           |            /     \             .
//           \           /       \            .
//            \         /         \           .
//             \       /           \          .
//             ┌───┴───┐       ┌────┴───┐     .
//             │  Add  │       │ Assign │......
//             └───┬───┘       └────────┘
//                 |
//            ┌────┴───┐
//            │ Result │
//            └────────┘

class InitGraphStatefulModel : virtual public ov::test::SubgraphBaseTest,
                               public CPUTestsBase {
public:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        ov::element::Type netPrc = ElementType::f32;

        const std::vector<InputShape> input_shapes = {
            // param0, to simulate cross-attention kv-cache of Whisper model. State should
            // keep the same value of the first iteration no matter how param0 changes.
            {{1, -1}, {{1, 3}, {1, 3}, {1, 1}, {1, 1}}}, // (B, L)
            // param1, to simulate input_ids for first-token, second-token, etc.
            {{1, -1}, {{1, 3}, {1, 3}, {1, 1}, {1, 1}}}, // (B, L)
         };
        init_input_shapes(input_shapes);

        auto arg_0 = std::make_shared<ov::op::v0::Parameter>(netPrc, inputDynamicShapes.front());
        arg_0->set_friendly_name("xa");
    
        auto arg_1 = std::make_shared<ov::op::v0::Parameter>(netPrc, inputDynamicShapes.back());
        arg_1->set_friendly_name("input_ids");

        // init_graph
        auto add_0 = std::make_shared<ov::op::v1::Add>(arg_0, ov::op::v0::Constant::create(netPrc, {1}, {1.0f}));
        add_0->set_friendly_name("init_graph/add");

        // The ReadValue/Assign operations must be used in pairs in the model.
        // For each such a pair, its own variable object must be created.
        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputDynamicShapes.front(), netPrc, variable_name});

        // Creating ov::Model
        auto read = std::make_shared<ov::op::v6::ReadValue>(add_0, variable);
        auto add_1 = std::make_shared<ov::op::v1::Add>(arg_1, read);
        add_1->set_friendly_name("add_1");
        auto assign = std::make_shared<ov::op::v6::Assign>(read, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add_1);
        function = std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), ov::ParameterVector({arg_0, arg_1}));

        ov::pass::Serialize serializer("InitGraphStatefulModel.xml", "InitGraphStatefulModel.bin");
        serializer.run_on_model(function);
    }

    std::vector<ov::Tensor> calculate_refs() override {
        for (const auto& param : functionRefs->get_parameters()) {
            inferRequestRef.set_tensor(param->get_default_output(), inputs.at(matched_parameters[param]));
        }
        inferRequestRef.infer();

        auto outputs = std::vector<ov::Tensor>{};
        for (const auto& output : functionRefs->outputs()) {
            outputs.push_back(inferRequestRef.get_tensor(output));
        }

        return outputs;
    }

    std::vector<ov::Tensor> get_plugin_outputs() override {
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
        auto outputs = std::vector<ov::Tensor>{};
        for (const auto& output : function->outputs()) {
            outputs.push_back(inferRequest.get_tensor(output));
        }
        return outputs;
    }

    void reset() {
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }

        for (auto&& state : inferRequestRef.query_state()) {
            state.reset();
        }
    }

    void prepare() {
        compile_model();

        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);

        // ref
        functionRefs = function->clone();
    
        matched_parameters.clear();
        const auto& ref_params = functionRefs->get_parameters();
        const auto& params = function->get_parameters();
        for (size_t in_idx = 0; in_idx < params.size(); ++in_idx) {
            matched_parameters.insert({ ref_params[in_idx], params[in_idx] });
        }

        auto compiledModelRef = core->compile_model(functionRefs, ov::test::utils::DEVICE_TEMPLATE);
        inferRequestRef = compiledModelRef.create_infer_request();
    }

    void run() override {
        prepare();

        // iterating with state reset
        for (auto iters = 0; iters < 3; iters++) {
            std::cout << "========= iters" << iters << std::endl;
            for (const auto& targetStaticShapeVec : targetStaticShapes) {
                std::cout << "========= targetStaticShapeVec" << targetStaticShapeVec.front() << std::endl;
                generate_inputs(targetStaticShapeVec);

                validate();
            }

            std::cout << "========= reset =============" << std::endl << std::endl;
            reset();
        }
    }

    ov::InferRequest inferRequestRef;    
};

TEST_F(InitGraphStatefulModel, smoke_StatefulInitGraph) {
    run();
}