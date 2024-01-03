// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_subgraph.hpp>

using namespace ov::test;

// This test case is for stateful model with an init_state graph,
// and Assign is straight from ReadValue.
// What's more, during each iteration (except the first iteration),
// the state should keep unchangeds untill it's resetted.
//
//          ┌─────────┐
//          │ Param1  │
//          └────┬────┘
//            ┌──┴──┐
//            │ Add │
//            └──┬──┘
//         ┌─────┴─────┐
//         | ReadValue │.........
//         └───────────┘        .
//           /      \           .
//          /        \          .
//      ┌──┴──┐  ┌────┴───┐     .
//      │ Add │  │ Assign │......
//      └──┬──┘  └────────┘
//         |
//    ┌────┴───┐
//    │ Result │
//    └────────┘

class InitGraphStatefulModel : public SubgraphBaseTest {
public:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        ov::element::Type netPrc = ElementType::f32;

        const ov::Shape inpShape = {1, 1};
        const InputShape input_shape = {{-1, 1}, {{1, 1}, {2, 1}}};
        init_input_shapes({input_shape});

        auto arg = std::make_shared<ov::op::v0::Parameter>(netPrc, inputDynamicShapes.front());

        // The ReadValue/Assign operations must be used in pairs in the model.
        // For each such a pair, its own variable object must be created.
        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputDynamicShapes.front(), netPrc, variable_name});

        // Creating ov::Model
        auto read = std::make_shared<ov::op::v6::ReadValue>(arg, variable);
        std::vector<std::shared_ptr<ov::Node>> args = {arg, read};
        auto add = std::make_shared<ov::op::v1::Add>(arg, read);
        constexpr int concat_axis = 0;
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{arg, add}, concat_axis);
        auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
        auto res = std::make_shared<ov::op::v0::Result>(concat);
        function = std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), ov::ParameterVector({arg}));
    }
};

TEST_F(InitGraphStatefulModel, smoke_StatefulInitGraph) {
    run();
}