// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/node_builders/eltwise.hpp"
#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "ov_api_conformance_helpers.hpp"
#include "common_test_utils/node_builders/constant.hpp"

namespace {
using namespace ov::test::behavior;
using namespace ov::test::conformance;

std::shared_ptr<ov::Model> ovGetFunction1() {
    const std::vector<size_t> inputShape = {1, 4, 20, 20};
    const ov::element::Type_t ngPrc = ov::element::Type_t::f32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});

    auto in2add = ov::test::utils::make_constant(ngPrc, {1, 4, 1, 1});
    auto add = ov::test::utils::make_eltwise(params[0], in2add, ov::test::utils::EltwiseTypes::ADD);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(add->output(0));
    relu1->get_output_tensor(0).set_names({"relu1"});
    auto relu2 = std::make_shared<ov::op::v0::Relu>(add->output(0));
    relu2->get_output_tensor(0).set_names({"relu2"});

    ov::NodeVector results{relu1, relu2};
    return std::make_shared<ov::Model>(results, params, "AddTwoOutputEdges");
}

std::shared_ptr<ov::Model> ovGetFunction2() {
    const std::vector<size_t> inputShape = {1, 4, 20, 20};
    const ov::element::Type_t ngPrc = ov::element::Type_t::f32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    params.front()->set_friendly_name("Param_1");
    params.front()->get_output_tensor(0).set_names({"input_tensor"});
    auto splitAxisOp = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto split = std::make_shared<ov::op::v1::Split>(params[0], splitAxisOp, 2);

    auto in2add = ov::test::utils::make_constant(ngPrc, {1, 2, 1, 1});
    auto add = ov::test::utils::make_eltwise(split->output(0), in2add, ov::test::utils::EltwiseTypes::ADD);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(add);

    auto in2mult = ov::test::utils::make_constant(ngPrc, {1, 2, 1, 1});
    auto mult = ov::test::utils::make_eltwise(split->output(1), in2mult, ov::test::utils::EltwiseTypes::MULTIPLY);
    auto relu2 = std::make_shared<ov::op::v0::Relu>(mult);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{relu1->output(0), relu2->output(0)}, 3);
    concat->get_output_tensor(0).set_names({"concat"});

    return std::make_shared<ov::Model>(concat, params, "SplitAddConcat");
}

INSTANTIATE_TEST_SUITE_P(ov_infer_request_1, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(ovGetFunction1()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 4, 20, 20}},
                                    {{2, 4, 20, 20}, {2, 4, 20, 20}}}),
                                ::testing::Values(ov::test::utils::target_device),
                                ::testing::Values(ov::AnyMap({}))),
                        OVInferRequestDynamicTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ov_infer_request_2, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(ovGetFunction2()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                    {{1, 4, 20, 20}, {1, 2, 20, 40}},
                                    {{2, 4, 20, 20}, {2, 2, 20, 40}}}),
                                ::testing::Values(ov::test::utils::target_device),
                                ::testing::Values(ov::AnyMap({}))),
                        OVInferRequestDynamicTests::getTestCaseName);
}  // namespace
