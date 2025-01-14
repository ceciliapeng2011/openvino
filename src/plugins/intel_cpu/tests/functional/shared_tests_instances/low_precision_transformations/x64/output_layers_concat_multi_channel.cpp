// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/output_layers_concat_multi_channel.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::pass::low_precision;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParams(),
    // LayerTestsUtils::LayerTransformationParamsFactory::createParamsI8I8(),
    // LayerTestsUtils::LayerTransformationParamsFactory::createParamsU8I8()
};

// TODO: issue #41231: enable previous LPT version tests
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_LPT, OutputLayersConcatMultiChannel,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::Shape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues)),
    OutputLayersConcatMultiChannel::getTestCaseName);
}  // namespace
