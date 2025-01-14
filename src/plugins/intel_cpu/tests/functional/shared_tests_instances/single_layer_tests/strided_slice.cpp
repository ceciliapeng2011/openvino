// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/strided_slice.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::StridedSliceLayerTest;

namespace {
struct RawParams  {
    std::vector<ov::Shape> input_shape;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    std::vector<int64_t> begin_mask;
    std::vector<int64_t> end_mask;
    std::vector<int64_t> new_axis_mask;
    std::vector<int64_t> shrink_axis_mask;
    std::vector<int64_t> ellipsis_axis_mask;
};

std::vector<RawParams> raw_test_cases = {
        RawParams{ {{ 16 }}, { 4 }, { 12 }, { 1 },
                                    { 0 }, { 0 },  { },  { },  { } },
        RawParams{ {{ 16 }}, { 0 }, { 8 }, { 2 },
                                    { 1 }, { 0 },  { },  { },  { } },
        RawParams{ {{ 128, 1 }}, { 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1 },
                            { 0, 1, 1 }, { 0, 1, 1 },  { 1, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 128, 1 }}, { 0, 0, 0 }, { 0, 0, 0 }, { 1, 1, 1},
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 1, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 2, 3 }}, { 1, 0 }, { 2, 3 }, { 1, 1 },
                            { 0, 0 }, { 0, 0 },  {  },  {  },  {  } },
        RawParams{ {{ 10, 3 }}, { 0, 0 }, { 20, 20 }, { 1, 1 },
                            { 0, 1 }, { 0, 1 },  {  },  {  },  {  } },
        RawParams{ {{ 1, 12, 100 }}, { 0, -1, 0 }, { 0, 0, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 1, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 1, 12, 100 }}, { 0, 9, 0 }, { 0, 11, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 1, 12, 100 }}, { 0, 1, 0 }, { 0, -1, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 2, 12, 100 }}, { 0, 9, 0 }, { 0, 7, 0 }, { -1, -1, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 2, 12, 100 }}, { 0, 7, 0 }, { 0, 9, 0 }, { -1, 1, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 1, 12, 100 }}, { 0, 4, 0 }, { 0, 9, 0 }, { -1, 2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 1, 12, 100 }}, { 0, 4, 0 }, { 0, 10, 0 }, { -1, 2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 1, 12, 100 }}, { 0, 9, 0 }, { 0, 4, 0 }, { -1, -2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 2, 12, 100 }}, { 0, 10, 0 }, { 0, 4, 0 }, { -1, -2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 1, 12, 100 }}, { 0, 11, 0 }, { 0, 0, 0 }, { -1, -2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 1, 12, 100 }}, { 0, -6, 0 }, { 0, -8, 0 }, { -1, -2, -1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 20, 10, 5 }}, { 0, 0, 0 }, { 3, 10, 0 }, { 1, 1, 1 },
                            { 0, 0, 1 }, { 0, 0, 1 },  { 0, 0, 0 },  { 0, 0, 0 },  { 0, 0, 0 } },
        RawParams{ {{ 1, 10, 20 }}, { 0, 0, 2 }, { 0, 0, 1000 }, { 1, 1, 1 },
                            { 1, 1, 0 }, { 1, 1, 0 },  {  },  {  },  {  } },
        RawParams{ {{ 1, 10, 10 }}, { 0, 1, 0 }, { 0, 1000, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  {  },  {  },  {  } },
        RawParams{ {{ 1, 10, 4 }}, { 0, 0, 0 }, { 0, 0, 2 }, { 1, 1, 1 },
                            { 1, 1, 0 }, { 1, 1, 0 },  {  },  {  },  {  } },
        RawParams{ {{ 1, 10, 4 }}, { 0, 0, 2 }, { 0, 0, 1000 }, { 1, 1, 1 },
                            { 1, 1, 0 }, { 1, 1, 0 },  {  },  {  },  {  } },
        RawParams{ {{ 1, 10, 2 }}, { 0, 0, 0 }, { 0, 0, 1 }, { 1, 1, 1 },
                            { 1, 1, 0 }, { 1, 1, 0 },  {  },  {  },  {  } },
        RawParams{ {{ 1, 10, 2 }}, { 0, 0, 0 }, { 1000, 0, 0 }, { 1, 1, 1 },
                            { 0, 1, 1 }, { 0, 1, 1 },  {  },  {  },  {  } },
        RawParams{ {{ 1, 10, 2 }}, { 0, 0, 0 }, { 0, 1000, 0 }, { 1, 1, 1 },
                            { 1, 0, 1 }, { 1, 0, 1 },  {  },  {  },  {  } },
        RawParams{ {{ 20, 10, 5 }}, { 0, 3 }, { 0, 4 }, { 1, 1 },
                            { 1, 0 }, { 1, 0 },  {  },  {  },  { 1, 0 } },
        RawParams{ {{ 20, 10, 5 }}, { 0, 0 }, { 0, -1 }, { 1, 1 },
                            { 1, 0 }, { 1, 0 },  {  },  {  },  { 1, 0 } },
        RawParams{ {{ 20, 10, 5 }}, { 0, 0 }, { 0, -1 }, { 1, 1 },
                                    { 1, 0 }, { 1, 0 },  { 0, 0 },  { 0, 0 },  { 0, 0 } },
        RawParams{ {{ 1, 8400, 6 }}, { 0, 2 }, { 0, 4 }, { 1, 1 },
                                    { 0 }, { 0 },  { 0 },  { 0 },  { 1 } },
        RawParams{ {{ 1, 12, 100, 1, 1 }}, { 0, -1, 0, 0 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 },
                            { 1, 0, 1, 0 }, { 1, 0, 1, 0 },  { },  { 0, 1, 0, 1 },  {} },
        RawParams{ {{ 2, 2, 2, 2 }}, { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                            { 1, 1, 1, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        RawParams{ {{ 2, 2, 2, 2 }}, { 0, 0 }, { 2, 2 }, { 1, 1 },
                            { 1, 1 }, { 1, 1 },  {},  {},  {} },
        RawParams{ {{ 2, 2, 3, 3 }}, { 0, -2, -2 }, { 2, -1, -1 }, { 1, 1, 1 },
                            { 1, 0 }, { 1, 0 },  {},  {},  {} },
        RawParams{ {{ 2, 2, 2, 2 }}, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                            { 0, 0, 0, 0}, { 1, 1, 1, 1},  {},  {},  {} },
        RawParams{ {{ 2, 2, 2, 2 }}, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 },
                            { 0, 0, 0, 0}, { 0, 0, 0, 0},  {},  {},  {} },
        RawParams{ {{ 1, 2, 6, 4 }}, { 0, 0, 4, 0 }, { 1, 2, 6, 4 }, { 1, 1, 1, 1 },
                            { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, {}, {}, {} },
        RawParams{ {{ 1, 2, 6, 4 }}, { 0, 0, -3, 0 }, { 1, 2, 6, 4 }, { 1, 1, 1, 1 },
                            { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, {}, {}, {} },
        RawParams{ {{ 1, 2, 6, 4 }}, { 0, 0, 4, 0 }, { 1, 2, 6, 4 }, { 1, 1, 1, 1 },
                            { 1, 1, 0, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        RawParams{ {{ 10, 2, 2, 2 }}, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 2, 1, 1, 1 },
                            { 1, 1, 1, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        RawParams{ {{ 2, 2, 4, 3 }}, { 0, 0, 0, 0 }, { 2, 2, 4, 3 }, { 1, 1, 2, 1 },
                            { 1, 1, 1, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        RawParams{ {{ 2, 2, 4, 2 }}, { 1, 0, 0, 1 }, { 2, 2, 4, 2 }, { 1, 1, 2, 1 },
                            { 0, 1, 1, 0}, { 1, 1, 0, 0},  {},  {},  {} },
        RawParams{ {{ 1, 2, 4, 2 }}, { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                            { 1, 1, 1, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        RawParams{ {{ 2, 2, 4, 2 }}, { 1, 0, 0, 0 }, { 1, 2, 4, 2 }, { 1, 1, -2, -1 },
                            { 0, 1, 1, 1}, { 1, 1, 1, 1},  {},  {},  {} },
        RawParams{ {{ 2, 3, 4, 5, 6 }}, { 0, 1, 0, 0, 0 }, { 2, 3, 4, 5, 6 }, { 1, 1, 1, 1, 1 },
                            { 1, 0, 1, 1, 1}, { 1, 0, 1, 1, 1 },  {},  { 0, 1, 0, 0, 0 },  {} },
        RawParams{ {{ 2, 3, 4, 5, 6 }}, { 0, 0, 3, 0, 0 }, { 2, 3, 4, 3, 6 }, { 1, 1, 1, 1, 1 },
                            { 1, 1, 0, 1, 1}, { 1, 1, 0, 0, 1 },  {},  { 0, 0, 1, 0, 0 },  {} },
        RawParams{ {{ 2, 3, 4, 5, 6 }}, { 0, 0, 0, 0, 3 }, { 1, 3, 4, 5, 6 }, { 1, 1, 1, 1, 1 },
                            { 0, 1, 1, 1, 0}, { 0, 1, 1, 1, 0 },  {},  { 1, 0, 0, 0, 1 },  {} },
        RawParams{ {{ 2, 3, 4, 5 }}, { 0, 0, 0, 0, 0 }, { 0, 2, 3, 4, 5 }, { 1, 1, 1, 1, 1 },
                            { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 },  { 1, 0, 0, 0, 0 },  {},  {} },
        RawParams{ {{ 2, 3, 4, 5 }}, { 0, 0, 0, 0, 0 }, { 0, 2, 3, 4, 5 }, { 1, 1, 1, 1, 1 },
                            { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 },  { 0, 0, 1, 0, 0 },  {},  {} },
        RawParams{ {{ 10, 12 }}, { -1, 1 }, { -9999, 0 }, { -1, 1 },
                            { 0, 1 }, { 0, 1 },  { 0, 0 },  { 0, 0 },  { 0, 0 } },
        RawParams{ {{ 5, 5, 5, 5 }}, { -1, 0, -1, 0 }, { -50, 0, -60, 0 }, { -1, 1, -1, 1 },
                            { 0, 0, 0, 0 }, { 0, 1, 0, 1 },  { 0, 0, 0, 0 },  { 0, 0, 0, 0 },  { 0, 0, 0, 0 } },
        RawParams{ {{ 1, 2, 4 }}, { 0, 2000, 3, 5 }, { 0, 0, 0, 2 }, { 1, 1, 1, 1 },
                            { 1, 0, 1, 1 }, { 1, 0, 1, 0 },  { 0, 1, 0, 0 },  { },  { } },
        RawParams{ {{ 2, 2, 4, 4 }}, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 2, 0 }, { 1, 1, 1, 1, 1 },
                            { 1, 1, 1, 0, 1 }, { 1, 1, 1, 0, 1 },  { 0, 1, 0, 0, 0 },  { },  { } },
        RawParams{ {{ 2, 2, 2, 4, 4 }}, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 2, 0 }, { 1, 1, 1, 1, 1 },
                            { 1, 1, 1, 0, 1 }, { 1, 1, 1, 0, 1 },  { },  { 0, 1, 0, 0, 0 },  { } },
        RawParams{ {{1, 6400, 3, 85}},
                               {0, 0},
                               {0, 2},
                               {1, 1},
                               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                               {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
};

auto ss_test_cases = [](const std::vector<RawParams>& raw_test_cases) {
        std::vector<ov::test::StridedSliceSpecificParams> cases;
        for (const auto& raw_case : raw_test_cases)
                cases.push_back(ov::test::StridedSliceSpecificParams{
                                ov::test::static_shapes_to_test_representation(raw_case.input_shape),
                                raw_case.begin,
                                raw_case.end,
                                raw_case.strides,
                                raw_case.begin_mask,
                                raw_case.end_mask,
                                raw_case.new_axis_mask,
                                raw_case.shrink_axis_mask,
                                raw_case.ellipsis_axis_mask});
        return cases;
}(raw_test_cases);

INSTANTIATE_TEST_SUITE_P(
        smoke, StridedSliceLayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(ss_test_cases),
            ::testing::Values(ov::element::f32),
            ::testing::Values(ov::test::utils::DEVICE_CPU)),
        StridedSliceLayerTest::getTestCaseName);

}  // namespace
