// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <sstream>
#include "gtest/gtest.h"

#include "default_opset.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "onnx_editor/editor.hpp"
#include "onnx_import/onnx.hpp"
#include "util/test_control.hpp"

#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

template <typename T>
class ElemTypesTests : public ::testing::Test
{
};
TYPED_TEST_CASE_P(ElemTypesTests);

TYPED_TEST_P(ElemTypesTests, onnx_test_add_abc_set_precission)
{
    using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});
    using DataType = TypeParam;
    const element::Type ng_type = element::from<DataType>();

    onnx_editor::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc_3d.prototxt")};

    editor.set_input_types({{"A", ng_type}, {"B", ng_type}, {"C", ng_type}});

    std::istringstream model_stream(editor.model_string());
    const auto function = onnx_import::import_onnx_model(model_stream);
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<DataType>(std::vector<DataType>{1, 2, 3});
    test_case.add_input<DataType>(std::vector<DataType>{4, 5, 6});
    test_case.add_input<DataType>(std::vector<DataType>{7, 8, 9});
    test_case.add_expected_output<DataType>(Shape{3}, std::vector<DataType>{12, 15, 18});
    test_case.run();
}

TYPED_TEST_P(ElemTypesTests, onnx_test_split_multioutput_set_precission)
{
    using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});
    using DataType = TypeParam;
    const element::Type ng_type = element::from<DataType>();

    onnx_editor::ONNXModelEditor editor{
        file_util::path_join(SERIALIZED_ZOO, "onnx/split_equal_parts_default.prototxt")};

    editor.set_input_types({{"input", ng_type}});

    std::istringstream model_stream(editor.model_string());
    const auto function = onnx_import::import_onnx_model(model_stream);
    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<DataType>(std::vector<DataType>{1, 2, 3, 4, 5, 6});
    test_case.add_expected_output<DataType>(Shape{2}, std::vector<DataType>{1, 2});
    test_case.add_expected_output<DataType>(Shape{2}, std::vector<DataType>{3, 4});
    test_case.add_expected_output<DataType>(Shape{2}, std::vector<DataType>{5, 6});
    test_case.run();
}

REGISTER_TYPED_TEST_CASE_P(ElemTypesTests,
                           onnx_test_add_abc_set_precission,
                           onnx_test_split_multioutput_set_precission);
typedef ::testing::Types<int8_t, int16_t, int32_t, uint8_t, float> ElemTypes;
INSTANTIATE_TYPED_TEST_CASE_P(${BACKEND_NAME}, ElemTypesTests, ElemTypes);
