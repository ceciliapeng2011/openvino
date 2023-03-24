// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset10.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;
using namespace testing;

class AdaptiveMaxPoolV8Test : public TypePropOpTest<op::v8::AdaptiveMaxPool> {};

TEST_F(AdaptiveMaxPoolV8Test, default_ctor) {
    const auto data = make_shared<Parameter>(element::f32, PartialShape{2, 6, 3, 2});
    const auto out_shape = Constant::create<int64_t>(element::i64, Shape{2}, {5, 7});

    const auto op = make_op();
    op->set_arguments(OutputVector{data, out_shape});
    op->set_index_element_type(element::i64);
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_index_element_type(), element::i64);
    EXPECT_EQ(op->get_input_size(), 2);
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Output type", &Output<Node>::get_element_type, element::f32),
                            Property("Indices type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(),
                Each(Property("PartialShape", &Output<Node>::get_partial_shape, PartialShape({2, 6, 5, 7}))));
}

TEST_F(AdaptiveMaxPoolV8Test, shape_infer) {
    const auto data = make_shared<Parameter>(element::f64, Shape{2, 6, 3, 2, 10});
    const auto out_shape = Constant::create<int64_t>(element::i64, Shape{3}, {5, 7, 1});

    const auto op = make_op(data, out_shape);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Output type", &Output<Node>::get_element_type, element::f64),
                            Property("Indices type", &Output<Node>::get_element_type, element::i64)));
    EXPECT_THAT(op->outputs(), Each(Property("Shape", &Output<Node>::get_shape, Shape({2, 6, 5, 7, 1}))));
}

TEST_F(AdaptiveMaxPoolV8Test, i32_indices) {
    auto data_shape = PartialShape{2, 6, 2, 10};
    set_shape_labels(data_shape, 10);

    const auto data = make_shared<Parameter>(element::f64, data_shape);
    const auto out_shape = Constant::create<int32_t>(element::i32, Shape{2}, {7, 1});

    const auto op = make_op(data, out_shape, element::i32);

    EXPECT_THAT(op->outputs(),
                ElementsAre(Property("Output type", &Output<Node>::get_element_type, element::f64),
                            Property("Indices type", &Output<Node>::get_element_type, element::i32)));
    EXPECT_THAT(op->outputs(),
                Each(Property("PartialShape", &Output<Node>::get_partial_shape, PartialShape({2, 6, 7, 1}))));
    EXPECT_THAT(op->outputs(),
                Each(Property(&Output<Node>::get_partial_shape,
                              ResultOf(get_shape_labels, ElementsAre(10, 11, ov::no_label, ov::no_label)))));
}

TEST_F(AdaptiveMaxPoolV8Test, dynamic_batch) {
    PartialShape data_shape{Dimension::dynamic(), 6, 8, 9};
    set_shape_labels(data_shape, 10);

    const auto data = make_shared<Parameter>(element::f32, data_shape);
    const auto out_shape = Constant::create<int64_t>(element::i64, Shape{2}, {9, 9});
    const auto op = make_op(data, out_shape);

    EXPECT_THAT(op->outputs(),
                Each(Property("PartialShape", &Output<Node>::get_partial_shape, PartialShape({-1, 6, 9, 9}))));
    EXPECT_THAT(op->outputs(),
                Each(Property(&Output<Node>::get_partial_shape,
                              ResultOf(get_shape_labels, ElementsAre(10, 11, ov::no_label, ov::no_label)))));
}

TEST_F(AdaptiveMaxPoolV8Test, dynamic_channel) {
    PartialShape data_shape{2, Dimension::dynamic(), {10, 20}, 9};
    set_shape_labels(data_shape, 10);

    const auto data = make_shared<Parameter>(element::f32, data_shape);
    const auto out_shape = Constant::create<int64_t>(element::i64, Shape{2}, {5, 7});
    const auto op = make_op(data, out_shape);

    EXPECT_THAT(op->outputs(),
                Each(Property("PartialShape", &Output<Node>::get_partial_shape, PartialShape({2, -1, 5, 7}))));
    EXPECT_THAT(op->outputs(),
                Each(Property(&Output<Node>::get_partial_shape,
                              ResultOf(get_shape_labels, ElementsAre(10, 11, ov::no_label, ov::no_label)))));
}

TEST_F(AdaptiveMaxPoolV8Test, dynamic_spatial) {
    PartialShape data_shape{2, 6, -1, -1};
    set_shape_labels(data_shape, 10);

    const auto data = make_shared<Parameter>(element::f32, data_shape);
    const auto out_shape = Constant::create<int64_t>(element::i64, Shape{2}, {5, 7});
    const auto op = make_op(data, out_shape);

    EXPECT_THAT(op->outputs(),
                Each(Property("PartialShape", &Output<Node>::get_partial_shape, PartialShape({2, 6, 5, 7}))));
    EXPECT_THAT(op->outputs(),
                Each(Property(&Output<Node>::get_partial_shape,
                              ResultOf(get_shape_labels, ElementsAre(10, 11, ov::no_label, ov::no_label)))));
}

TEST_F(AdaptiveMaxPoolV8Test, dynamic_output_shape) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{1, 6, 8, 9, 2});
    auto out_shape = make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(data, out_shape);

    EXPECT_THAT(op->outputs(),
                Each(Property("PartialShape", &Output<Node>::get_partial_shape, PartialShape({1, 6, -1, -1, -1}))));
}

TEST_F(AdaptiveMaxPoolV8Test, output_shape_as_parameter) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{1, 6, 8, 9, 2});
    auto out_shape = make_shared<Parameter>(element::i64, PartialShape{3});
    const auto op = make_op(data, out_shape);

    EXPECT_THAT(op->outputs(),
                Each(Property("PartialShape", &Output<Node>::get_partial_shape, PartialShape({1, 6, -1, -1, -1}))));
}

TEST_F(AdaptiveMaxPoolV8Test, data_dynamic_rank) {
    auto data = make_shared<Parameter>(element::f32, PartialShape::dynamic());
    auto out_shape = make_shared<Parameter>(element::i32, Shape{3});
    const auto op = make_op(data, out_shape);

    EXPECT_THAT(op->outputs(),
                Each(Property("PartialShape", &Output<Node>::get_partial_shape, PartialShape::dynamic())));
}

TEST_F(AdaptiveMaxPoolV8Test, out_spatial_shape_size_not_match_data_spatial_dimensions) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{2, 3, 5, 6});
    auto out_shape = make_shared<Parameter>(element::i32, Shape{3});

    OV_EXPECT_THROW(const auto op = make_op(data, out_shape),
                    NodeValidationFailure,
                    HasSubstr("Output shape for spatial dimension not compatible with data shape."));
}

TEST_F(AdaptiveMaxPoolV8Test, preserve_partial_values_and_labels_on_output_shape_input) {
    auto data_shape = PartialShape{{1, 2}, {2, 4}, 5, {10, 20}, -1};
    set_shape_labels(data_shape, 10);
    auto out_shape = PartialShape{{2, 6}, -1, {12, 13}};
    set_shape_labels(out_shape, 20);

    const auto data = make_shared<Parameter>(element::f32, data_shape);
    const auto spatial_dim_shape = make_shared<ShapeOf>(make_shared<Parameter>(element::i64, out_shape));
    const auto op = make_op(data, spatial_dim_shape);

    EXPECT_THAT(op->outputs(),
                Each(Property("PartialShape",
                              &Output<Node>::get_partial_shape,
                              PartialShape({{1, 2}, {2, 4}, {2, 6}, -1, {12, 13}}))));
    EXPECT_THAT(
        op->outputs(),
        Each(Property(&Output<Node>::get_partial_shape, ResultOf(get_shape_labels, ElementsAre(10, 11, 20, 21, 22)))));
}

TEST_F(AdaptiveMaxPoolV8Test, unsupported_input_shape) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{1, 6});
    auto out_shape = Constant::create<int64_t>(element::i64, Shape{}, {1});

    OV_EXPECT_THROW(const auto op = make_op(data, out_shape),
                    NodeValidationFailure,
                    HasSubstr("Expected a 3D, 4D or 5D tensor for the input. Got:"));
}

TEST_F(AdaptiveMaxPoolV8Test, wrong_out_shape) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{1, 6, 8, 9});
    auto out_shape = Constant::create<int64_t>(element::i64, Shape{3}, {5, 7, 8});

    OV_EXPECT_THROW(const auto op = make_op(data, out_shape),
                    NodeValidationFailure,
                    HasSubstr("Output shape for spatial dimension not compatible with data shape."));
}

TEST_F(AdaptiveMaxPoolV8Test, wrong_index_element_type) {
    auto data = make_shared<Parameter>(element::f32, PartialShape{1, 6, 8, 9});
    auto out_shape = Constant::create<int64_t>(element::i16, Shape{2}, {5, 7});

    OV_EXPECT_THROW(const auto op = make_op(data, out_shape, element::i16),
                    NodeValidationFailure,
                    HasSubstr("Index element type must be i32 or i64"));
}
