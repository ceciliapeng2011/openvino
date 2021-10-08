// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocess_steps_impl.hpp"

#include "color_utils.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/nv12_to_rgb.hpp"

namespace ov {
namespace preprocess {

static Shape construct_mean_scale_shape(const std::shared_ptr<Node>& node,
                                        size_t values_size,
                                        const PreprocessingContext& context) {
    // TODO: support also Mean/Scale image case
    auto node_shape = node->get_output_partial_shape(0);
    auto node_rank = node_shape.rank();
    auto channels_index = get_and_check_channels_idx(context.layout(), node_shape);
    std::vector<std::size_t> v(node_rank.get_length(), 1);

    OPENVINO_ASSERT(node_shape[channels_index].is_dynamic() || node_shape[channels_index] == values_size,
                    "Number of channels and mean/values size mismatch: Channels = ",
                    node_shape[channels_index].get_length(),
                    ", mean/scale = ",
                    values_size);
    v[channels_index] = values_size;
    return {v};
}

void PreProcessSteps::PreProcessStepsImpl::add_scale_impl(const std::vector<float>& values) {
    m_actions.emplace_back(std::make_tuple(
        [values](const std::vector<std::shared_ptr<Node>>& nodes,
                 const std::shared_ptr<ov::Function>& function,
                 PreprocessingContext& context) -> std::vector<std::shared_ptr<ov::Node>> {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't apply scale preprocessing for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply scale preprocessing for multi-plane input. Suggesting to convert current "
                            "image to RGB/BGR color format using 'convert_color'");
            Shape shape;
            if (values.size() == 1) {
                shape = Shape{1};
            } else {
                shape = construct_mean_scale_shape(nodes[0], values.size(), context);
            }
            auto constant = op::v0::Constant::create(element::f32, shape, values);
            inherit_friendly_names(function, nodes[0], constant, "/scale/Divide_Factor");

            auto new_op = std::make_shared<op::v1::Divide>(nodes[0], constant);
            inherit_friendly_names(function, nodes[0], new_op, "/scale/Divide");
            return {new_op};
        },
        false));
}

void PreProcessSteps::PreProcessStepsImpl::add_mean_impl(const std::vector<float>& values) {
    m_actions.emplace_back(std::make_tuple(
        [values](const std::vector<std::shared_ptr<Node>>& nodes,
                 const std::shared_ptr<ov::Function>& function,
                 PreprocessingContext& context) -> std::vector<std::shared_ptr<ov::Node>> {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't apply mean preprocessing for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply scale preprocessing for multi-plane input. Suggesting to convert current "
                            "image to RGB/BGR color format using 'convert_color'");
            Shape shape;
            if (values.size() == 1) {
                shape = Shape{1};
            } else {
                shape = construct_mean_scale_shape(nodes[0], values.size(), context);
            }
            auto constant = op::v0::Constant::create(element::f32, shape, values);
            inherit_friendly_names(function, nodes[0], constant, "/mean/Mean_Const");

            auto new_op = std::make_shared<op::v1::Subtract>(nodes[0], constant);
            inherit_friendly_names(function, nodes[0], new_op, "/mean/Subtract");
            return {new_op};
        },
        false));
}

void PreProcessSteps::PreProcessStepsImpl::add_convert_impl(const ov::element::Type& type) {
    m_actions.emplace_back(std::make_tuple(
        [type](const std::vector<std::shared_ptr<Node>>& nodes,
               const std::shared_ptr<ov::Function>& function,
               PreprocessingContext&) -> std::vector<std::shared_ptr<ov::Node>> {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't set element type for empty input.");
            std::vector<std::shared_ptr<ov::Node>> res;
            for (const auto& node : nodes) {
                OPENVINO_ASSERT(node->get_element_type().is_static(),
                                "Can't insert 'convert_element_type' for dynamic source tensor type.");
                if (node->get_element_type() != type) {
                    auto convert = std::make_shared<op::v0::Convert>(node, type);
                    inherit_friendly_names(function, node, convert, "/convert_element_type");
                    res.emplace_back(convert);
                } else {
                    res.emplace_back(node);
                }
            }
            return res;
        },
        true));
}

void PreProcessSteps::PreProcessStepsImpl::add_resize_impl(ResizeAlgorithm alg, int dst_height, int dst_width) {
    using InterpolateMode = op::v4::Interpolate::InterpolateMode;
    m_actions.emplace_back(std::make_tuple(
        [alg, dst_width, dst_height](const std::vector<std::shared_ptr<Node>>& nodes,
                                     const std::shared_ptr<ov::Function>& function,
                                     PreprocessingContext& ctxt) -> std::vector<std::shared_ptr<ov::Node>> {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't add resize for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't resize multi-plane input. Suggesting to convert current image to "
                            "RGB/BGR color format using 'PreProcessSteps::convert_color'");
            auto to_mode = [](ResizeAlgorithm alg) -> InterpolateMode {
                switch (alg) {
                case ResizeAlgorithm::RESIZE_NEAREST:
                    return InterpolateMode::NEAREST;
                case ResizeAlgorithm::RESIZE_CUBIC:
                    return InterpolateMode::CUBIC;
                case ResizeAlgorithm::RESIZE_LINEAR:
                default:
                    return InterpolateMode::LINEAR;
                }
            };
            auto node = nodes.front();
            auto layout = ctxt.layout();
            OPENVINO_ASSERT(ov::layout::has_height(layout) && ov::layout::has_width(layout),
                            "Can't add resize for layout without W/H specified. Use 'set_layout' API to define layout "
                            "of image data, like `NCHW`");
            auto node_rank = node->get_output_partial_shape(0).rank();
            OPENVINO_ASSERT(node_rank.is_static(), "Resize operation is not supported for fully dynamic shape");

            auto height_idx = static_cast<int64_t>(get_and_check_height_idx(layout, node->get_output_partial_shape(0)));
            auto width_idx = static_cast<int64_t>(get_and_check_width_idx(layout, node->get_output_partial_shape(0)));
            if (dst_height < 0 || dst_width < 0) {
                OPENVINO_ASSERT(ctxt.network_shape().rank().is_static(),
                                "Resize is not fully specified while target network shape is dynamic");
            }
            int new_image_width = dst_width < 0 ? static_cast<int>(ctxt.get_network_width_for_resize()) : dst_width;
            int new_image_height = dst_height < 0 ? static_cast<int>(ctxt.get_network_height_for_resize()) : dst_height;

            auto target_spatial_shape =
                op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {new_image_height, new_image_width});
            auto scales = op::v0::Constant::create<float>(element::f32, Shape{2}, {1, 1});
            // In future consider replacing this to set of new OV operations like `getDimByName(node, "H")`
            // This is to allow specifying layout on 'evaluation' stage
            auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {height_idx, width_idx});

            op::v4::Interpolate::InterpolateAttrs attrs(to_mode(alg),
                                                        op::v4::Interpolate::ShapeCalcMode::SIZES,
                                                        {0, 0},
                                                        {0, 0});

            auto interp = std::make_shared<op::v4::Interpolate>(node, target_spatial_shape, scales, axes, attrs);
            inherit_friendly_names(function, nodes[0], interp, "/resize");
            return {interp};
        },
        true));
}

void PreProcessSteps::PreProcessStepsImpl::add_convert_layout_impl(const Layout& layout) {
    m_actions.emplace_back(std::make_tuple(
        [layout](const std::vector<std::shared_ptr<Node>>& nodes,
                 const std::shared_ptr<ov::Function>& function,
                 PreprocessingContext& context) -> std::vector<std::shared_ptr<ov::Node>> {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't convert layout for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't convert layout for multi-plane input. Suggesting to convert current image to "
                            "RGB/BGR color format using 'convert_color'");
            Layout dst_layout = layout.empty() ? context.target_layout() : layout;
            auto permutation =
                layout::find_permutation(context.layout(), nodes[0]->get_output_partial_shape(0).rank(), dst_layout);
            auto perm_constant =
                op::v0::Constant::create<int64_t>(element::i64, Shape{permutation.size()}, permutation);
            auto transpose = std::make_shared<op::v1::Transpose>(nodes[0], perm_constant);
            inherit_friendly_names(function, nodes[0], transpose, "/convert_layout");
            context.layout() = dst_layout;  // Update context's current layout
            return {transpose};
        },
        true));
}

void PreProcessSteps::PreProcessStepsImpl::add_convert_color_impl(const ColorFormat& dst_format) {
    m_actions.emplace_back(std::make_tuple(
        [&, dst_format](const std::vector<std::shared_ptr<Node>>& nodes,
                        const std::shared_ptr<ov::Function>& function,
                        PreprocessingContext& context) -> std::vector<std::shared_ptr<ov::Node>> {
            if (context.color_format() == dst_format) {
                return nodes;
            }
            if (context.color_format() == ColorFormat::NV12_SINGLE_PLANE) {
                OPENVINO_ASSERT(nodes.size() == 1,
                                "Internal error: single plane NV12 image can't have multiple inputs");
                std::shared_ptr<Node> convert;
                switch (dst_format) {
                case ColorFormat::RGB:
                    convert = std::make_shared<op::v8::NV12toRGB>(nodes[0]);
                    break;
                case ColorFormat::BGR:
                    convert = std::make_shared<op::v8::NV12toBGR>(nodes[0]);
                    break;
                default:
                    OPENVINO_ASSERT(false,
                                    "Unsupported conversion from NV12 to '",
                                    color_format_name(dst_format),
                                    "' format:");
                }
                inherit_friendly_names(function, nodes[0], convert, "/convert_color_nv12_single");
                context.color_format() = dst_format;
                return {convert};
            } else if (context.color_format() == ColorFormat::NV12_TWO_PLANES) {
                OPENVINO_ASSERT(nodes.size() == 2, "Internal error: two-plane NV12 image must have exactly two inputs");
                std::shared_ptr<Node> convert;
                switch (dst_format) {
                case ColorFormat::RGB:
                    convert = std::make_shared<op::v8::NV12toRGB>(nodes[0], nodes[1]);
                    break;
                case ColorFormat::BGR:
                    convert = std::make_shared<op::v8::NV12toBGR>(nodes[0], nodes[1]);
                    break;
                default:
                    OPENVINO_ASSERT(false,
                                    "Unsupported conversion from NV12 to '",
                                    color_format_name(dst_format),
                                    "' format:");
                }
                inherit_friendly_names(function, nodes[0], convert, "/convert_color_nv12_two_planes");
                context.color_format() = dst_format;
                return {convert};
            }
            OPENVINO_ASSERT(false,
                            "Source color format '",
                            color_format_name(context.color_format()),
                            "' is not convertible to any other");
        },
        true));
}

//------------- Post processing ------
void PostStepsList::add_convert_impl(const ov::element::Type& type) {
    m_actions.emplace_back([type](const ov::Output<Node>& node, PostprocessingContext& ctxt) {
        ov::element::Type t = type;
        if (t == element::Type{}) {
            t = ctxt.target_element_type();
        }
        if (t == node.get_node()->get_element_type()) {
            return std::make_tuple(node, false);
        }
        OPENVINO_ASSERT(
            !t.is_dynamic() && t != element::undefined,
            "Can't convert to dynamic/unknown element type, consider using of InputTensorInfo::set_element_type");
        auto convert = std::make_shared<op::v0::Convert>(node, t);
        convert->set_friendly_name(node.get_node()->get_friendly_name() + "/convert_element_type");
        return std::make_tuple(ov::Output<ov::Node>(convert), true);
    });
}

void PostStepsList::add_convert_layout_impl(const Layout& layout) {
    m_actions.emplace_back([layout](const ov::Output<Node>& node, PostprocessingContext& context) {
        Layout dst_layout = layout.empty() ? context.target_layout() : layout;
        if (dst_layout == context.layout()) {
            return std::make_tuple(node, false);
        }
        auto permutation = layout::find_permutation(context.layout(), node.get_partial_shape().rank(), dst_layout);
        auto perm_constant = op::v0::Constant::create<int64_t>(element::i64, Shape{permutation.size()}, permutation);
        auto transpose = std::make_shared<op::v1::Transpose>(node, perm_constant);
        transpose->set_friendly_name(node.get_node()->get_friendly_name() + "/convert_layout");
        context.layout() = dst_layout;  // Update context's current layout
        return std::make_tuple(ov::Output<ov::Node>(transpose), true);
    });
}

}  // namespace preprocess
}  // namespace ov
