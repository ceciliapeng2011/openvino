// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensoriterator.h"

#include <string>
#include <vector>
#include <dnnl_extension_utils.h>
#include <ie_ngraph_utils.hpp>
#include <utils/general_utils.h>
#include "common/blocked_desc_creator.h"
#include "utils/ngraph_utils.hpp"
#include "transformations/utils/utils.hpp"
#include "common/cpu_memcpy.h"
#include "common/reorder_prim.h"
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>

using namespace dnnl;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {
namespace node {

#define THROW_ERROR IE_THROW() << getTypeStr() << " layer with name '" << getName() << "' "

static NodeConfig make_plain_config(const std::shared_ptr<ov::Node>& op) {
    NodeConfig config;

    for (size_t i = 0; i < op->get_input_size(); i++) {
        const auto &origShape = op->get_input_partial_shape(i);
        const auto& shape = Shape(origShape.rank().get_length() == 0 ? ov::PartialShape{1} : origShape);
        const auto prec = InferenceEngine::details::convertPrecision(op->get_input_element_type(i));

        PortConfig data_conf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        data_conf.setMemDesc(descCreator->createSharedDesc(prec, shape));
        config.inConfs.push_back(data_conf);
    }

    for (size_t i = 0; i < op->get_output_size(); i++) {
        const auto &origShape = op->get_output_partial_shape(i);
        const auto& shape = Shape(origShape.rank().get_length() == 0 ? ov::PartialShape{1} : origShape);
        const auto prec = InferenceEngine::details::convertPrecision(op->get_output_element_type(i));

        PortConfig data_conf {};
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        data_conf.setMemDesc(descCreator->createSharedDesc(prec, shape));
        config.outConfs.push_back(data_conf);
    }

    config.dynBatchSupport = true;
    return config;
}

static void redefineToMemories(const std::vector<MemoryPtr>& to_mems, MemoryDescPtr new_desc) {
    const auto &currDesc = to_mems.front()->getDesc();
    if (currDesc.getShape().isDynamic() || currDesc.getShape().getStaticDims() != new_desc->getShape().getStaticDims()) {
        // TODO : check the entire dstMemPtrs usage considering the proper memory sharing
        for (size_t j = 0; j < to_mems.size(); j++) {
            to_mems[j]->redefineDesc(new_desc);
        }
    }
}

// this method get all memory ptrs of childs of one port to redefine descs for them
static std::vector<MemoryPtr> getToMemories(const Node* node, const size_t port) {
    std::vector<MemoryPtr> memories;
    for (auto& edge : node->getChildEdgesAtPort(port))
        memories.push_back(edge->getMemoryPtr());
    return memories;
}

static void nullifyUndefinedDims(VectorDims& dims) {
    std::transform(dims.begin(), dims.end(), dims.begin(), [](const size_t& dim) {
        return dim == Shape::UNDEFINED_DIM ? 0 : dim;
    });
}

class PortIteratorHelper : public PortMapHelper {
public:
    PortIteratorHelper(MultiCachePtr cache, const MemoryPtr &from, const MemoryPtr &to, bool sliced_src,
                       const PortMap &slice_rule, const dnnl::engine& eng)
                       : sliced_src(sliced_src) {
        const auto &full_blob = sliced_src ? from : to;
        const auto &part_blob = !sliced_src ? from : to;

        auto axis = slice_rule.axis;
        auto stride = slice_rule.stride;

        auto full_dims = full_blob->GetShape().getStaticDims();
        auto part_dims = part_blob->GetShape().getStaticDims();

        auto abs_stride = std::abs(stride);
        auto sign_of_stride = stride < 0.0f ? -1 : 1;

        iter_count = full_dims[axis] / abs_stride;

        full_dims[axis] = abs_stride;
        IE_ASSERT(full_dims == part_dims) << "Shape mismatch for tensor iterator port";

        // make chunk view
        auto chunk_desc = full_blob->GetDescWithType<DnnlMemoryDesc>()->getDnnlDesc();
        chunk_desc.data.dims[axis] = abs_stride;
        chunk_desc.data.padded_dims[axis] = abs_stride;  // TODO: asamption that plain tensor

        full_mem = full_blob->GetPrimitive();
        const auto full_mem_handler = full_mem.get_data_handle();
        dnnl::memory chunk_mem = {chunk_desc, eng, full_mem_handler};

        auto elem_size = DnnlExtensionUtils::sizeOfDataType(dnnl::memory::data_type(chunk_desc.data.data_type));

        chunk_stride_in_byte = chunk_desc.data.format_desc.blocking.strides[axis] * elem_size * abs_stride;
        chunk_offset_in_byte = sign_of_stride < 0 ? (iter_count - 1) * chunk_stride_in_byte : 0;
        chunk_stride_in_byte *= sign_of_stride;

        if (sliced_src) {
            mem_holder_src = chunk_mem;
            mem_holder_dst = to->GetPrimitive();
        } else {
            mem_holder_src = from->GetPrimitive();
            mem_holder_dst = chunk_mem;
        }
        reorder = getReorderPrim(cache, mem_holder_dst.get_engine(), mem_holder_src.get_desc(), mem_holder_dst.get_desc());
    }

    void execute(dnnl::stream strm, int iter) override {
        IE_ASSERT(iter >= 0 && iter < iter_count);

        auto &chunk_mem = sliced_src ? mem_holder_src : mem_holder_dst;
        chunk_mem.set_data_handle(static_cast<uint8_t *>(full_mem.get_data_handle()) +
                                          chunk_offset_in_byte + chunk_stride_in_byte * iter);

        reorder.execute(strm, {{DNNL_ARG_FROM, mem_holder_src}, {DNNL_ARG_TO, mem_holder_dst}});
    }

private:
    ptrdiff_t chunk_stride_in_byte = 0;
    ptrdiff_t chunk_offset_in_byte = 0;

    bool sliced_src;
    dnnl::memory full_mem;

    int iter_count;
};

class BackEdgePortHelper : public PortMapHelper {
public:
    BackEdgePortHelper(MultiCachePtr cache, const MemoryPtr &from, const MemoryPtr &to, const dnnl::engine& eng) {
        mem_holder_src = from->GetPrimitive();
        mem_holder_dst = to->GetPrimitive();
        reorder = getReorderPrim(cache, mem_holder_dst.get_engine(), mem_holder_src.get_desc(), mem_holder_dst.get_desc());
    }

    void execute(dnnl::stream strm, int iter = -1) override {
        if (iter != 0) {
            reorder.execute(strm, {{DNNL_ARG_FROM, mem_holder_src}, {DNNL_ARG_TO, mem_holder_dst}});
        }
    }
};

class IterCountPortHelper : public PortMapHelper {
public:
    IterCountPortHelper(const MemoryPtr &to, const dnnl::engine& eng) {
        // Only scalar I32 tensor is supported
        IE_ASSERT(to->GetDataType() == memory::data_type::s32);
        IE_ASSERT(to->GetShape() == Shape(VectorDims{1}));
        mem_holder_dst = to->GetPrimitive();
    }

    void execute(dnnl::stream strm, int n_iter) override {
        auto mem = mem_holder_dst;
        auto data_ptr = static_cast<uint32_t*>(mem.get_data_handle());
        if (data_ptr == nullptr) {
            IE_THROW() << "TensorIterator node has not allocated memory for IterCountPortHelper";
        }
        *data_ptr = n_iter;
    }
};

class asBoolCheck : public PortChecker {
public:
    asBoolCheck(const MemoryPtr &mem) {
        IE_ASSERT(mem->GetDataType() == memory::data_type::u8);
        IE_ASSERT(mem->GetShape() == Shape(InferenceEngine::SizeVector{1}));
        mem_holder = mem->GetPrimitive();
    }

    int getStatus() override {
        auto data_ptr = static_cast<uint8_t*>(mem_holder.get_data_handle());
        if (data_ptr == nullptr) {
            IE_THROW() << "TensorIterator node has not allocated memory for asBoolCheck";
        }
        return *data_ptr == static_cast<uint8_t>(0) ? 0 : 1;
    }
};

class asIntCheck : public PortChecker {
public:
    asIntCheck(const MemoryPtr &mem) {
        IE_ASSERT(mem->GetDataType() == memory::data_type::s32);
        IE_ASSERT(mem->GetShape() == Shape(InferenceEngine::SizeVector{1}));
        mem_holder = mem->GetPrimitive();
    }

    int getStatus() override {
        auto data_ptr = static_cast<uint32_t*>(mem_holder.get_data_handle());
        if (data_ptr == nullptr) {
            IE_THROW() << "TensorIterator node has not allocated memory for asIntCheck";
        }
        return *data_ptr;
    }
};

class staticValueCheck : public PortChecker {
public:
    staticValueCheck(const int &value) : value(value) {}

    int getStatus() override {
        return value;
    }
private:
    int value;
};

DynamicBuffer::DynamicBuffer(const MemoryPtr &from_, const std::vector<MemoryPtr> &to_,
                             const PortMap &map_rule_) : from(from_), to(to_), map_rule(map_rule_) {
    elem_size = DnnlExtensionUtils::sizeOfDataType(from->GetDataType());
}

void DynamicBuffer::execute(const dnnl::engine& eng, const int iter) {
    if (from->getStaticDims()[map_rule.axis] != std::abs(map_rule.stride))
        IE_THROW() << "TensorIterator (Loop) has incorrect output shape[axis] after iteration for concatenation. " << std::abs(map_rule.stride) <<
        " is expected, but actual: " << from->getStaticDims()[map_rule.axis];

    if (iter == 0) {
        init(eng);
    }

    // if chunk_offset_in_byte out of range of buffer holder, reallocate a larger chunk
    if (check_buffer()) {
        auto new_buffer = create_buffer(eng);
        move_buffer(new_buffer);
    }

    move_data();
}

void DynamicBuffer::reset(int max_iter_count_) {
    max_iter_count = max_iter_count_;
}

void DynamicBuffer::init(const dnnl::engine& eng) {
    const auto stride = map_rule.stride;
    const auto abs_stride = std::abs(stride);

    // We have no idea of "from" node memory dims until the sub_graph has been executed.
    const auto& src_mem = from->GetPrimitive();
    const auto& src_desc = src_mem.get_desc();
    const auto& dims = src_desc.dims();
    count = std::accumulate(dims.begin(), dims.begin() + map_rule.axis, size_t(1), std::multiplies<size_t>());
    len = std::accumulate(dims.begin() + map_rule.axis + 1, dims.end(), elem_size, std::multiplies<size_t>());
    chunk_unit_in_byte = abs_stride * len;

    if (!mem_holder_buffer) { // else reuse buffer holder of last inference
        // preallocate a large chunk of memory to hold intermediate concated outputs of all iterations.
        mem_holder_buffer = create_buffer(eng);
    }

    // reset chunk_offset_in_byte since the first execution
    chunk_stride_in_byte = mem_holder_buffer->GetSize() / count;
    chunk_offset_in_byte = stride > 0 ? 0 : (chunk_stride_in_byte - chunk_unit_in_byte);
    num_execs = 0;
}

bool DynamicBuffer::check_buffer() {
    if (map_rule.stride > 0) {
        if (chunk_offset_in_byte + chunk_unit_in_byte > chunk_stride_in_byte) return true;
    } else {
        if (chunk_offset_in_byte < 0) return true;
    }
    return false;
}

MemoryPtr DynamicBuffer::create_buffer(const dnnl::engine& eng) {
    const auto abs_stride = std::abs(map_rule.stride);

    const auto estimate_iters = [&] () {
        if (max_iter_count != -1) return max_iter_count;

        // in case of no idea of memory upper boundary
        return (num_execs == 0) ? 1 : 2 * num_execs; // growth factor 2
    };
    const auto estimated_iters = estimate_iters();
    const Shape _shape = Shape({count, static_cast<size_t>(abs_stride * estimated_iters), len/elem_size});
    auto _descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
    auto new_buffer_desc = _descCreator->createSharedDesc(from->getDesc().getPrecision(), _shape);

    auto _ptr = std::make_shared<Memory>(eng);
    _ptr->Create(*new_buffer_desc);
    return _ptr;
}

void DynamicBuffer::move_buffer(MemoryPtr& new_buffer) {
    const auto stride = map_rule.stride;

    // copy data from old buffer to new buffer
    const auto src_stride = chunk_stride_in_byte;
    const auto dst_stride = new_buffer->getDesc().as<BlockedMemoryDesc>()->getBlockDims()[1] * len;

    const auto valid_size = chunk_unit_in_byte * num_execs;
    const auto src_offset_in_byte = stride > 0 ? 0 : (src_stride - valid_size);
    chunk_offset_in_byte = stride > 0 ? 0 : (dst_stride - valid_size);  // reset chunk_offset_in_byte

    copy(reinterpret_cast<uint8_t*>(mem_holder_buffer->GetPtr()) + src_offset_in_byte, reinterpret_cast<uint8_t*>(new_buffer->GetPtr()) + chunk_offset_in_byte,
        src_stride, dst_stride, count, valid_size);

    // assign mem_holder_buffer
    mem_holder_buffer = new_buffer;
    chunk_stride_in_byte = mem_holder_buffer->GetSize() / count;

    // adjust for next execution
    if (stride > 0) {
        chunk_offset_in_byte += valid_size;
    } else {
        chunk_offset_in_byte -= chunk_unit_in_byte;
    }
}

void DynamicBuffer::move_data() {
    const auto src_stride = abs(map_rule.stride) * len;
    const auto dst_stride = chunk_stride_in_byte;

    copy(reinterpret_cast<const uint8_t*>(from->GetPtr()), reinterpret_cast<uint8_t*>(mem_holder_buffer->GetPtr()) + chunk_offset_in_byte,
         src_stride, dst_stride, count, chunk_unit_in_byte);

    // adjust for next execution
    num_execs++;
    if (map_rule.stride > 0) {
        chunk_offset_in_byte += chunk_unit_in_byte;
    } else {
        chunk_offset_in_byte -= chunk_unit_in_byte;
    }
}

void DynamicBuffer::transfer(const Node* node) {
    if (mem_holder_buffer && num_execs > 0) {
        const auto axis = map_rule.axis;
        const auto stride = map_rule.stride;
        const auto abs_stride = std::abs(stride);

        const auto& src_mem = from->GetPrimitive();
        const auto& src_desc = src_mem.get_desc();
        auto dims = src_desc.dims();
        dims[axis] = abs_stride * num_execs;
        const auto desc = node->getBaseMemDescAtOutputPort(map_rule.from)->cloneWithNewDims(
                DnnlExtensionUtils::convertToVectorDims(dims));

        redefineToMemories(to, desc);

        const auto src_stride = chunk_stride_in_byte;
        const auto dst_stride = to.front()->getStaticDims()[axis] * len;
        const auto valid_size = chunk_unit_in_byte * num_execs;
        const auto src_offset_in_byte = stride > 0 ? 0 : (src_stride - valid_size);
        copy(reinterpret_cast<uint8_t*>(mem_holder_buffer->GetPtr()) + src_offset_in_byte, reinterpret_cast<uint8_t*>(to.front()->GetPtr()),
            src_stride, dst_stride, count, dst_stride);
    } else {
        VectorDims newDims = to.front()->GetShape().getDims();
        nullifyUndefinedDims(newDims);

        const auto desc = node->getBaseMemDescAtOutputPort(map_rule.from)->cloneWithNewDims(newDims);
        redefineToMemories(to, desc);
    }
}

void DynamicBuffer::copy(const uint8_t* src, uint8_t* dst, const size_t src_stride, const size_t dst_stride, const size_t count, const size_t len) {
    parallel_for(count, [&](const size_t i) {
        cpu_memcpy(&dst[i * dst_stride], &src[i * src_stride], len);
    });
}

bool TensorIterator::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::op::v0::TensorIterator::get_type_info_static(),
                    ov::op::v5::Loop::get_type_info_static())) {
            errorMessage = "Only opset1 TensorIterator or opset5 Loop operations are supported.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

TensorIterator::TensorIterator(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, InternalDynShapeInferFactory()), ngraphOp(op) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void TensorIterator::getSupportedDescriptors() {
    auto tiOp = ov::as_type_ptr<const ov::op::util::SubGraphOp>(ngraphOp);
    if (!tiOp) {
        THROW_ERROR << "cannot be cast to ov::op::util::SubGraphOp";
    }
    const std::shared_ptr<const ov::Model> body = tiOp->get_function();
    sub_graph.CreateGraph(body, context);

    const auto &inMap = sub_graph.GetInputNodesMap();
    for (const auto &param : tiOp->get_function()->get_parameters()) {
        auto inNode = inMap.find(param->get_friendly_name());
        if (inNode != inMap.end()) {
            input_mems.push_back(getToMemories(inNode->second.get(), 0));
        }
    }

    const auto &outMap = sub_graph.GetOutputNodesMap();
    for (const auto &out : tiOp->get_function()->get_results()) {
        const auto prev = out->input_value(0);
        const auto inputID = ov::op::util::create_ie_output_name(prev);
        auto outNode = outMap.find(inputID);
        if (outNode != outMap.end()) {
            auto outMem = outNode->second->getParentEdgeAt(0)->getMemoryPtr();
            output_mem.push_back(outMem);
        }
    }

    // Port map: outputs
    for (const auto& desc : tiOp->get_output_descriptions()) {
        auto body_output_idx = desc->m_body_value_index;

        std::string type_name = desc->get_type_info().name;
        if (type_name == "ConcatOutputDescription") {
            auto output_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::ConcatOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            outputPortMap.emplace_back(PortMap {
                    static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx),
                    static_cast<int>(output_desc->m_axis), static_cast<int>(output_desc->m_stride),
                    static_cast<int>(output_desc->m_start), static_cast<int>(output_desc->m_end),
                    static_cast<int>(output_desc->m_part_size)});
        } else if (type_name == "BodyOutputDescription") {
            auto output_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::BodyOutputDescription>(desc);
            IE_ASSERT(output_desc != nullptr);

            outputPortMap.emplace_back(PortMap {
                    static_cast<int>(output_desc->m_output_index), static_cast<int>(body_output_idx), -1, 1, 0, -1, 1});
        } else {
            IE_THROW() << "Incorrect type of the output description.";
        }
    }

    // Port map : inputs and back edges
    for (const auto& desc : tiOp->get_input_descriptions()) {
        auto body_input_index = desc->m_body_parameter_index;

        if (auto slice_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::SliceInputDescription>(desc)) {
            inputPortMap.emplace_back(PortMap {
                    static_cast<int>(slice_desc->m_input_index), static_cast<int>(body_input_index),
                    static_cast<int>(slice_desc->m_axis), static_cast<int>(slice_desc->m_stride),
                    static_cast<int>(slice_desc->m_start), static_cast<int>(slice_desc->m_end),
                    static_cast<int>(slice_desc->m_part_size)});
        } else if (auto merge_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::MergedInputDescription>(desc)) {
            inputPortMap.emplace_back(PortMap {
                    static_cast<int>(merge_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});

            auto body_output_idx = merge_desc->m_body_value_index;

            backEdges.emplace_back(PortMap {
                    static_cast<int>(body_output_idx), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else if (auto inv_desc = ov::as_type_ptr<const ov::op::util::SubGraphOp::InvariantInputDescription>(desc)) {
            inputPortMap.emplace_back(PortMap {
                    static_cast<int>(inv_desc->m_input_index), static_cast<int>(body_input_index), -1, 1, 0, -1, 1});
        } else {
            THROW_ERROR << "has incorrect type of the input description.";
        }
    }

    if (auto loopOp = ov::as_type_ptr<const ov::op::v5::Loop>(ngraphOp)) {
        algorithm = Algorithm::TensorIteratorLoop;
        auto spec_port = loopOp->get_special_body_ports();
        if (spec_port.current_iteration_input_idx != -1) {
            loopBodyCurrentIterationIdx.push_back(spec_port.current_iteration_input_idx);
        }
        if (spec_port.body_condition_output_idx != -1) {
            loopBodyConditionOutputIdx = spec_port.body_condition_output_idx;
        }
        loopTripCountIdx = 0;
        loopExecutionConditionIdx = 1;
    } else if (auto ti = ov::as_type_ptr<const ov::op::v0::TensorIterator>(ngraphOp)) {
        algorithm = Algorithm::TensorIteratorCommon;
    } else {
        THROW_ERROR << "isn't supported!";
    }
}

void TensorIterator::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    supportedPrimitiveDescriptors.emplace_back(make_plain_config(ngraphOp), impl_desc_type::unknown);
}

void TensorIterator::createPrimitive() {
    if (loopBodyConditionOutputIdx == -1)
        continue_cond_check.reset(new staticValueCheck(true)); // always true
    if (loopExecutionConditionIdx == -1) {
        initial_cond_check.reset(new staticValueCheck(true));
        lastUsedCond = initial_cond_check->getStatus();
    }

    if (isDynamicNode())
        prepareDynamicBuffers();

    Node::createPrimitive();
}

bool TensorIterator::needPrepareParams() const {
    if (getAlgorithm() == Algorithm::TensorIteratorLoop) {
        const auto tripCountPtr = reinterpret_cast<const uint32_t*>(getParentEdgesAtPort(loopTripCountIdx).front()->getMemoryPtr()->GetPtr());
        const auto condPtr = reinterpret_cast<const uint8_t*>(getParentEdgesAtPort(loopExecutionConditionIdx).front()->getMemoryPtr()->GetPtr());
        if (tripCountPtr[0] != lastUsedTripCount || static_cast<bool>(condPtr[0]) != lastUsedCond)
            return true;
    }

    // If sliced input shapes of node and body input shapes aren't equal, we should reshape body
    if (checkForInputAndBodyShapesInequality()) {
        return true;
    }

    // In cases, when sliced input shapes of node and body input shapes are equal,
    // original input shapes of node may be not equal to previous input shapes and count of iterations will be another
    // For example, TensorIterator with single sliced input by axis 1:
    //    Input shape of node: [10, 8, 10]  ->  Sliced input shape: [10, 1, 10]  ->  Body input shape:  [10, 1, 10]  -> 8 iterations
    //    Input shape of node: [10, 4, 10]  ->  Sliced input shape: [10, 1, 10]  ->  Body input shape:  [10, 1, 10]  -> 4 iterations
    // Thus, sliced input shapes and body input shapes are equal but iteration counts are different. So we should update trip count
    return Node::needPrepareParams();
}

void TensorIterator::prepareParams() {
    prepareTripCount();
    prepareInitialCond();

    first_mappers.clear();
    before_mappers.clear();
    back_mappers.clear();

    if ((lastUsedCond && lastUsedTripCount != 0) || !isDynamicNode()) {
        reshapeSubgraphInput();

        prepareInputPorts();
        prepareContinueCond();
        prepareLoopBodyCurrentIteration();

        if (!isDynamicNode()) {
            prepareOutputPorts();
            prepareBackEdges();
        }

        // reset local states of DynamicBuffer
        for (auto& buffer : buffers)
            buffer->reset(lastUsedTripCount);
    }
}

void TensorIterator::execute(dnnl::stream strm) {
    sub_graph.ResetInferCount();

    bool continue_cond = initial_cond_check->getStatus();
    int max_num_iter = trip_count_check->getStatus();

    for (auto &mapper : first_mappers)
        mapper->execute(strm);

    // use  "i != max_num_iter" only to allow "-1" works like infinite loop
    for (int i = 0; i != max_num_iter && continue_cond; i++) {
        // copy data to subgraph iteration
        for (auto &mapper : before_mappers)
            mapper->execute(strm, i);

        sub_graph.Infer();

        continue_cond = continue_cond_check->getStatus();

        // copy data from subgraph iteration to outputs
        // or to the next iteration inputs
        for (auto &mapper : after_mappers)
            mapper->execute(strm, i);
    }

    for (auto &mapper : last_mappers)
        mapper->execute(strm);
}

void TensorIterator::executeDynamicImpl(dnnl::stream strm) {
    const auto &eng = getEngine();
    sub_graph.ResetInferCount();

    bool continue_cond = initial_cond_check->getStatus();
    int max_num_iter = trip_count_check->getStatus();

    for (auto &mapper : first_mappers)
        mapper->execute(strm);

    // use  "i != max_num_iter" only to allow "-1" works like infinite loop
    for (int i = 0; i != max_num_iter && continue_cond; i++) {
        // copy data to subgraph iteration
        for (auto &mapper : before_mappers)
            mapper->execute(strm, i);
        for (auto &mapper : back_mappers)
            mapper->execute(strm, i);

        sub_graph.Infer();

        continue_cond = continue_cond_check->getStatus();

        for (auto& buffer : buffers)
            buffer->execute(eng, i);

        // on the last iteration we shouldn't reshape body inputs and init back edges
        if ((i + 1 != max_num_iter) && continue_cond)
            prepareDynamicBackEdges();
    }

    reshapeAndFillOutput(strm);
}

/* *==============* Prepare reorders, edges between body and TI *==============* */

void TensorIterator::prepareInputPorts() {
    const auto &eng = getEngine();
    for (auto map_rule : inputPortMap) {
        auto &from_mem = getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &to_mem = input_mems[map_rule.to].front();  // first memory is enough to access the shared underlying physical memory

        if (map_rule.axis == -1)
            first_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(context->getParamsCache(), from_mem, to_mem, eng));
        else
            before_mappers.emplace_back(
                    std::make_shared<PortIteratorHelper>(context->getParamsCache(), from_mem, to_mem, true, map_rule, eng));
    }
}

void TensorIterator::prepareOutputPorts() {
    const auto &eng = getEngine();
    for (auto map_rule : outputPortMap) {
        auto &to_mem = getChildEdgesAtPort(map_rule.from)[0]->getMemoryPtr();
        auto &from_mem = output_mem[map_rule.to];

        if (map_rule.axis == -1)
            last_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(context->getParamsCache(), from_mem, to_mem, eng));
        else
            after_mappers.emplace_back(std::make_shared<PortIteratorHelper>(context->getParamsCache(), from_mem, to_mem, false, map_rule, eng));
    }
}

void TensorIterator::prepareBackEdges() {
    const auto &eng = getEngine();
    for (auto map_rule : backEdges) {
        auto from_mem = output_mem[map_rule.from];
        auto to_mem = input_mems[map_rule.to].front();

        before_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(context->getParamsCache(), from_mem, to_mem, eng));
    }
}

void TensorIterator::prepareDynamicBackEdges() {
    const auto &eng = getEngine();
    back_mappers.clear();
    for (auto map_rule : backEdges) {
        auto from_mem = output_mem[map_rule.from];
        auto to_mems = input_mems[map_rule.to];

        redefineToMemories(to_mems, from_mem->getDescPtr());

        // first memory is enough to get common memory ptr
        back_mappers.emplace_back(std::make_shared<BackEdgePortHelper>(context->getParamsCache(), from_mem, to_mems.front(), eng));
    }
}

void TensorIterator::prepareDynamicBuffers() {
    for (auto map_rule : outputPortMap) {
        if (map_rule.axis != -1) {
            auto to_mems = getToMemories(this, map_rule.from);
            auto &from_mem = output_mem[map_rule.to];
            buffers.emplace_back(std::make_shared<DynamicBuffer>(from_mem, to_mems, map_rule));
        }
    }
}

void TensorIterator::prepareLoopBodyCurrentIteration() {
    const auto &eng = getEngine();
    for (auto idx : loopBodyCurrentIterationIdx) {
        auto to_mem = input_mems[idx].front();  // first memory is enough to get common memory ptr
        before_mappers.emplace_back(std::make_shared<IterCountPortHelper>(to_mem, eng));
    }
}

void TensorIterator::prepareContinueCond() {
    if (loopBodyConditionOutputIdx != -1 || !continue_cond_check) {
        auto mem = output_mem[loopBodyConditionOutputIdx];
        continue_cond_check.reset(new asBoolCheck(mem));
    }
}

void TensorIterator::prepareInitialCond() {
    if (loopExecutionConditionIdx != -1 || !initial_cond_check) {
        auto mem = getParentEdgesAtPort(loopExecutionConditionIdx)[0]->getMemoryPtr();
        initial_cond_check.reset(new asBoolCheck(mem));
        lastUsedCond = initial_cond_check->getStatus();
    }
}

void TensorIterator::prepareTripCount() {
    if (loopTripCountIdx == -1) {
        trip_count_check.reset(new staticValueCheck(getNumIteration(inputPortMap, outputPortMap)));
    } else {
        auto mem = getParentEdgesAtPort(loopTripCountIdx)[0]->getMemoryPtr();
        trip_count_check.reset(new asIntCheck(mem));
    }
    lastUsedTripCount = trip_count_check->getStatus();
}

/* *==============* *==============* *==============* *==============* *==============* */

inline SizeVector sliced_input_dims(const MemoryPtr& mem, const int axis, const int stride) {
    auto dims = mem->getStaticDims();
    if (axis != -1)
        dims[axis] = abs(stride);
    return dims;
}

void TensorIterator::reshapeSubgraphInput() {
    for (auto map_rule : inputPortMap) {
        auto new_dims = sliced_input_dims(getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr(), map_rule.axis, map_rule.stride);
        auto &to_mems = input_mems[map_rule.to];
        const auto& body_inshape = to_mems.front()->GetShape();
        if (body_inshape.isDynamic() || body_inshape.getDims() != new_dims) {
            const auto desc = std::make_shared<CpuBlockedMemoryDesc>(to_mems.front()->getDesc().getPrecision(), Shape(new_dims));
            redefineToMemories(to_mems, desc);
        }
    }
}

void TensorIterator::reshapeAndFillOutput(dnnl::stream strm) {
    auto eng = strm.get_engine();
    for (auto map_rule : outputPortMap) {
        if (map_rule.axis == -1) {
            auto to_mems = getToMemories(this, map_rule.from);
            auto &from_mem = output_mem[map_rule.to];

            // if Loop or TI isn't executed we should fill dynamic dims by zero
            auto newShape = from_mem->GetShape();
            auto newDims = newShape.getDims();
            nullifyUndefinedDims(newDims);

            const bool hasZeroDims = std::count(std::begin(newDims), std::end(newDims), 0) > 0;
            const auto desc = getBaseMemDescAtOutputPort(map_rule.from)->cloneWithNewDims(newDims, hasZeroDims);
            redefineToMemories(to_mems, desc);

            if (!newShape.isDynamic()) {
                BackEdgePortHelper mapper(context->getParamsCache(), from_mem, to_mems.front(), eng);
                mapper.execute(strm);
            }
        }
    }

    for (auto buffer : buffers) {
        buffer->transfer(this);
    }
}

bool TensorIterator::checkForInputAndBodyShapesInequality() const {
    for (auto map_rule : inputPortMap) {
        auto original_dims = sliced_input_dims(getParentEdgesAtPort(map_rule.from)[0]->getMemoryPtr(), map_rule.axis, map_rule.stride);
        auto &to_mems = input_mems[map_rule.to];
        const auto& body_inshape = to_mems.front()->GetShape();
        if (body_inshape.isDynamic() || body_inshape.getDims() != original_dims) {
            return true;
        }
    }

    return false;
}

int TensorIterator::getNumIteration(const std::vector<PortMap>& inputPortMap, const std::vector<PortMap>& outputPortMap) const {
    const auto isIterable = [](const PortMap& rule) {
        return rule.axis != -1;
    };

    const auto getNumIterations = [this](const PortMap& rule, const std::vector<size_t>& dimensions) -> int {
        const auto axis = rule.axis;
        if (axis < 0 || static_cast<std::size_t>(axis) >= dimensions.size()) {
            THROW_ERROR << ": Invalid \"axis\" value in an iteration component: "
                        << rule.axis  << ", dimensions number = " << dimensions.size() << " (out of range)";
        }
        const auto space = dimensions[axis];
        const int start = static_cast<int>((rule.start < 0 ? (space + 1) : 0) + rule.start);
        const int end   = static_cast<int>((rule.end   < 0 ? (space + 1) : 0) + rule.end);

        const auto stride = rule.stride;
        if (stride == 0) {
            THROW_ERROR << ": Invalid \"stride\" value in an iteration component: " << rule.stride << " (infinite loop)";
        }
        const auto step = std::abs(stride);

        const auto src = stride < 0 ? end : start;
        const auto dst = stride < 0 ? start : end;
        const auto length = dst - src;
        if (src < 0 || src >= dst || dst > static_cast<int64_t>(space) || length < step) {
            THROW_ERROR << ": Invalid \"start\",\"stride\",\"end\" values in an iteration component"
                        << ": \"start\" = " << rule.start << ", \"stride\" = " << rule.stride  << ", \"end\" = " << rule.end;
        }

        if (length % step != 0) {
            THROW_ERROR << ": Each iteration must be the same size: length (" << length << ") is not divisible by step (" << step << ")";
        }

        return static_cast<int>(length / step);
    };


    int numIterations = 1;
    bool isDefault = true;
    for (const auto& rule : inputPortMap) {
        const auto& dims = getParentEdgesAtPort(rule.from)[0]->getMemoryPtr()->getStaticDims();
        if (!isIterable(rule)) {
            continue;
        }

        if (rule.from < 0 || rule.from >= static_cast<int64_t>(inputShapes.size())) {
            THROW_ERROR << ": Invalid \"from\" value: \"from\" = " << rule.from
                        << " inputs number = " << inputShapes.size() << " (out of range)";
        }

        const auto currentNumIterations = getNumIterations(rule, dims);
        if (isDefault) {
            isDefault = false;
            numIterations = currentNumIterations;
        } else if (numIterations != currentNumIterations) {
            THROW_ERROR << ": There are at least two different iterations numbers: " << numIterations << " and " << currentNumIterations;
        }
    }

    for (const auto& rule : outputPortMap) {
        const auto& dims = getBaseMemDescAtOutputPort(rule.from)->getShape().getDims();
        if (!isIterable(rule)) {
            continue;
        }

        if (dims[rule.axis] == Shape::UNDEFINED_DIM)
            continue;

        if (rule.from < 0 || rule.from >= static_cast<int64_t>(outputShapes.size())) {
            THROW_ERROR << ": Invalid \"from\" value: \"from\" = " << rule.from
                        << " inputs number = " << outputShapes.size() << " (out of range)";
        }

        const auto currentNumIterations = getNumIterations(rule, dims);
        if (isDefault) {
            isDefault = false;
            numIterations = currentNumIterations;
        } else if (numIterations != currentNumIterations) {
            THROW_ERROR << ": There are at least two different iterations numbers: " << numIterations << " and " << currentNumIterations;
        }
    }

    return numIterations;
}

bool TensorIterator::created() const {
    return getType() == Type::TensorIterator;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
