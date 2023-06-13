// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "output_mem_mgr.h"
#include "utils/debug_capabilities.h"

using namespace ov::intel_cpu;

void* OutputMemoryMngr::getRawPtr() const noexcept {
    if (m_allocator) {
        return m_allocator->getData();
    } else {
        return _pMemMngr->getRawPtr();
    }
}

void OutputMemoryMngr::setExtBuff(void* ptr, size_t size) {
    DEBUG_LOG(ptr, "_", size);
    if (m_allocator) {
        return;
    } else {
        return _pMemMngr->setExtBuff(ptr, size);
    }
}

bool OutputMemoryMngr::resize(size_t size) {
    if (m_allocator) {
        constexpr int cacheLineSize = 64;
        bool sizeChanged = true;
        m_allocator->allocate(size, cacheLineSize);
        DEBUG_LOG(sizeChanged);
        return sizeChanged;
    } else {
        bool sizeChanged = false;
        sizeChanged = _pMemMngr->resize(size);
        DEBUG_LOG(sizeChanged);
        return sizeChanged;
    }
}

bool OutputMemoryMngr::hasExtBuffer() const noexcept {
    return true;
}

void OutputMemoryMngr::setMemDesc(MemoryDescPtr desc) {
    if (m_allocator) m_allocator->setMemDesc(desc);
}

void* OutputAllocator::allocate(const size_t bytes, const size_t alignment) {
    (void)alignment;
    const auto actualDesc = MemoryDescUtils::convertToTensorDesc(*m_memDesc.get());
    IE_ASSERT(m_memDesc->getCurrentMemSize()==bytes);

    auto &currentDesc = m_blob->getTensorDesc();
    const auto outDims = actualDesc.getDims();
    if (currentDesc.getDims() != outDims) {
        // WA: because input/output info initially contains non empty dims, order etc.
        // and setDims (called inside setShape) can't correct modify blocked desc for desc with blocked layout
        if (currentDesc.getLayout() == InferenceEngine::Layout::BLOCKED) {
            currentDesc = InferenceEngine::TensorDesc(currentDesc.getPrecision(), currentDesc.getLayout());
        }
        m_blob->setShape(outDims);
    }
    return m_blob->buffer();
}

void OutputAllocator::setMemDesc(MemoryDescPtr desc) {
    m_memDesc = desc;
    return;
}

void* OutputAllocator::getData() const noexcept {
    return m_blob->buffer();
}