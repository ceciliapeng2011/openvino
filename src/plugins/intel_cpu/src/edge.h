// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_blob.h>
#include "cpu_shape.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/node_config.h"
#include "weights_cache.hpp"

#include <map>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {

class Node;
class Edge;

using EdgePtr = std::shared_ptr<Edge>;
using EdgeWeakPtr = std::weak_ptr<Edge>;

class Edge {
public:
    Edge(const std::shared_ptr<Node>& parent,
         const std::shared_ptr<Node>& child,
         int pr_port = 0, int ch_port = 0);

    enum class Status {
        Uninitialized,
        NeedAllocation,
        NotAllocated,
        Allocated,
        Validated
    };

    enum class ReorderStatus {
        Regular = 0,
        Optimized = 1,
        No = 2
    };

    enum LOOK { LOOK_UP = 1, LOOK_DOWN = 2, LOOK_BOTH = LOOK_UP | LOOK_DOWN };

    inline Status getStatus() const noexcept {
        return status;
    }

    void changeStatus(Status state);
    bool inPlace(LOOK look = LOOK_BOTH) const;

    void init();
    void allocate(const void* mem_ptr = nullptr);
    void allocate(MemoryMngrPtr memMngr);
    void externalAllocate(WeightsSharing::Ptr weightsCache);
    void reuse(MemoryPtr ptr);
    void validate();
    void drop();

    const std::shared_ptr<Node> getParent() const;
    const std::shared_ptr<Node> getChild() const;

    const Memory& getMemory();
    MemoryPtr getMemoryPtr() const;
    void resetMemoryPtr(MemoryPtr mem);

    ReorderStatus needReorder();
    bool isDropped() const;
    bool isUseExternalMemory() const;

    int getInputNum() const;
    int getOutputNum() const;

    void setChildPort(const size_t port) { child_port = port; }

    void sharedMemFrom(const EdgePtr& edge);
    EdgePtr getSharedEdge() const;
    EdgePtr getSharedEdge(std::nothrow_t) const;

    bool hasDefinedMaxSize() const {
        return getDesc().hasDefinedMaxSize();
    }

    std::string name() const;

private:
    std::weak_ptr<Node> parent;
    std::weak_ptr<Node> child;
    int parent_port;
    int child_port;

    bool useExternalMemory = false;
    EdgeWeakPtr memoryFromEdge;
    MemoryPtr memoryPtr;
    Status status = Status::Uninitialized;

    const MemoryDesc& getInputDesc() const;
    const MemoryDesc& getOutputDesc() const;
    PortDescBaseCPtr getInputPortDesc() const;
    PortDescBaseCPtr getOutputPortDesc() const;

    const MemoryDesc& getDesc() const;
    bool enforceReorder();

    void collectConsumers(std::vector<std::shared_ptr<Node>>& result) const;

    EdgePtr getBaseEdge(int look = LOOK_BOTH);
    void allocateCommon(const std::function<void(const MemoryPtr&, const MemoryDesc&)>& allocate);

    friend class Graph;
};

}   // namespace intel_cpu
}   // namespace ov

