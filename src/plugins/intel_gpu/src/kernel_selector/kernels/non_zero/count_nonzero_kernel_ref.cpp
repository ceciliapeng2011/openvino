﻿// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "count_nonzero_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey CountNonzeroKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::UINT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::UINT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

CountNonzeroKernelRef::DispatchData CountNonzeroKernelRef::SetDefault(const count_nonzero_params& params) const {
    DispatchData dispatchData;
    const auto& input = params.inputs[0];

    // Set 1 work group to avoid synchornization issue for summation of nonzero counting.
    dispatchData.dataSize = input.LogicalSize();
    size_t max_dim_size = (dispatchData.dataSize > params.engineInfo.maxWorkGroupSize) ?
                                    params.engineInfo.maxWorkGroupSize : dispatchData.dataSize;
    dispatchData.lws = dispatchData.gws = { max_dim_size, 1, 1};

    return dispatchData;
}

DeviceFeaturesKey CountNonzeroKernelRef::get_required_device_features_key(const Params& params, const optional_params& options) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_reduce();

    return k;
}

KernelsData CountNonzeroKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::COUNT_NONZERO);

    KernelData kd = KernelData::Default<count_nonzero_params>(params);
    count_nonzero_params& newParams = *static_cast<count_nonzero_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = MakeBaseParamsJitConstants(newParams);
    if (newParams.has_dynamic_tensors()) {
        const auto& input = newParams.inputs[0];
        auto x = toCodeString(input.X(), 5);
        auto y = toCodeString(input.Y(), 4);
        auto z = toCodeString(input.Z(), 3);
        auto w = toCodeString(input.W(), 2);
        auto f = toCodeString(input.Feature(), 1);
        auto b = toCodeString(input.Batch(), 0);
        cldnn_jit.AddConstants({
            MakeJitConstant("DATA_SIZE", toVectorMulString({x, y, z, w, f, b}))
        });
    } else {
        cldnn_jit.AddConstants({
            MakeJitConstant("DATA_SIZE", dispatchData.dataSize)
        });
    }
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const count_nonzero_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
    };

    // In case of count-nonzero, the output shape is static unconditionally,
    // so it should be checked as dynamic of the input shape
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     newParams.inputs[0].is_dynamic());

    return {kd};
}

KernelsPriority CountNonzeroKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool CountNonzeroKernelRef::Validate(const Params& p, const optional_params& op) const {
    if (!KernelBaseOpenCL::Validate(p, op))
        return false;

    const auto& rp = static_cast<const count_nonzero_params&>(p);

    return Tensor::SimpleLayout(rp.inputs[0].GetLayout());
}
}  // namespace kernel_selector
