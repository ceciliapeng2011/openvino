// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <future>

#include "behavior/ov_infer_request/cancellation.hpp"
#include "openvino/runtime/exception.hpp"

namespace ov {
namespace test {
namespace behavior {

std::string OVInferRequestCancellationTests::getTestCaseName(const testing::TestParamInfo<InferRequestParams>& obj) {
    return OVInferRequestTests::getTestCaseName(obj);
}

TEST_P(OVInferRequestCancellationTests, canCancelAsyncRequest) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.cancel());
    try {
        req.wait();
    } catch (const ov::Cancelled&) {
        SUCCEED();
    }
}

TEST_P(OVInferRequestCancellationTests, CanResetAfterCancelAsyncRequest) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.cancel());
    try {
        req.wait();
    } catch (const ov::Cancelled&) {
        SUCCEED();
    }
    OV_ASSERT_NO_THROW(req.start_async());
    OV_ASSERT_NO_THROW(req.wait());
}

TEST_P(OVInferRequestCancellationTests, canCancelBeforeAsyncRequest) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    OV_ASSERT_NO_THROW(req.cancel());
}

TEST_P(OVInferRequestCancellationTests, canCancelInferRequest) {
    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
    auto infer = std::async(std::launch::async, [&req]{req.infer();});
    while (!req.wait_for({})) {
    }
    OV_ASSERT_NO_THROW(req.cancel());
    try {
        infer.get();
    } catch (const ov::Cancelled&) {
        SUCCEED();
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
