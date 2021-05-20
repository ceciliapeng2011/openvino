// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "ngraph/check.hpp"

namespace ngraph {
namespace frontend {

class GeneralFailure : public CheckFailure {
public:
    GeneralFailure(const CheckLocInfo &check_loc_info,
                 const std::string &context,
                 const std::string &explanation)
    : CheckFailure(check_loc_info, "FrontEnd API failed with GeneralFailure" + context, explanation) {
    }
};

class InitializationFailure : public CheckFailure {
public:
    InitializationFailure(const CheckLocInfo &check_loc_info,
                 const std::string &context,
                 const std::string &explanation)
            : CheckFailure(check_loc_info, "FrontEnd API failed with InitializationFailure" + context, explanation) {
    }
};

class OpValidationFailure : public CheckFailure {
public:
    OpValidationFailure(const CheckLocInfo &check_loc_info,
                      const std::string &context,
                      const std::string &explanation)
            : CheckFailure(check_loc_info, "FrontEnd API failed with OpValidationFailure" + context, explanation) {
    }
};

class OpConversionFailure : public CheckFailure {
public:
    OpConversionFailure(const CheckLocInfo &check_loc_info,
                      const std::string &context,
                      const std::string &explanation)
            : CheckFailure(check_loc_info, "FrontEnd API failed with OpConversionFailure" + context, explanation) {
    }
};

class NotImplementedFailure : public CheckFailure {
public:
    NotImplementedFailure(const CheckLocInfo &check_loc_info,
                        const std::string &context,
                        const std::string &explanation)
            : CheckFailure(check_loc_info, "FrontEnd API failed with NotImplementedFailure" + context, explanation) {
    }
};

/// \brief Macro to check whether a boolean condition holds.
/// \param cond Condition to check
/// \param ... Additional error message info to be added to the error message via the `<<`
///            stream-insertion operator. Note that the expressions here will be evaluated lazily,
///            i.e., only if the `cond` evalutes to `false`.
/// \throws ::ngraph::CheckFailurePDPD if `cond` is false.
#define FRONT_END_GENERAL_CHECK(...) NGRAPH_CHECK_HELPER(::ngraph::frontend::GeneralFailure, "", __VA_ARGS__)
#define FRONT_END_INITIALIZATION_CHECK(...) NGRAPH_CHECK_HELPER(::ngraph::frontend::InitializationFailure, "", __VA_ARGS__)
#define FRONT_END_NOT_IMPLEMENTED(NAME) NGRAPH_CHECK_HELPER(::ngraph::frontend::NotImplementedFailure, "", false, #NAME" is not implemented for this FrontEnd class")
#define FRONT_END_THROW(MSG) FRONT_END_GENERAL_CHECK(false, MSG)

} // frontend
} // ngraph