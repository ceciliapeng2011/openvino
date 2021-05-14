// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "node_context.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
	
NamedOutputs reshape2(const NodeContext& node);

} // namespace op
} // namespace pdpd
} // namespace frontend
} // namespace ngraph