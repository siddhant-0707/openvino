// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector reduce_max(const Node& node);

}  // namespace set_1

namespace set_11 {
OutputVector reduce_max(const Node& node);

}  // namespace set_11

namespace set_12 {
OutputVector reduce_max(const Node& node);

}  // namespace set_12

namespace set_13 {
OutputVector reduce_max(const Node& node);

}  // namespace set_13

namespace set_18 {
OutputVector reduce_max(const Node& node);

}  // namespace set_18

namespace set_20 {
OutputVector reduce_max(const Node& node);

}  // namespace set_20
}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
