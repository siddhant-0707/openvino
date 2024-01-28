// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/cos.hpp"

#include "openvino/op/cos.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
ov::OutputVector cos(const Node& node) {
    return {std::make_shared<v0::Cos>(node.get_ng_inputs().at(0))};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
