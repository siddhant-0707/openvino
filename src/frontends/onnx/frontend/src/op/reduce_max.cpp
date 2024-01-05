// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/softmax.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/validation_util.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector reduce_max(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);
    // to be done
}
}  // namespace set_1
namespace set_11 {
OutputVector reduce_max(const Node& node) {
    // check types, allowing an extra one
    const auto data = node.get_ng_inputs().at(0);
    const auto axes = node.get_ng_inputs().at(1);

    const auto keep_dims = node.get_attribute_value<bool>("keep_dims", 0);

    return {std::make_shared<ov::op::v1::ReduceMax>(data, axes, keep_dims)};
}
}  // namespace set_11
namespace set_12 {
OutputVector reduce_max(const Node& node) {
    // 
    const auto data = node.get_ng_inputs().at(0);
    const auto axes = node.get_ng_inputs().at(1);

    const auto keep_dims = node.get_attribute_value<bool>("keep_dims", 0);

    return {std::make_shared<ov::op::v1::ReduceMax>(data, axes, keep_dims)};
}
}  // namespace set_12
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
