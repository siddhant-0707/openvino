// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/dropout.hpp"

#include "exceptions.hpp"
#include "onnx_import/core/null_node.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/op_types.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace {
ov::OutputVector build_dropout(const Node& node, bool training_mode) {
    CHECK_VALID_NODE(node, !training_mode, "Training mode is not supported for Dropout op");

    const auto input_data = node.get_ng_inputs().at(0);
    const bool return_mask = node.get_outputs_size() > 1;

    if (return_mask) {
        const auto mask = std::make_shared<v3::Broadcast>(v0::Constant::create(ov::element::boolean, Shape{}, {true}),
                                                          std::make_shared<v3::ShapeOf>(input_data));
        return {input_data, mask};
    } else {
        return {input_data};
    }
}
}  // namespace

namespace set_12 {
ov::OutputVector dropout(const Node& node) {
    const auto ng_inputs = node.get_ng_inputs();
    // seed attribute and ratio input are ignored because traning mode is not
    // supported anyway
    bool training_mode = false;  // default value
    if (ng_inputs.size() > 2 && !ov::op::util::is_null(ng_inputs.at(2))) {
        CHECK_VALID_NODE(node,
                         ov::op::util::is_constant(ng_inputs.at(2).get_node_shared_ptr()),
                         "Non-constant training_mode input is not supported.");
        training_mode = ov::as_type_ptr<v0::Constant>(ng_inputs.at(2).get_node_shared_ptr())->cast_vector<bool>()[0];
    }
    return build_dropout(node, training_mode);
}
}  // namespace set_12

namespace set_7 {
ov::OutputVector dropout(const Node& node) {
    // "is_test" attribute was removed
    // ratio attribute is ignored because traning mode is not supported
    const bool training_mode = false;

    return build_dropout(node, training_mode);
}
}  // namespace set_7

namespace set_1 {
ov::OutputVector dropout(const Node& node) {
    // legacy consumed_inputs attribute ignored
    // ratio attribute is ignored because traning mode is not supported
    const bool training_mode = !node.get_attribute_value<int64_t>("is_test", 0);

    return build_dropout(node, training_mode);
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
