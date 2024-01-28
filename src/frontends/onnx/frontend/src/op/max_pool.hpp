// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
///
/// \brief Convert ONNX MaxPool operation to an OV node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing OV nodes producing output of ONNX MaxPool
///         operation.
///
ov::OutputVector max_pool(const Node& node);

}  // namespace set_1

namespace set_8 {
///
/// \brief Convert ONNX MaxPool operation to an OV node.
///
/// \param node   The ONNX node object representing this operation.
///
/// \return The vector containing OV nodes producing output of ONNX MaxPool
///         operation.
///
ov::OutputVector max_pool(const Node& node);

}  // namespace set_8

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
