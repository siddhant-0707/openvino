// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/reorg_yolo.hpp"

namespace LayerTestsDefinitions {

std::string ReorgYoloLayerTest::getTestCaseName(const testing::TestParamInfo<ReorgYoloParamsTuple> &obj) {
    ov::Shape inputShape;
    size_t stride;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::tie(inputShape, stride, netPrecision, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << inputShape << "_";
    result << "stride=" << stride << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetName << "_";
    return result.str();
}

void ReorgYoloLayerTest::SetUp() {
    ov::Shape inputShape;
    size_t stride;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, stride, netPrecision, targetDevice) = this->GetParam();
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    auto reorg_yolo = std::make_shared<ov::op::v0::ReorgYolo>(param, stride);
    function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(reorg_yolo), ov::ParameterVector{param}, "ReorgYolo");
}

} // namespace LayerTestsDefinitions
