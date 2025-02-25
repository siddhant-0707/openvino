// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/single_layer/adaptive_pooling.hpp"

using namespace InferenceEngine;
using namespace FuncTestUtils::PrecisionUtils;

namespace LayerTestsDefinitions {

std::string AdaPoolLayerTest::getTestCaseName(const testing::TestParamInfo<adapoolParams>& obj) {
    std::vector<size_t> inputShape;
    std::vector<int> pooledSpatialShape;

    std::string poolingMode;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(inputShape, pooledSpatialShape, poolingMode, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;

    result << "in_shape=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "pooled_spatial_shape=" << ov::test::utils::vec2str(pooledSpatialShape) << "_";
    result << "mode=" << poolingMode << "_";
    result << "prec=" << netPrecision.name() << "_";
    result << "dev=" << targetDevice;
    return result.str();
}

void AdaPoolLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<int> pooledSpatialShape;
    std::string poolingMode;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, pooledSpatialShape, poolingMode, netPrecision, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    ov::Shape pooledShape = {pooledSpatialShape.size() };
    auto pooledParam = ov::test::utils::deprecated::make_constant<int32_t>(ov::element::i32, pooledShape, pooledSpatialShape);

    // we cannot create abstract Op to use polymorphism
    auto adapoolMax = std::make_shared<ov::op::v8::AdaptiveMaxPool>(params[0], pooledParam, ov::element::i32);
    auto adapoolAvg = std::make_shared<ov::op::v8::AdaptiveAvgPool>(params[0], pooledParam);

    function = (poolingMode == "max" ? std::make_shared<ov::Model>(adapoolMax->outputs(), params, "AdaPoolMax") :
                std::make_shared<ov::Model>(adapoolAvg->outputs(), params, "AdaPoolAvg"));
}
}  // namespace LayerTestsDefinitions
