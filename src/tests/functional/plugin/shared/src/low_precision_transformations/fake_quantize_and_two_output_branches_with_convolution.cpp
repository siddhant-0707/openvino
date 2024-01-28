// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fake_quantize_and_two_output_branches_with_convolution.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ov_models/pass/convert_prc.hpp"

namespace LayerTestsDefinitions {

std::string FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::getTestCaseName(
    const testing::TestParamInfo<FakeQuantizeAndTwoOutputBranchesWithConvolutionParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeAndTwoOutputBranchesWithConvolution testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = obj.param;

    std::ostringstream result;
    result << netPrecision << "_" << inputShape << "_"
           << targetDevice << "_" << testValues.fqOnData << "_"
           << testValues.fqOnWeights1 << "_" << testValues.fqOnWeights2;
    return result.str();
}

void FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation::SetUp() {
    rel_threshold = 0.1;
    abs_threshold = 0.1;

    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    FakeQuantizeAndTwoOutputBranchesWithConvolution testValues;
    std::tie(netPrecision, inputShape, targetDevice, params, testValues) = this->GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::FakeQuantizeAndTwoOutputBranchesWithConvolutionFunction::getOriginal(
        netPrecision,
        inputShape,
        testValues.fqOnData,
        testValues.fqOnWeights1,
        testValues.fqOnWeights2);
}

TEST_P(FakeQuantizeAndTwoOutputBranchesWithConvolutionTransformation, CompareWithRefImpl) {
    run();
};

}  // namespace LayerTestsDefinitions
