// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fully_connected_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>


#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_lpt_models/mat_mul.hpp"

namespace LayerTestsDefinitions {

std::string FullyConnectedTransformation::getTestCaseName(const testing::TestParamInfo<FullyConnectedTransformationParams>& obj) {
    ov::element::Type precision;
    MatMulShapes shapes;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(precision, shapes, targetDevice, params) = obj.param;

    std::ostringstream result;
    result <<
           get_test_case_name_by_params(precision, shapes.inputA, targetDevice, params) <<
           shapes.inputB << "_" <<
        shapes.transposeA << "_" <<
        shapes.transposeB;

    return result.str();
}

void FullyConnectedTransformation::SetUp() {
    abs_threshold = 0.6;

    ov::element::Type precision;
    MatMulShapes shapes;
    ov::pass::low_precision::LayerTransformation::Params params;
    std::tie(precision, shapes, targetDevice, params) = this->GetParam();

    init_input_shapes({ shapes.inputA, shapes.inputB });

    function = ov::builder::subgraph::MatMulFunction::getOriginal(
        precision,
        shapes.inputA,
        shapes.inputB,
        shapes.transposeA,
        shapes.transposeB);
}

TEST_P(FullyConnectedTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

}  // namespace LayerTestsDefinitions
