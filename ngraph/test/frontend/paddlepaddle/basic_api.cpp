// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <frontend_manager/frontend_manager.hpp>
#include <paddlepaddle_frontend/frontend.hpp>

#include "gtest/gtest.h"

using namespace ngraph;
using namespace ngraph::frontend;

static const auto PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using BasicTestParam = std::string;

std::string getTestCaseName(const testing::TestParamInfo<BasicTestParam> &obj) {
    return obj.param;
}

class PDPDBasicTest : public ::testing::TestWithParam<BasicTestParam> {
public:
    std::string                      m_modelFile;
    FrontEndManager                  m_fem;
    std::shared_ptr<FrontEndPDPD>    m_frontEnd;
    InputModel::Ptr                  m_inputModel;

    void initParamTest() {
        m_modelFile = std::string(TEST_FILES) + PATH_TO_MODELS + GetParam();
        std::cout << "model file:" << m_modelFile << std::endl;
    }

    void SetUp() override {
        initParamTest();
    }

    void TearDown() override {
    }

    std::shared_ptr<FrontEndPDPD> doLoadFrontEnd() {
        auto frontends = m_fem.availableFrontEnds();
        auto fe = m_fem.loadByFramework(PDPD);
        return std::dynamic_pointer_cast<FrontEndPDPD>(fe);
    }
};

TEST_P(PDPDBasicTest, testLoadFromFile)
{
    ASSERT_NO_THROW(m_frontEnd = doLoadFrontEnd());
    ASSERT_NE(m_frontEnd, nullptr);
    ASSERT_NO_THROW(m_inputModel = m_frontEnd->loadFromFile(m_modelFile));
    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}

static const std::vector<std::string> models {
        std::string("conv2d"),
};

INSTANTIATE_TEST_CASE_P(PDPDBasicTest, PDPDBasicTest,
                        ::testing::ValuesIn(models),
                        getTestCaseName);