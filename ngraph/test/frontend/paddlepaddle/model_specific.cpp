// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <frontend_manager/frontend_manager.hpp>
#include <paddlepaddle_frontend/frontend.hpp>

#include <gtest/gtest.h>

using namespace ngraph;
using namespace ngraph::frontend;

static const auto PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

class PDPDModelsTest : public ::testing::Test {
public:
    std::string                      m_modelFile;
    FrontEndManager                  m_fem;
    std::shared_ptr<FrontEndPDPD>    m_frontEnd;
    std::shared_ptr<InputModelPDPD>  m_inputModel;

    void SetUp() override {
    }

    void TearDown() override {
    }

    void doLoadFromFile(const std::string& modelFile) {
        m_modelFile = std::string(TEST_FILES) + PATH_TO_MODELS + modelFile;
        std::vector<std::string> frontends;
        FrontEnd::Ptr fe;
        ASSERT_NO_THROW(frontends = m_fem.availableFrontEnds());
        ASSERT_NO_THROW(fe = m_fem.loadByFramework(PDPD));
        ASSERT_NO_THROW(m_frontEnd = std::dynamic_pointer_cast<FrontEndPDPD>(fe));
        ASSERT_NE(m_frontEnd, nullptr);
        ASSERT_NO_THROW(m_inputModel = std::dynamic_pointer_cast<InputModelPDPD>(
                m_frontEnd->loadFromFile(m_modelFile)));
        ASSERT_NE(m_inputModel, nullptr);
    }
};

TEST_F(PDPDModelsTest, testConv2D)
{
    ASSERT_NO_THROW(doLoadFromFile("conv2d"));
    // TODO: check all inputs/outputs for conv2d
    ASSERT_NO_THROW(m_inputModel->getInputs());
    ASSERT_EQ(m_inputModel->getInputs().size(), 1);
    ASSERT_EQ(m_inputModel->getOutputs().size(), 1);
}
