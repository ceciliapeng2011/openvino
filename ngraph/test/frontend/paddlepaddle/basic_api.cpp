// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <regex>
#include <frontend_manager/frontend_manager.hpp>
#include <paddlepaddle_frontend/frontend.hpp>

#include <gtest/gtest.h>

using namespace ngraph;
using namespace ngraph::frontend;

static const auto PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using BasicTestParam = std::string;

std::string getTestCaseName(const testing::TestParamInfo<BasicTestParam> &obj) {
    auto fileName = obj.param;
    // need to replace special characters to create valid test case name
    fileName = std::regex_replace(fileName, std::regex("[/\\.]"), "_");
    std::cout << "Test case name: " << fileName << std::endl;
    return fileName;
}

class PDPDBasicTest : public ::testing::TestWithParam<BasicTestParam> {
public:
    std::string                      m_modelFile;
    FrontEndManager                  m_fem;
    std::shared_ptr<FrontEndPDPD>    m_frontEnd;
    std::shared_ptr<InputModelPDPD>  m_inputModel;

    void initParamTest() {
        m_modelFile = std::string(TEST_FILES) + PATH_TO_MODELS + GetParam();
        std::cout << "Model: " << m_modelFile << std::endl;
    }

    void SetUp() override {
        initParamTest();
    }

    void TearDown() override {
    }

    void doLoadFromFile() {
        std::vector<std::string> frontends;
        FrontEnd::Ptr fe;
        ASSERT_NO_THROW(frontends = m_fem.availableFrontEnds());
        ASSERT_NO_THROW(fe = m_fem.loadByFramework(PDPD));
        ASSERT_NO_THROW(m_frontEnd = std::dynamic_pointer_cast<FrontEndPDPD>(fe));
        ASSERT_NE(m_frontEnd, nullptr);
        ASSERT_NO_THROW(m_inputModel = std::dynamic_pointer_cast<InputModelPDPD>(m_frontEnd->loadFromFile(m_modelFile)));
        ASSERT_NE(m_inputModel, nullptr);
    }
};

TEST_P(PDPDBasicTest, testLoadFromFile)
{
    ASSERT_NO_THROW(doLoadFromFile());
    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
}

TEST_P(PDPDBasicTest, testInputModel_getInputs)
{
    ASSERT_NO_THROW(doLoadFromFile());

    std::vector<Place::Ptr> inputs;
    ASSERT_NO_THROW(inputs = m_inputModel->getInputs());
    EXPECT_GT(inputs.size(), 0);
    std::set<Place::Ptr> inputsSet(inputs.begin(), inputs.end());
    EXPECT_EQ(inputsSet.size(), inputs.size());
    std::for_each(inputs.begin(), inputs.end(), [](Place::Ptr place) {
        ASSERT_NE(place, nullptr);
        EXPECT_GT(place->getNames().size(), 0);
        //TODO: shall we verify that each 'place->isInput() returns true'?
    });
}

TEST_P(PDPDBasicTest, testInputModel_getOutputs)
{
    ASSERT_NO_THROW(doLoadFromFile());

    std::vector<Place::Ptr> outputs;
    ASSERT_NO_THROW(outputs = m_inputModel->getOutputs());
    EXPECT_GT(outputs.size(), 0);
    std::set<Place::Ptr> outputsSet(outputs.begin(), outputs.end());
    EXPECT_EQ(outputsSet.size(), outputs.size());
    std::for_each(outputs.begin(), outputs.end(), [](Place::Ptr place) {
        EXPECT_NE(place, nullptr);
        EXPECT_GT(place->getNames().size(), 0);
        //TODO: shall we verify that each 'place->isOutput() returns true'?
    });
}

TEST_P(PDPDBasicTest, DISABLED_testInputModel_getPlaceByTensorName)
{
    ASSERT_NO_THROW(doLoadFromFile());

    // TODO: not clear now
}

TEST_P(PDPDBasicTest, testInputModel_overrideAllOutputs)
{
    ASSERT_NO_THROW(doLoadFromFile());

    std::vector<Place::Ptr> outputs;
    ASSERT_NO_THROW(outputs = m_inputModel->getOutputs());
    std::set<Place::Ptr> outputsSet(outputs.begin(), outputs.end());

    auto outputsReversed = outputs;
    std::reverse(outputsReversed.begin(), outputsReversed.end());
    ASSERT_NO_THROW(m_inputModel->overrideAllOutputs(outputsReversed));
    ASSERT_NO_THROW(outputs = m_inputModel->getOutputs());
    EXPECT_GT(outputs.size(), 0);
    std::set<Place::Ptr> outputsSetAfter(outputs.begin(), outputs.end());
    EXPECT_EQ(outputsSet.size(), outputsSetAfter.size());
    std::for_each(outputs.begin(), outputs.end(), [&](Place::Ptr place) {
        EXPECT_GT(outputsSet.count(place), 0);
    });
}

TEST_P(PDPDBasicTest, testInputModel_overrideAllOutputs_empty)
{
    ASSERT_NO_THROW(doLoadFromFile());

    std::vector<Place::Ptr> outputs;
    ASSERT_NO_THROW(outputs = m_inputModel->getOutputs());
    ASSERT_NO_THROW(m_inputModel->overrideAllOutputs({}));
    ASSERT_NO_THROW(outputs = m_inputModel->getOutputs());
    ASSERT_EQ(outputs.size(), 0);
    // TODO: should we check that all previous outputs will return false to Place::isOutput()?
}

TEST_P(PDPDBasicTest, testInputModel_overrideAllInputs)
{
    ASSERT_NO_THROW(doLoadFromFile());

    std::vector<Place::Ptr> inputs;
    ASSERT_NO_THROW(inputs = m_inputModel->getInputs());
    std::set<Place::Ptr> inputsSet(inputs.begin(), inputs.end());

    auto inputsReversed = inputs;
    std::reverse(inputsReversed.begin(), inputsReversed.end());
    ASSERT_NO_THROW(m_inputModel->overrideAllInputs(inputsReversed));
    ASSERT_NO_THROW(inputs = m_inputModel->getInputs());
    EXPECT_GT(inputs.size(), 0);
    std::set<Place::Ptr> inputsSetAfter(inputs.begin(), inputs.end());
    EXPECT_EQ(inputsSet.size(), inputsSetAfter.size());
    std::for_each(inputs.begin(), inputs.end(), [&](Place::Ptr place) {
        EXPECT_GT(inputsSet.count(place), 0);
    });
}

TEST_P(PDPDBasicTest, testInputModel_overrideAllInputs_empty)
{
    ASSERT_NO_THROW(doLoadFromFile());

    std::vector<Place::Ptr> inputs;
    ASSERT_NO_THROW(inputs = m_inputModel->getInputs());
    ASSERT_NO_THROW(m_inputModel->overrideAllInputs({}));
    ASSERT_NO_THROW(inputs = m_inputModel->getInputs());
    ASSERT_EQ(inputs.size(), 0);
    // TODO: should we check that all previous outputs will return false to Place::isInput()?
}

TEST_P(PDPDBasicTest, DISABLED_testInputModel_extractSubgraph)
{
    ASSERT_NO_THROW(doLoadFromFile());

    // TODO: not clear now
}

TEST_P(PDPDBasicTest, DISABLED_testInputModel_setPartialShape)
{
    ASSERT_NO_THROW(doLoadFromFile());

    // TODO: not clear now
}

static const std::vector<std::string> models {
        std::string("conv2d"),
        std::string("conv2d_s/conv2d.pdmodel"),
};

INSTANTIATE_TEST_CASE_P(PDPDBasicTest, PDPDBasicTest,
                        ::testing::ValuesIn(models),
                        getTestCaseName);