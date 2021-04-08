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
    ASSERT_NO_THROW(m_inputModel->getInputs());
    ASSERT_NO_THROW(m_inputModel->getOutputs());
    ASSERT_EQ(m_inputModel->getInputs().size(), 1);
    ASSERT_EQ(m_inputModel->getOutputs().size(), 1);

    auto input = m_inputModel->getInputs().front();
    auto output = m_inputModel->getOutputs().front();
    ASSERT_TRUE(input->isInput());
    ASSERT_FALSE(input->isOutput());
    ASSERT_FALSE(output->isInput());
    ASSERT_TRUE(output->isOutput());
    ASSERT_FALSE(input->isEqual(output));
    ASSERT_FALSE(output->isEqual(input));

    // Verify input is 'x' (see generating script)
    ASSERT_EQ(input->getNames().size(), 1);
    auto inputName = input->getNames().front();
    EXPECT_EQ(inputName, "x");

    // Verify output name
    ASSERT_EQ(output->getNames().size(), 1);
    auto outputName = output->getNames().front();
    EXPECT_EQ(outputName, "save_infer_model/scale_0.tmp_0");
}

TEST_F(PDPDModelsTest, testConv2D_relu_cut_outputs)
{
    ASSERT_NO_THROW(doLoadFromFile("conv2d_relu/conv2d_relu.pdmodel"));
    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    // Ensure that it contains 'relu'
    EXPECT_TRUE(std::find_if(function->get_ordered_ops().begin(), function->get_ordered_ops().end(),
                             [&](const std::shared_ptr<ngraph::Node>& node) {
                                 return node->get_friendly_name().find("relu") != std::string::npos;
                             }) != function->get_ordered_ops().end());

    Place::Ptr place;
    ASSERT_NO_THROW(place = m_inputModel->getPlaceByTensorName("conv2d_0.tmp_0"));
    ASSERT_NE(place, nullptr);
    ASSERT_NO_THROW(m_inputModel->overrideAllOutputs({place}));

    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    // Verify that after override nGraph function doesn't have 'relu' anymore
    EXPECT_FALSE(std::find_if(function->get_ordered_ops().begin(), function->get_ordered_ops().end(),
                              [&](const std::shared_ptr<ngraph::Node>& node) {
                                  return node->get_friendly_name().find("relu") != std::string::npos;
                              }) != function->get_ordered_ops().end());
}

TEST_F(PDPDModelsTest, testConv2D_conv2d_cut_inputs)
{
    ASSERT_NO_THROW(doLoadFromFile("conv2d_relu/conv2d_relu.pdmodel"));
    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    // Ensure that it contains 'conv2d'
    EXPECT_TRUE(std::find_if(function->get_ordered_ops().begin(), function->get_ordered_ops().end(),
                             [&](const std::shared_ptr<ngraph::Node>& node) {
                                 return node->get_friendly_name().find("conv2d") != std::string::npos;
                             }) != function->get_ordered_ops().end());

    Place::Ptr place;
    ASSERT_NO_THROW(place = m_inputModel->getPlaceByTensorName("relu_0.tmp_0"));
    ASSERT_NE(place, nullptr);
    ASSERT_NO_THROW(m_inputModel->overrideAllInputs({place}));

    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    // Verify that after override nGraph function doesn't have 'conv2d' anymore
    EXPECT_FALSE(std::find_if(function->get_ordered_ops().begin(), function->get_ordered_ops().end(),
                              [&](const std::shared_ptr<ngraph::Node>& node) {
                                  return node->get_friendly_name().find("conv2d") != std::string::npos;
                              }) != function->get_ordered_ops().end());
}
