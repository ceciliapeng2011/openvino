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

// TODO: in future, move it to shared header file so that every frontend can run same test suite
// ----- This is common part for all frontends -----
using BasicTestParam = std::tuple<std::string, std::string>;

std::string getTestCaseName(const testing::TestParamInfo<BasicTestParam> &obj) {
    std::string fe, fileName;
    std::tie(fe, fileName) = obj.param;
    // need to replace special characters to create valid test case name
    fileName = std::regex_replace(fileName, std::regex("[/\\.]"), "_");
    return fe + "_" + fileName;
}

class FrontEndBasicTest : public ::testing::TestWithParam<BasicTestParam> {
public:
    std::string                      m_feName;
    std::string                      m_modelFile;
    FrontEndManager                  m_fem;
    FrontEnd::Ptr                    m_frontEnd;
    InputModel::Ptr                  m_inputModel;

    void initParamTest() {
        std::tie(m_feName, m_modelFile) = GetParam();
        m_modelFile = std::string(TEST_FILES) + PATH_TO_MODELS + m_modelFile;
        std::cout << "Model: " << m_modelFile << std::endl;
    }

    void SetUp() override {
        initParamTest();
    }

    void TearDown() override {
    }

    void doLoadFromFile() {
        std::vector<std::string> frontends;
        ASSERT_NO_THROW(frontends = m_fem.availableFrontEnds());
        ASSERT_NO_THROW(m_frontEnd = m_fem.loadByFramework(m_feName));
        ASSERT_NE(m_frontEnd, nullptr);
        ASSERT_NO_THROW(m_inputModel = m_frontEnd->loadFromFile(m_modelFile));
        ASSERT_NE(m_inputModel, nullptr);
    }
};

TEST_P(FrontEndBasicTest, testLoadFromFile) {
    ASSERT_NO_THROW(doLoadFromFile());
    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(function, nullptr);
//    std::cout << "Ordered ops names\n";
//    for (const auto &n : function->get_ordered_ops()) {
//        std::cout << "----" << n->get_friendly_name() << "---\n";
//    }
}

TEST_P(FrontEndBasicTest, testInputModel_getInputsOutputs)
{
    ASSERT_NO_THROW(doLoadFromFile());

    using CustomCheck = std::function<void(Place::Ptr)>;
    auto checkPlaces = [&](const std::vector<Place::Ptr>& places, CustomCheck cb) {
        EXPECT_GT(places.size(), 0);
        std::set<Place::Ptr> placesSet(places.begin(), places.end());
        EXPECT_EQ(placesSet.size(), places.size());
        std::for_each(places.begin(), places.end(), [&](Place::Ptr place) {
            ASSERT_NE(place, nullptr);
            EXPECT_GT(place->getNames().size(), 0);
            cb(place);
        });
    };
    std::vector<Place::Ptr> inputs;
    ASSERT_NO_THROW(inputs = m_inputModel->getInputs());
    checkPlaces(inputs, [&](Place::Ptr place) {
        EXPECT_TRUE(place->isInput());
    });

    std::vector<Place::Ptr> outputs;
    ASSERT_NO_THROW(outputs = m_inputModel->getOutputs());
    checkPlaces(outputs, [&](Place::Ptr place) {
        EXPECT_TRUE(place->isOutput());
    });
}

TEST_P(FrontEndBasicTest, testInputModel_getPlaceByTensorName)
{
    ASSERT_NO_THROW(doLoadFromFile());

    auto testGetPlaceByTensorName = [&](const std::vector<Place::Ptr>& places) {
        EXPECT_GT(places.size(), 0);
        for (auto place : places) {
            ASSERT_NE(place, nullptr);
            for (auto name : place->getNames()) {
                EXPECT_NE(name, std::string());
                Place::Ptr placeByName;
                ASSERT_NO_THROW(placeByName = m_inputModel->getPlaceByTensorName(name));
                ASSERT_NE(placeByName, nullptr);
                EXPECT_TRUE(placeByName->isEqual(place));
            }
        }
    };

    std::vector<Place::Ptr> outputs;
    ASSERT_NO_THROW(outputs = m_inputModel->getOutputs());
    testGetPlaceByTensorName(outputs);

    std::vector<Place::Ptr> inputs;
    ASSERT_NO_THROW(inputs = m_inputModel->getInputs());
    testGetPlaceByTensorName(inputs);
}

TEST_P(FrontEndBasicTest, testInputModel_overrideAll)
{
    ASSERT_NO_THROW(doLoadFromFile());

    using GetPlaces = std::function<std::vector<Place::Ptr>()>;
    using OverridePlaces = std::function<void(const std::vector<Place::Ptr>&)>;
    auto verifyOverride = [](GetPlaces getCB, OverridePlaces overrideCB) {
        std::vector<Place::Ptr> places;
        ASSERT_NO_THROW(places = getCB());
        std::set<Place::Ptr> placesSet(places.begin(), places.end());

        auto placesReversed = places;
        std::reverse(placesReversed.begin(), placesReversed.end());
        ASSERT_NO_THROW(overrideCB(placesReversed));
        ASSERT_NO_THROW(places = getCB());
        EXPECT_GT(places.size(), 0);
        std::set<Place::Ptr> placesSetAfter(places.begin(), places.end());
        EXPECT_EQ(placesSet.size(), placesSet.size());
        std::for_each(places.begin(), places.end(), [&](Place::Ptr place) {
            EXPECT_GT(placesSet.count(place), 0);
        });
    };
    verifyOverride([&]() { return m_inputModel->getInputs(); },
                   [&](const std::vector<Place::Ptr>& p) { m_inputModel->overrideAllInputs(p); });

    verifyOverride([&]() { return m_inputModel->getOutputs(); },
                   [&](const std::vector<Place::Ptr>& p) { m_inputModel->overrideAllOutputs(p); });
}

TEST_P(FrontEndBasicTest, testInputModel_overrideAll_empty)
{
    ASSERT_NO_THROW(doLoadFromFile());
    using GetPlaces = std::function<std::vector<Place::Ptr>()>;
    using OverrideEmpty = std::function<void(void)>;
    using CustomCheck = std::function<void(std::string)>;
    auto verifyOverride = [](GetPlaces getCB, OverrideEmpty overrideCB, CustomCheck customCB) {
        std::vector<Place::Ptr> places;
        std::vector<Place::Ptr> newPlaces;
        ASSERT_NO_THROW(places = getCB());
        ASSERT_NO_THROW(overrideCB());
        ASSERT_NO_THROW(newPlaces = getCB());
        ASSERT_EQ(newPlaces.size(), 0);
        std::for_each(places.begin(), places.end(), [&](Place::Ptr place) {
            for (auto name : place->getNames()) {
                customCB(name);
            }
        });
    };
    verifyOverride([&]() { return m_inputModel->getOutputs(); },
                   [&]() { m_inputModel->overrideAllOutputs({}); },
                   [&](const std::string &name) {
                       EXPECT_FALSE(m_inputModel->getPlaceByTensorName(name)->isOutput());
                   });

    verifyOverride([&]() { return m_inputModel->getInputs(); },
                   [&]() { m_inputModel->overrideAllInputs({}); },
                   [&](const std::string &name) {
                       EXPECT_FALSE(m_inputModel->getPlaceByTensorName(name)->isInput());
                   });
}

TEST_P(FrontEndBasicTest, DISABLED_testInputModel_extractSubgraph)
{
    ASSERT_NO_THROW(doLoadFromFile());

    // TODO: not clear now
}

TEST_P(FrontEndBasicTest, DISABLED_testInputModel_setPartialShape)
{
    ASSERT_NO_THROW(doLoadFromFile());

    // TODO: not clear now
}
// ----- End of common part for all frontends -----

// This part is frontend-specific
using PDPDBasicTest = FrontEndBasicTest;

static const std::vector<std::string> models {
        std::string("conv2d"),
        std::string("conv2d_s/conv2d.pdmodel"),
        std::string("conv2d_relu/conv2d_relu.pdmodel"),
        std::string("2in_2out/2in_2out.pdmodel"),
};

INSTANTIATE_TEST_CASE_P(PDPDBasicTest, FrontEndBasicTest,
                        ::testing::Combine(
                            ::testing::Values(std::string(PDPD)),
                            ::testing::ValuesIn(models)),
                        getTestCaseName);