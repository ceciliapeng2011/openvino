// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>
#include <fstream>

#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

#include "ngraph/ngraph.hpp"

using namespace ngraph;
using namespace InferenceEngine;

#include "../shared/include/basic_api.hpp"

// library taken from https://github.com/llohse/libnpy
#include "../shared/include/npy.hpp"

using namespace ngraph;
using namespace ngraph::frontend;
using TestEngine = test::IE_CPU_Engine;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

/* helper */
static bool ends_with(std::string const & value, std::string const & ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

static bool starts_with(std::string const & value, std::string const & starting) {
    if (starting.size() > value.size()) return false;
    return std::equal(starting.begin(), starting.end(), value.begin());
}

static std::string get_modelfolder(std::string& modelfile) { 
    if (!ends_with(modelfile, ".pdmodel")) return modelfile;
    size_t found = modelfile.find_last_of("/\\");
    return  modelfile.substr(0,found);             
};

static const std::string& trim_space(std::string& str) //trim leading and tailing spaces
{
    //leading
    auto it = str.begin();
    for (; it != str.end() && isspace(*it); it++);                 
    auto d = std::distance(str.begin(), it);
    str.erase(0,d);

    //tailing
    auto rit = str.rbegin();
    for (; rit != str.rend() && isspace(*rit); rit++) {
        str.pop_back();
    }

    //std::cout << "[" << str << "]" << std::endl; 
    return str;        
} 

static std::vector<std::string> get_models(void) {
    std::string models_csv = std::string(TEST_FILES) + PATH_TO_MODELS + "models.csv";
    std::ifstream f(models_csv);
    std::vector<std::string> models;
    std::string line;
    while (getline(f, line, ',')) {            
        auto line_trim = trim_space(line);
        if(line_trim.empty() || 
            starts_with(line_trim, "#")) 
            continue;
        // std::cout<< "line in csv: [" << line_trim<< "]" << std::endl;            
        models.emplace_back(line_trim);
    }
    return models;
} 

static void visualizer(std::shared_ptr<ngraph::Function> function, std::string path) {
        ngraph::pass::VisualizeTree("function.png").run_on_function(function);
        
        CNNNetwork network(function);
        network.serialize(path+".xml", path+".bin");    
}

namespace fuzzyOp {
    using PDPDFuzzyOpTest = FrontEndBasicTest; 
    using PDPDFuzzyOpTestParam = std::tuple<std::string,  // FrontEnd name
                                            std::string,  // Base path to models
                                            std::string>; // modelname 

    void run_fuzzy(std::shared_ptr<ngraph::Function> function, std::string& modelfile) {
        auto _load_from_npy = [&](std::string& file_path) {
            std::ifstream npy_file(file_path);
            std::vector<unsigned long> npy_shape;
            std::vector<float> npy_data;
            if (npy_file.good())
                npy::LoadArrayFromNumpy(file_path, npy_shape, npy_data);

            return npy_data;
        };
        auto _load_int32_from_npy = [&](std::string& file_path) {
            std::ifstream npy_file(file_path);
            std::vector<unsigned long> npy_shape;
            std::vector<int32_t> npy_data;
            if (npy_file.good())
                npy::LoadArrayFromNumpy(file_path, npy_shape, npy_data);

            return npy_data;
        };  

        auto _get_npy_dtype = [&](std::string& filename) {
            std::ifstream stream(filename, std::ifstream::binary);
            if(!stream) {
                throw std::runtime_error("io error: failed to open a file.");
            }

            std::string header = npy::read_header(stream);

            // parse header
            bool fortran_order;
            std::string typestr;
            std::vector<unsigned long>shape;

            npy::parse_header(header, typestr, fortran_order, shape);

            std::cout << "$$$$$$$$$" << filename << ": " << typestr << std::endl;
            return typestr;       
        };      

        auto modelfolder = get_modelfolder(modelfile);

        // run test
        auto test_case = test::TestCase<TestEngine>(function);

        const auto parameters = function->get_parameters();
        for (auto i = 0; i < parameters.size(); i++) {
            // read input npy file
            std::string input_path = modelfolder+"/input"+std::to_string((parameters.size()-1)-i)+".npy";

            auto dtype = _get_npy_dtype(input_path);
            if (dtype == "<f4")
            {
                auto npy_input = _load_from_npy(input_path);
                if (npy_input.empty()) {
                    throw std::runtime_error("failed to load input npy for test case. Tried " + input_path);
                }  

                std::vector<float> data_input(npy_input.size());
                std::copy_n(npy_input.data(), npy_input.size(), data_input.begin());

                const auto& input_pshape = parameters.at(i)->get_output_partial_shape(0);
                std::cout << "input_pshape###########" << input_pshape << std::endl;
                test_case.add_input(data_input);                     

            } else if (dtype == "<i4")
            {
                auto npy_input = _load_int32_from_npy(input_path);
                if (npy_input.empty()) {
                    throw std::runtime_error("failed to load input npy for test case. Tried " + input_path);
                }  

                std::vector<int32_t> data_input(npy_input.size());
                std::copy_n(npy_input.data(), npy_input.size(), data_input.begin());

                const auto& input_pshape = parameters.at(i)->get_output_partial_shape(0);
                std::cout << "input_pshape###########" << input_pshape << std::endl;
                test_case.add_input(data_input);                  
            } else {
                throw std::runtime_error("not supported dtype in" + dtype);
            }          
        }
        
        const auto results = function->get_results();
        for (auto i = 0; i < results.size(); i++) {
            // read input npy file
            std::string datafile = modelfolder+"/output"+std::to_string(i)+".npy";

            auto dtype = _get_npy_dtype(datafile);
            if (dtype == "<f4")
            {
                auto npy_input = _load_from_npy(datafile);
                if (npy_input.empty()) {
                    throw std::runtime_error("failed to load input npy for test case. Tried " + datafile);
                }  

                std::vector<float> data_input(npy_input.size());
                std::copy_n(npy_input.data(), npy_input.size(), data_input.begin());

                const auto& input_pshape = results.at(i)->get_output_partial_shape(0);
                std::cout << "input_pshape###########" << input_pshape << std::endl;
                test_case.add_expected_output(data_input);                     

            } else if (dtype == "<i4")
            {
                auto npy_input = _load_int32_from_npy(datafile);
                if (npy_input.empty()) {
                    throw std::runtime_error("failed to load input npy for test case. Tried " + datafile);
                }  

                std::vector<int32_t> data_input(npy_input.size());
                std::copy_n(npy_input.data(), npy_input.size(), data_input.begin());

                const auto& input_pshape = results.at(i)->get_output_partial_shape(0);
                std::cout << "output_pshape###########" << input_pshape << std::endl;
                test_case.add_expected_output(data_input);         
            } else {
                throw std::runtime_error("not supported dtype out "+ dtype);
            }          
        }
            
        test_case.run_with_tolerance_as_fp(1e-4);
    }

    TEST_P(PDPDFuzzyOpTest, test_fuzzy) {
        // load
        ASSERT_NO_THROW(doLoadFromFile());

        // convert
        std::shared_ptr<ngraph::Function> function;
        function = m_frontEnd->convert(m_inputModel);
        ASSERT_NE(function, nullptr);

        // debug
        //visualizer(function, get_modelfolder(m_modelFile)+"/fuzzy");

        // run
        run_fuzzy(function, m_modelFile);
    }

    INSTANTIATE_TEST_CASE_P(FrontendOpTest, PDPDFuzzyOpTest,
                        ::testing::Combine(
                            ::testing::Values(PDPD),
                            ::testing::Values(PATH_TO_MODELS),
                            ::testing::ValuesIn(get_models())),                                                                
                            PDPDFuzzyOpTest::getTestCaseName);                                                 

}
