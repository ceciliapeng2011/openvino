// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <inference_engine.hpp>
#include "ngraph/ngraph.hpp"

using namespace InferenceEngine;
using namespace ngraph;
using namespace element;

const std::string FLAGS_d("CPU");

std::shared_ptr<Function> createNgraphFunction(const Shape& boxes_shape, const Shape& scores_shape) {
    const int64_t nms_top_k = 3;
    const float iou_threshold = 0.5f;
    const float score_threshold = 0.0f;
    const auto sort_result_type = op::v8::MulticlassNms::SortResultType::CLASSID;
    const auto keep_top_k = -1;
    const auto background_class = -1;
    const auto nms_eta = 1.0f;

    // ----input---------
    const auto boxes = std::make_shared<op::Parameter>(element::f32, boxes_shape);
    const auto scores = std::make_shared<op::Parameter>(element::f32, scores_shape);

    // ----nms---------
    auto nms = std::make_shared<op::v8::MulticlassNms>(boxes,
                                                  scores,
                                                  sort_result_type,
                                                  false,
                                                  element::i64,
                                                  iou_threshold,
                                                  score_threshold,
                                                  nms_top_k,
                                                  keep_top_k,
                                                  background_class,
                                                  nms_eta);

    // ----gather---------
    auto const_axis0 = op::Constant::create<int64_t>(i64, {1}, {0});
    auto gather_outputs = std::make_shared<op::v7::Gather>(nms->output(0), nms->output(2), const_axis0);
    auto gather_indices = std::make_shared<op::v7::Gather>(nms->output(1), nms->output(2), const_axis0);

    // -------ngraph function-- 
    auto result2 = std::make_shared<op::Result>(nms->output(2));
    auto result1 = std::make_shared<op::Result>(gather_indices->output(0));
    auto result0 = std::make_shared<op::Result>(gather_outputs->output(0));
    auto result_full={ result0, result1, result2}; 

    auto _result2 = std::make_shared<op::Result>(nms->output(2));
    auto _result1 = std::make_shared<op::Result>(nms->output(1));
    auto _result0 = std::make_shared<op::Result>(nms->output(0));
    auto _result_full={ _result0, _result1, _result2};                              

    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<Function>(_result_full, ParameterVector{boxes, scores}, "testnet");

    return fnPtr;
}

template <typename T>
void print_output(const T* outputData, const SizeVector& dim)
{
    if(dim.size() == 1)
    {
        for(auto i = 0; i < dim[0]; i++)
        {
            std::cout << outputData[i] << std::endl;
        }
    } else if(dim.size() == 2)
    {
        for(auto i = 0; i < dim[0]; i++)
        {
            for(auto j = 0; j < dim[1]; j++)
                std::cout << outputData[i*dim[1]+j] << std::endl;
        }
    } else
    {
        size_t sum = 0;
        for(auto& d:dim)
            sum += d;
        for(auto i = 0; i < sum; i++)
        {
            std::cout << outputData[i] << std::endl;
        }        
    }    
}

int main(int argc, char* argv[]) {
    // N 1, C 2, M 6
    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    std::vector<float> scores_data = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 
        0.95, 0.75, 0.6, 0.80, 0.5, 0.3};

    const auto boxes_shape = Shape{1, 6, 4}; // N 1, C 2, M 6
    const auto scores_shape = Shape{1, 2, 6};

    try {
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        std::cout << "Loading Inference Engine" << std::endl;
        Core ie;

        std::cout << "Device info: " << std::endl;
        //std::cout << ie.GetVersions(FLAGS_d) << std::endl;

        // Create network using ngraph function
        CNNNetwork network(createNgraphFunction(boxes_shape, scores_shape));

        // Configure input & output
        // Prepare input blobs
        std::cout << "Preparing input blobs" << std::endl;
        std::string scoresInputName, boxesInputName;

        InputsDataMap inputInfo = network.getInputsInfo();
        for (auto& item : inputInfo) {
            std::cout << "input name: " << item.first << ", input data: " << item.second << std::endl;

            std::cout << "input dim: {";
            for(auto& dim:item.second->getTensorDesc().getDims())
            {
                std::cout<<dim<<", ";
            }
            std::cout << "}\n";

            if(item.second->getInputData()->getTensorDesc().getDims()[2] == 4)
            {
                boxesInputName = item.first;
            }
            else
            {
                scoresInputName = item.first;
            }            
        }

        /** Setting batch size using image count **/
        network.setBatchSize(1);
        size_t batchSize = network.getBatchSize();
        std::cout << "Batch size is " << std::to_string(batchSize) << std::endl;

        // Prepare output blobs
        std::cout << "Checking that the outputs are as expects" << std::endl;
        OutputsDataMap outputsInfo(network.getOutputsInfo());
        std::string valid_outputs, selected_outputs, selected_indices;     

        for (auto& item : outputsInfo) {
            std::cout << "output name: " << item.first << ", output data: " << item.second << std::endl;

            std::cout << "output dim: {";
            for(auto& dim:item.second->getTensorDesc().getDims())
            {
                std::cout<<dim<<", ";
            }
            std::cout << "}\n";

            if(item.second->getTensorDesc().getDims().size() == 1)
            {
                valid_outputs = item.first;
            } else if(item.second->getTensorDesc().getDims()[1] == 6)
            {
                selected_outputs = item.first;
            } else
            {
                selected_indices = item.first;
            }            
        }

        // Loading model to the device
        std::cout << "Loading model to the device" << std::endl;
        ExecutableNetwork exeNetwork = ie.LoadNetwork(network, FLAGS_d);

        // Create infer request
        std::cout << "Create infer request" << std::endl;
        InferRequest infer_request = exeNetwork.CreateInferRequest();

        // Prepare input
        Blob::Ptr input1 = infer_request.GetBlob(scoresInputName);
        auto data1 = input1->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
        std::memcpy(data1, scores_data.data(), scores_data.size()*sizeof(float));

        Blob::Ptr input2 = infer_request.GetBlob(boxesInputName);
        auto data2 = input2->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
        std::memcpy(data2, boxes_data.data(), boxes_data.size()*sizeof(float));

        inputInfo = {};

        // Do inference
        std::cout << "Start inference" << std::endl;
        infer_request.Infer();

        auto show_output_func = [](float iou, float adaptive_threshold) {
            return iou <= adaptive_threshold ? 1.0f : 0.0f;
        };        

        // Process output
        std::cout << "Processing output blobs" << std::endl;
        {
            std::cout << "Processing output valid_outputs : " << valid_outputs << std::endl;

            const Blob::Ptr output_blob = infer_request.GetBlob(valid_outputs);            

            MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
            if (!moutput) {
                throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                    "but by fact we were not able to cast output to MemoryBlob");
            }  
            auto moutputHolder = moutput->rmap();
            const int* outputData = moutputHolder.as<const PrecisionTrait<Precision::I32>::value_type*>();
            print_output(outputData, outputsInfo[valid_outputs]->getTensorDesc().getDims());
        }
        {
            std::cout << "Processing output selected_outputs : " << selected_outputs << std::endl;

            const Blob::Ptr output_blob = infer_request.GetBlob(selected_outputs);            

            MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
            if (!moutput) {
                throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                    "but by fact we were not able to cast output to MemoryBlob");
            }  
            auto moutputHolder = moutput->rmap();
            const float* outputData = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();
            print_output(outputData, outputsInfo[selected_outputs]->getTensorDesc().getDims());
        }
        {
            std::cout << "Processing output selected_indices : " << selected_indices << std::endl;

            const Blob::Ptr output_blob = infer_request.GetBlob(selected_indices);            

            MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
            if (!moutput) {
                throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                    "but by fact we were not able to cast output to MemoryBlob");
            }  
            auto moutputHolder = moutput->rmap();
            const int* outputData = moutputHolder.as<const PrecisionTrait<Precision::I32>::value_type*>();
            print_output(outputData, outputsInfo[selected_indices]->getTensorDesc().getDims());
        }                  
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "The end." << std::endl;
    return 0;
}
