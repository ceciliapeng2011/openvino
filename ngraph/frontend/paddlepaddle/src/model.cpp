// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../include/paddlepaddle_frontend/model.hpp"

#include "framework.pb.h"

namespace ngraph {
namespace frontend {

InputModelPDPD::InputModelPDPD(const std::string& _path) : path(_path) {
    std::string ext = ".pdmodel";            
    std::string model_file(path);
    if (path.length() >= ext.length() && (0 == path.compare(path.length() - ext.length(), ext.length(), ext)))
    {
        weights_composed = true;
        auto weights_file = path.replace(path.size() - ext.size(), ext.size(), ".pdiparams");
        weights_stream = std::ifstream(weights_file, std::ios::binary);
        if (!weights_stream || !weights_stream.is_open())
        {
            std::cerr << "Model file cannot be opened" << std::endl;
        }
    } else {
        weights_composed = false;
        model_file += "/__model__";
    }

    paddle::framework::proto::ProgramDesc* _fw_ptr = new paddle::framework::proto::ProgramDesc();
    fw_ptr = _fw_ptr;
    std::ifstream pb_stream(model_file, std::ios::binary);
    std::cout << "Model Parsed: " << _fw_ptr->ParseFromIstream(&pb_stream) << std::endl;

    std::cout << "Blocks number: " << _fw_ptr->blocks().size() << std::endl;
    for (const auto& block : _fw_ptr->blocks()) {
        places_map.push_back(std::vector<std::shared_ptr<PlacePDPD>>());

        for (int i = 0; i < block.ops_size(); i++) {
            places_map.back().push_back(std::make_shared<PlacePDPD>(PlacePDPD(&(block.ops()[i]))));
        }
    }
}

InputModelPDPD::~InputModelPDPD() {
    delete (paddle::framework::proto::ProgramDesc*)fw_ptr;
}

} // namespace frontend
} // namespace ngraph
