#include "multiclass_nms.hpp"

using namespace ngraph;

int main(int argc, char *argv[])
{
    // flipped_coordinates
/*     std::vector<float> boxes_data = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                     1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}; */

    std::vector<float> boxes_data = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                     0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                     0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};                                     

    std::vector<float> scores_data = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    const int64_t nms_top_k = 3;
    const int64_t keep_top_k = -1;
    const float iou_threshold_data = 0.5f;
    const float score_threshold_data = 0.0f;
    const int64_t background_class = 0;
    const std::string sort_result = "none";
    const float nms_eta = 1.0;

    const auto boxes_shape = Shape{1, 6, 4};
    const auto scores_shape = Shape{1, 1, 6};

    const auto out_shape = Shape{3, 3};
    const auto out_shape_size = 3*3;   

    std::vector<int64_t> selected_indices(out_shape_size);
    std::vector<float> selected_scores(out_shape_size);
    int64_t valid_outputs = 0;

    runtime::reference::multiclass_nms(boxes_data.data(),
                                            boxes_shape,
                                            scores_data.data(),
                                            scores_shape,                                            
                                            iou_threshold_data,
                                            score_threshold_data,
                                            nms_top_k,
                                            keep_top_k,
                                            background_class,
                                            nms_eta,
                                            selected_indices.data(),
                                            out_shape,
                                            selected_scores.data(),
                                            out_shape,
                                            &valid_outputs,
                                            sort_result);

    std::cout << "selected_indices: { ";
    for(auto &v : selected_indices)
        std::cout<< v << ", ";
    std::cout<< "}" << std::endl;

    std::cout << "selected_scores: { ";
    for(auto &v : selected_scores)
        std::cout<< v << ", ";
    std::cout<< "}" << std::endl;  

    std::cout << "valid_outputs: { ";
    std::cout<< valid_outputs << ", ";
    std::cout<< "}" << std::endl;                                             

/*     auto selected_scores_type =
        (inputs.size() < 4) ? element::f32 : inputs[3]->get_element_type(); */

/*     runtime::reference::multiclass_nms_postprocessing(outputs,
                                            output_type,
                                            selected_indices,
                                            selected_scores,
                                            valid_outputs,
                                            selected_scores_type); */

    return 0;
}
