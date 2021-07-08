// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include "ie_parallel.hpp"
#include "mkldnn_multiclass_non_max_suppression.hpp"
#include "utils/general_utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <ie_ngraph_utils.hpp>
#include <ngraph_ops/multiclass_nms_ie_internal.hpp>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <chrono>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNMultiClassNonMaxSuppressionNode::isSupportedOperation(
    const std::shared_ptr<ngraph::Node> &op,
    std::string &errorMessage) noexcept {
  try {
    const auto nms = std::dynamic_pointer_cast<
        const ngraph::op::internal::MulticlassNonMaxSuppressionIEInternal>(op);
    if (!nms) {
      errorMessage = "Only internal NonMaxSuppression operation is supported";
      return false;
    }
  } catch (...) {
    return false;
  }
  return true;
}

MKLDNNMultiClassNonMaxSuppressionNode::MKLDNNMultiClassNonMaxSuppressionNode(
    const std::shared_ptr<ngraph::Node> &op, const mkldnn::engine &eng,
    MKLDNNWeightsSharing::Ptr &cache)
    : MKLDNNNode(op, eng, cache) {
  std::string errorMessage;
  if (!isSupportedOperation(op, errorMessage)) {
    IE_THROW(NotImplemented) << errorMessage;
  }
  errorPrefix =
      "multiclass_nms layer with name '" + op->get_friendly_name() + "' ";
  const auto nms = std::dynamic_pointer_cast<
      const ngraph::op::internal::MulticlassNonMaxSuppressionIEInternal>(op);

  if (nms->get_input_size() != 2)
    IE_THROW() << errorPrefix << "has incorrect number of input edges: "
               << nms->get_input_size();

  if (nms->get_output_size() < 1 || nms->get_output_size() > 3)
    IE_THROW() << errorPrefix << "has incorrect number of output edges: "
               << nms->get_output_size();

  sort_result_across_batch = nms->m_sort_result_across_batch;

  max_output_boxes_per_class = nms->m_nms_top_k;
  iou_threshold = nms->m_iou_threshold;
  score_threshold = nms->m_score_threshold;
  background_class = nms->m_background_class;
  keep_top_k = nms->m_keep_top_k;
  sort_result_type = nms->m_sort_result_type;
  nms_eta = nms->m_nms_eta;
  normalized = nms->m_normalized;

  const SizeVector &boxes_dims = op->get_input_shape(NMS_BOXES);
  num_batches = boxes_dims[0];
  num_boxes = boxes_dims[1];
  if (boxes_dims.size() != 3)
    IE_THROW() << errorPrefix
               << "has unsupported 'boxes' input rank: " << boxes_dims.size();
  if (boxes_dims[2] != 4)
    IE_THROW() << errorPrefix
               << "has unsupported 'boxes' input 3rd dimension size: "
               << boxes_dims[2];

  const SizeVector &scores_dims = op->get_input_shape(NMS_SCORES);
  num_classes = scores_dims[1];
  if (scores_dims.size() != 3)
    IE_THROW() << errorPrefix
               << "has unsupported 'scores' input rank: " << scores_dims.size();

  if (num_batches != scores_dims[0])
    IE_THROW() << errorPrefix
               << " num_batches is different in 'boxes' and 'scores' inputs";
  if (num_boxes != scores_dims[2])
    IE_THROW() << errorPrefix
               << " num_boxes is different in 'boxes' and 'scores' inputs";

  numFiltBox.resize(num_batches); // batches
  numBoxOffset.resize(num_batches);
  for (size_t i = 0; i < numFiltBox.size(); i++) {
    numFiltBox[i].resize(num_classes); // classes
  }

  outputShape_SELECTEDINDICES = op->get_output_shape(NMS_SELECTEDINDICES);
  outputShape_SELECTEDOUTPUTS = op->get_output_shape(NMS_SELECTEDOUTPUTS);
  const SizeVector &valid_outputs_dims = op->get_output_shape(NMS_SELECTEDNUM);
  if (valid_outputs_dims.size() != 1)
    IE_THROW() << errorPrefix << "has unsupported 'valid_outputs' output rank: "
               << valid_outputs_dims.size();
  if (valid_outputs_dims[0] !=
      num_batches) // valid_outputs_dims[0] != num_batches
    IE_THROW() << errorPrefix
               << "has unsupported 'valid_outputs' output 1st dimension size: "
               << valid_outputs_dims[0];
}

void MKLDNNMultiClassNonMaxSuppressionNode::
    initSupportedPrimitiveDescriptors() {
  if (!supportedPrimitiveDescriptors.empty())
    return;
  const std::vector<Precision> supportedFloatPrecision = {
      Precision::FP32, Precision::BF16}; // support BF16?
  const std::vector<Precision> supportedIntOutputPrecision = {Precision::I32,
                                                              Precision::I64};

  checkPrecision(getOriginalInputPrecisionAtPort(NMS_BOXES),
                 supportedFloatPrecision, "boxes", inType);

  checkPrecision(getOriginalInputPrecisionAtPort(NMS_SCORES),
                 supportedFloatPrecision, "scores", inType);

  const std::vector<Precision> supportedPrecision = {
      Precision::I16, Precision::U8,  Precision::I8,  Precision::U16,
      Precision::I32, Precision::U32, Precision::I64, Precision::U64};

  checkOutput(outputShape_SELECTEDINDICES, supportedIntOutputPrecision,
              "selected_indices", NMS_SELECTEDINDICES);
  checkOutput(outputShape_SELECTEDOUTPUTS, supportedFloatPrecision,
              "selected_outputs", NMS_SELECTEDOUTPUTS);
  checkPrecision(getOriginalOutputPrecisionAtPort(NMS_SELECTEDNUM),
                 supportedIntOutputPrecision, "selected_num", outType);

  std::vector<DataConfigurator> inDataConf;
  inDataConf.reserve(getOriginalInputsNumber());
  for (int i = 0; i < getOriginalInputsNumber(); ++i) {
    Precision inPrecision = Precision::FP32;
    inDataConf.emplace_back(TensorDescCreatorTypes::ncsp, inPrecision);
  }

  std::vector<DataConfigurator> outDataConf;
  outDataConf.reserve(getOriginalOutputsNumber());
  for (int i = 0; i < getOriginalOutputsNumber(); ++i) {
    Precision outPrecision =
        i == NMS_SELECTEDOUTPUTS ? Precision::FP32 : Precision::I32;
    outDataConf.emplace_back(TensorDescCreatorTypes::ncsp, outPrecision);
  }

  addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
}

void MKLDNNMultiClassNonMaxSuppressionNode::execute(mkldnn::stream strm) {
  auto beforeTime = std::chrono::steady_clock::now();
  const float *boxes = reinterpret_cast<const float *>(
      getParentEdgeAt(NMS_BOXES)->getMemoryPtr()->GetPtr());
  const float *scores = reinterpret_cast<const float *>(
      getParentEdgeAt(NMS_SCORES)->getMemoryPtr()->GetPtr());

  auto dims_boxes =
      getParentEdgeAt(NMS_BOXES)->getDesc().getDims(); // TODO right way?
  /////////////////////printf inputs data//////////////////////////
  // printf("Input Box:\n");
  // for (int i = 0; i < dims_boxes[0]; i++)
  //   for (int j = 0; j < dims_boxes[1]; j++)
  //     printf("Batch: %d Box %d: %f, %f, %f, %f \n", i, j,
  //            boxes[i * dims_boxes[1] * 4 + j * 4],
  //            boxes[i * dims_boxes[1] * 4 + j * 4 + 1],
  //            boxes[i * dims_boxes[1] * 4 + j * 4 + 2],
  //            boxes[i * dims_boxes[1] * 4 + j * 4 + 3]);

  // printf("Input Scores:\n");
  // auto dims_scores = inputs[NMS_SCORES]->getTensorDesc().getDims();
  // for (int i = 0; i < dims_scores[0]; i++)
  //   for (int j = 0; j < dims_scores[1]; j++)
  //     for (int k = 0; k < dims_scores[2]; k++)
  //       printf("Batch: %d Class %d: %f\n", i, j,
  //              scores[i * dims_scores[1] * dims_scores[2] +
  //                     j * dims_scores[2] + k]);
  /////////////////////printf inputs data//////////////////////////
  if (max_output_boxes_per_class == 0)
    return;
  else if (max_output_boxes_per_class == -1)
    max_output_boxes_per_class = dims_boxes[1];

  int *selected_indices = reinterpret_cast<int *>(
      getChildEdgesAtPort(NMS_SELECTEDINDICES)[0]->getMemoryPtr()->GetPtr());

  float *selected_outputs = nullptr;
  if (outDims.size() > NMS_SELECTEDOUTPUTS)
    selected_outputs = reinterpret_cast<float *>(
        getChildEdgesAtPort(NMS_SELECTEDOUTPUTS)[0]->getMemoryPtr()->GetPtr());

  int *selected_num = nullptr;
  if (outDims.size() > NMS_SELECTEDNUM)
    selected_num = reinterpret_cast<int *>(
        getChildEdgesAtPort(NMS_SELECTEDNUM)[0]->getMemoryPtr()->GetPtr());

  auto boxesStrides =
      getParentEdgeAt(NMS_BOXES)->getDesc().getBlockingDesc().getStrides();
  auto scoresStrides =
      getParentEdgeAt(NMS_SCORES)->getDesc().getBlockingDesc().getStrides();

  std::vector<filteredBoxes> filtBoxes(max_output_boxes_per_class *
                                       num_batches * num_classes);

  std::vector<size_t> numBoxperBatch(num_batches);

  if ((nms_eta >= 0) && (nms_eta < 1)) {
    nmsWithEta(boxes, scores, boxesStrides, scoresStrides, filtBoxes);
  } else {
    nmsWithoutEta(boxes, scores, boxesStrides, scoresStrides, filtBoxes);
  }

  size_t startOffset = numFiltBox[0][0];
  numBoxOffset[0] = 0;
  for (size_t b = 0; b < numFiltBox.size(); b++) {
    size_t batchOffsetNew = 0;
    size_t batchOffset = b * num_classes * max_output_boxes_per_class;
    for (size_t c = (b == 0 ? 1 : 0); c < numFiltBox[b].size(); c++) {
      size_t offset = batchOffset + c * max_output_boxes_per_class;
      for (size_t i = 0; i < numFiltBox[b][c]; i++) {
        filtBoxes[startOffset + i] = filtBoxes[offset + i];
      }
      startOffset += numFiltBox[b][c];
      batchOffsetNew += numFiltBox[b][c];
    }
    numBoxOffset[b] = batchOffsetNew;
    if (b == 0)
      numBoxOffset[b] += numFiltBox[0][0];
  }
  filtBoxes.resize(startOffset);

  // sort element before go through keep_top_k
  parallel_sort(filtBoxes.begin(), filtBoxes.end(),
                [](const filteredBoxes &l, const filteredBoxes &r) {
                  return ((l.batch_index < r.batch_index) ||
                          ((l.batch_index == r.batch_index) &&
                           ((l.score > r.score) ||
                            ((std::fabs(l.score - r.score) < 1e-6) &&
                             l.class_index < r.class_index) ||
                            ((std::fabs(l.score - r.score) < 1e-6) &&
                             l.class_index == r.class_index &&
                             l.box_index < r.box_index))));
                });

  if (keep_top_k > -1) {
    startOffset = 0;
    size_t offset = 0;
    for (size_t b = 0; b < numFiltBox.size(); b++) {
      if (numBoxOffset[b] > keep_top_k) {
        if (startOffset == offset) {
          startOffset += keep_top_k;
          offset += numBoxOffset[b];
        } else {
          for (size_t i = 0; i < keep_top_k; i++) {
            filtBoxes[startOffset + i] = filtBoxes[offset + i];
          }
          startOffset += keep_top_k;
          offset += numBoxOffset[b];
        }
      } else {
        if (startOffset == offset) {
          startOffset += numBoxOffset[b];
          offset += numBoxOffset[b];
        } else {
          for (size_t i = 0; i < numBoxOffset[b]; i++) {
            filtBoxes[startOffset + i] = filtBoxes[offset + i];
          }
          startOffset += numBoxOffset[b];
          offset += numBoxOffset[b];
        }
      }
    }
    filtBoxes.resize(startOffset);
  }

  if (sort_result_across_batch) {
    if (sort_result_type == 1) {
      parallel_sort(
          filtBoxes.begin(), filtBoxes.end(),
          [](const filteredBoxes &l, const filteredBoxes &r) {
            return (l.score > r.score) ||
                   (l.score == r.score && l.batch_index < r.batch_index) ||
                   (l.score == r.score && l.batch_index == r.batch_index &&
                    l.class_index < r.class_index) ||
                   (l.score == r.score && l.batch_index == r.batch_index &&
                    l.class_index == r.class_index &&
                    l.box_index < r.box_index);
          });
    } else if (sort_result_type == 0) {
      parallel_sort(filtBoxes.begin(), filtBoxes.end(),
                    [](const filteredBoxes &l, const filteredBoxes &r) {
                      return (l.class_index < r.class_index) ||
                             (l.class_index == r.class_index &&
                              l.batch_index < r.batch_index) ||
                             (l.class_index == r.class_index &&
                              l.batch_index == r.batch_index &&
                              l.score > r.score) ||
                             (l.class_index == r.class_index &&
                              l.batch_index == r.batch_index &&
                              l.score == r.score && l.box_index < r.box_index);
                    });
    }
  } else if (sort_result_type == 0) {
    parallel_sort(filtBoxes.begin(), filtBoxes.end(),
                  [](const filteredBoxes &l, const filteredBoxes &r) {
                    return ((l.batch_index < r.batch_index) ||
                            (l.batch_index == r.batch_index) &&
                                ((l.class_index < r.class_index) ||
                                 ((l.class_index == r.class_index) &&
                                  l.score > r.score) ||
                                 ((std::fabs(l.score - r.score) <= 1e-6) &&
                                  l.class_index == r.class_index &&
                                  l.box_index < r.box_index)));
                  });
  }

  const size_t selectedBoxesNum =
      getChildEdgeAt(NMS_SELECTEDINDICES)->getDesc().getDims()[0];
  const size_t validOutputs = std::min(filtBoxes.size(), selectedBoxesNum);
  // printf("validOutputs: %ld\n", validOutputs);

  int *selectedIndicesPtr = selected_indices;
  float *selectedoutputsPtr = selected_outputs;
  auto selectedIndicesStride = getChildEdgeAt(NMS_SELECTEDINDICES)
                                   ->getDesc()
                                   .getBlockingDesc()
                                   .getStrides()[0];
  auto selectedoutputsStride = getChildEdgeAt(NMS_SELECTEDOUTPUTS)
                                   ->getDesc()
                                   .getBlockingDesc()
                                   .getStrides()[1];

  std::vector<size_t> m_selected_num;
  m_selected_num.resize(dims_boxes[0]);

  /////////////////////printf output data//////////////////////////
  // printf("selected_outputs output: \n");
  // size_t idx = 0lu;
  // for (; idx < validOutputs; idx++) {
  //   *selectedIndicesPtr =
  //       filtBoxes[idx].batch_index * dims_boxes[1] +
  //       filtBoxes[idx].box_index;
  //   printf("selectedOutput Num %ld: ", idx);
  //   *selectedoutputsPtr = filtBoxes[idx].class_index;
  //   printf("class_id %f ", *selectedoutputsPtr);
  //   selectedoutputsPtr += selectedoutputsStride;
  //   *selectedoutputsPtr = filtBoxes[idx].score;
  //   printf("score %f ", *selectedoutputsPtr);
  //   selectedoutputsPtr += selectedoutputsStride;
  //   *selectedoutputsPtr = boxes[*selectedIndicesPtr * 4];
  //   printf("ROI %f ", *selectedoutputsPtr);
  //   selectedoutputsPtr += selectedoutputsStride;
  //   *selectedoutputsPtr = boxes[*selectedIndicesPtr * 4 + 1];
  //   printf("%f ", *selectedoutputsPtr);
  //   selectedoutputsPtr += selectedoutputsStride;
  //   *selectedoutputsPtr = boxes[*selectedIndicesPtr * 4 + 2];
  //   printf("%f ", *selectedoutputsPtr);
  //   selectedoutputsPtr += selectedoutputsStride;
  //   *selectedoutputsPtr = boxes[*selectedIndicesPtr * 4 + 3];
  //   printf("%f ", *selectedoutputsPtr);
  //   selectedoutputsPtr += selectedoutputsStride;
  //   printf("\n");
  //   selectedIndicesPtr += selectedIndicesStride;
  // }

  // printf("selectedIndices output: \n"); // for print, can be merged into
  // the
  //                                       // former loop
  // selectedIndicesPtr = selected_indices;
  // idx = 0lu;
  // for (; idx < validOutputs; idx++) {
  //   m_selected_num[filtBoxes[idx].batch_index]++;
  //   printf("selectedIndices Num %ld: selectedIndices %d\n", idx,
  //          *selectedIndicesPtr);
  //   selectedIndicesPtr += selectedIndicesStride;
  // }

  // // use '0' instead of '-1' To align with ref
  // std::fill(selectedoutputsPtr,
  //           selectedoutputsPtr +
  //               (selectedBoxesNum - idx) * selectedoutputsStride * 6,
  //           0);
  // std::fill(selectedIndicesPtr,
  //           selectedIndicesPtr +
  //               (selectedBoxesNum - idx) * selectedIndicesStride,
  //           0);

  // printf("selected Number output: ");
  // if (outputs.size() > NMS_SELECTEDNUM) {
  //   // *selected_num = static_cast<int>(validOutputs);
  //   for (int i = 0; i < dims_boxes[0]; i++) {
  //     selected_num[i] = static_cast<int>(m_selected_num[i]);
  //     printf("%d ", selected_num[i]);
  //   }
  // }
  // printf("\n");
  /////////////////////printf output data//////////////////////////
  size_t idx = 0lu;
  for (; idx < validOutputs; idx++) {
    *selectedIndicesPtr =
        filtBoxes[idx].batch_index * dims_boxes[1] + filtBoxes[idx].box_index;
    *selectedoutputsPtr = filtBoxes[idx].class_index;
    selectedoutputsPtr += selectedoutputsStride;
    *selectedoutputsPtr = filtBoxes[idx].score;
    selectedoutputsPtr += selectedoutputsStride;
    *selectedoutputsPtr = boxes[*selectedIndicesPtr * 4];
    selectedoutputsPtr += selectedoutputsStride;
    *selectedoutputsPtr = boxes[*selectedIndicesPtr * 4 + 1];
    selectedoutputsPtr += selectedoutputsStride;
    *selectedoutputsPtr = boxes[*selectedIndicesPtr * 4 + 2];
    selectedoutputsPtr += selectedoutputsStride;
    *selectedoutputsPtr = boxes[*selectedIndicesPtr * 4 + 3];
    selectedoutputsPtr += selectedoutputsStride;

    m_selected_num[filtBoxes[idx].batch_index]++;

    selectedIndicesPtr += selectedIndicesStride;
  }
  // use '0' instead of '-1' To align with ref
  std::fill(selectedoutputsPtr,
            selectedoutputsPtr +
                (selectedBoxesNum - idx) * selectedoutputsStride * 6,
            0);
  std::fill(
      selectedIndicesPtr,
      selectedIndicesPtr + (selectedBoxesNum - idx) * selectedIndicesStride, 0);

  if (outDims.size() > NMS_SELECTEDNUM) {
    for (int i = 0; i < dims_boxes[0]; i++) {
      selected_num[i] = static_cast<int>(m_selected_num[i]);
    }
  }
  auto afterTime = std::chrono::steady_clock::now();
  std::cout << "CPU plugin Node算法耗时:" << std::endl;
  double duration_millsecond =
      std::chrono::duration<double, std::milli>(afterTime - beforeTime).count();
  std::cout << duration_millsecond << "毫秒" << std::endl;

  return;
}

bool MKLDNNMultiClassNonMaxSuppressionNode::created() const {
  return getType() == MulticlassNonMaxSuppression;
}

float MKLDNNMultiClassNonMaxSuppressionNode::intersectionOverUnion(
    const float *boxesI, const float *boxesJ, const bool normalized) {
  float yminI, xminI, ymaxI, xmaxI, yminJ, xminJ, ymaxJ, xmaxJ;
  const float norm = static_cast<float>(normalized == false);
  //  box format: y1, x1, y2, x2
  // yminI = (std::min)(boxesI[0], boxesI[2]);
  // xminI = (std::min)(boxesI[1], boxesI[3]);
  // ymaxI = (std::max)(boxesI[0], boxesI[2]);
  // xmaxI = (std::max)(boxesI[1], boxesI[3]);
  // yminJ = (std::min)(boxesJ[0], boxesJ[2]);
  // xminJ = (std::min)(boxesJ[1], boxesJ[3]);
  // ymaxJ = (std::max)(boxesJ[0], boxesJ[2]);
  // xmaxJ = (std::max)(boxesJ[1], boxesJ[3]);

  // to align with reference
  yminI = boxesI[0];
  xminI = boxesI[1];
  ymaxI = boxesI[2];
  xmaxI = boxesI[3];
  yminJ = boxesJ[0];
  xminJ = boxesJ[1];
  ymaxJ = boxesJ[2];
  xmaxJ = boxesJ[3];

  float areaI = (ymaxI - yminI + norm) * (xmaxI - xminI + norm);
  float areaJ = (ymaxJ - yminJ + norm) * (xmaxJ - xminJ + norm);
  if (areaI <= 0.f || areaJ <= 0.f)
    return 0.f;

  float intersection_area =
      (std::max)((std::min)(ymaxI, ymaxJ) - (std::max)(yminI, yminJ) + norm,
                 0.f) *
      (std::max)((std::min)(xmaxI, xmaxJ) - (std::max)(xminI, xminJ) + norm,
                 0.f);
  return intersection_area / (areaI + areaJ - intersection_area);
}

void MKLDNNMultiClassNonMaxSuppressionNode::nmsWithEta(
    const float *boxes, const float *scores, const SizeVector &boxesStrides,
    const SizeVector &scoresStrides, std::vector<filteredBoxes> &filtBoxes) {
  auto less = [](const boxInfo &l, const boxInfo &r) {
    return l.score < r.score || ((l.score == r.score) && (l.idx > r.idx));
  };

  auto func = [](float iou, float adaptive_threshold) {
    return iou <= adaptive_threshold ? 1.0f : 0.0f;
  };

  parallel_for2d(num_batches, num_classes, [&](int batch_idx, int class_idx) {
    if (class_idx != background_class) {
      std::vector<filteredBoxes> fb;
      const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
      const float *scoresPtr =
          scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

      std::priority_queue<boxInfo, std::vector<boxInfo>, decltype(less)>
          sorted_boxes(less);
      for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
        if (scoresPtr[box_idx] >= score_threshold) // algin with ref
          sorted_boxes.emplace(boxInfo({scoresPtr[box_idx], box_idx, 0}));
      }
      fb.reserve(sorted_boxes.size());
      if (sorted_boxes.size() > 0) {
        auto adaptive_threshold = iou_threshold;
        int max_out_box = (max_output_boxes_per_class > sorted_boxes.size())
                              ? sorted_boxes.size()
                              : max_output_boxes_per_class;
        while (max_out_box && !sorted_boxes.empty()) {
          boxInfo currBox = sorted_boxes.top();
          float origScore = currBox.score;
          sorted_boxes.pop();
          max_out_box--;

          bool box_is_selected = true;
          for (int idx = static_cast<int>(fb.size()) - 1;
               idx >= currBox.suppress_begin_index; idx--) {
            float iou = intersectionOverUnion(&boxesPtr[currBox.idx * 4],
                                              &boxesPtr[fb[idx].box_index * 4],
                                              normalized);
            currBox.score *= func(iou, adaptive_threshold);
            if (iou >= adaptive_threshold) {
              box_is_selected = false;
              break;
            }
            if (currBox.score <= score_threshold)
              break;
          }

          currBox.suppress_begin_index = fb.size();
          if (box_is_selected) {
            if (nms_eta < 1 && adaptive_threshold > 0.5) {
              adaptive_threshold *= nms_eta;
            }
            if (currBox.score == origScore) {
              fb.push_back({currBox.score, batch_idx, class_idx, currBox.idx});
              continue;
            }
            if (currBox.score > score_threshold) {
              sorted_boxes.push(currBox);
            }
          }
        }
      }
      numFiltBox[batch_idx][class_idx] = fb.size();
      size_t offset = batch_idx * num_classes * max_output_boxes_per_class +
                      class_idx * max_output_boxes_per_class;
      for (size_t i = 0; i < fb.size(); i++) {
        filtBoxes[offset + i] = fb[i];
      }
    }
  });
}

void MKLDNNMultiClassNonMaxSuppressionNode::nmsWithoutEta(
    const float *boxes, const float *scores, const SizeVector &boxesStrides,
    const SizeVector &scoresStrides, std::vector<filteredBoxes> &filtBoxes) {
  parallel_for2d(num_batches, num_classes, [&](int batch_idx, int class_idx) {
    if (class_idx != background_class) {
      const float *boxesPtr = boxes + batch_idx * boxesStrides[0];
      const float *scoresPtr =
          scores + batch_idx * scoresStrides[0] + class_idx * scoresStrides[1];

      std::vector<std::pair<float, int>> sorted_boxes;
      for (int box_idx = 0; box_idx < num_boxes; box_idx++) {
        if (scoresPtr[box_idx] >= score_threshold) // algin with ref
          sorted_boxes.emplace_back(
              std::make_pair(scoresPtr[box_idx], box_idx));
      }

      int io_selection_size = 0;
      if (sorted_boxes.size() > 0) {
        parallel_sort(
            sorted_boxes.begin(), sorted_boxes.end(),
            [](const std::pair<float, int> &l, const std::pair<float, int> &r) {
              return (l.first > r.first ||
                      ((l.first == r.first) && (l.second < r.second)));
            });
        int offset = batch_idx * num_classes * max_output_boxes_per_class +
                     class_idx * max_output_boxes_per_class;
        filtBoxes[offset + 0] =
            filteredBoxes(sorted_boxes[0].first, batch_idx, class_idx,
                          sorted_boxes[0].second);
        io_selection_size++;
        int max_out_box = (max_output_boxes_per_class > sorted_boxes.size())
                              ? sorted_boxes.size()
                              : max_output_boxes_per_class;
        for (size_t box_idx = 1; box_idx < max_out_box; box_idx++) {
          bool box_is_selected = true;
          for (int idx = io_selection_size - 1; idx >= 0; idx--) {
            float iou = intersectionOverUnion(
                &boxesPtr[sorted_boxes[box_idx].second * 4],
                &boxesPtr[filtBoxes[offset + idx].box_index * 4], normalized);
            if (iou >= iou_threshold) {
              box_is_selected = false;
              break;
            }
          }

          if (box_is_selected) {
            filtBoxes[offset + io_selection_size] =
                filteredBoxes(sorted_boxes[box_idx].first, batch_idx, class_idx,
                              sorted_boxes[box_idx].second);
            io_selection_size++;
          }
        }
      }
      numFiltBox[batch_idx][class_idx] = io_selection_size;
    }
  });
}

void MKLDNNMultiClassNonMaxSuppressionNode::checkPrecision(
    const Precision prec, const std::vector<Precision> precList,
    const std::string name, const std::string type) {
  if (std::find(precList.begin(), precList.end(), prec) == precList.end())
    IE_THROW() << errorPrefix << "has unsupported '" << name << "' " << type
               << " precision: " << prec;
}

void MKLDNNMultiClassNonMaxSuppressionNode::checkOutput(
    const SizeVector &dims, const std::vector<Precision> precList,
    const std::string name, const size_t port) {
  checkPrecision(getOriginalOutputPrecisionAtPort(port), precList, name,
                 outType);
}

REG_MKLDNN_PRIM_FOR(MKLDNNMultiClassNonMaxSuppressionNode,
                    MulticlassNonMaxSuppression)