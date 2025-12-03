//============================================================================
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//============================================================================

#ifndef NATIVEINFERENCE_INFERENCE_H
#define NATIVEINFERENCE_INFERENCE_H

#include "zdl/DlSystem/TensorShape.hpp"
#include "zdl/DlSystem/TensorMap.hpp"
#include "zdl/DlSystem/TensorShapeMap.hpp"
#include "zdl/DlSystem/IUserBufferFactory.hpp"
#include "zdl/DlSystem/IUserBuffer.hpp"
#include "zdl/DlSystem/UserBufferMap.hpp"
#include "zdl/DlSystem/IBufferAttributes.hpp"

#include "zdl/DlSystem/StringList.hpp"

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"
#include "zdl/DlSystem/DlVersion.hpp"
#include "zdl/DlSystem/DlEnums.hpp"
#include "zdl/DlSystem/String.hpp"
#include "zdl/DlContainer/IDlContainer.hpp"
#include "zdl/SNPE/SNPEBuilder.hpp"

#include "zdl/DlSystem/ITensor.hpp"
#include "zdl/DlSystem/ITensorFactory.hpp"

#include <unordered_map>
#include "android/log.h"

#include <opencv2/opencv.hpp>

#define  LOG_TAG    "SNPE_INF"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG,__VA_ARGS__)

std::string build_depth_network(const uint8_t * dlc_buffer, const size_t dlc_size, const char runtime_arg);
bool execute_depth(cv::Mat &rgbaImage, int orig_width, int orig_height,
                   std::vector<float> &normalizedDepth,
                   float &minValue, float &maxValue);

bool SetAdspLibraryPath(std::string nativeLibPath);

int getDepthInputWidth();
int getDepthInputHeight();
int getDepthOutputWidth();
int getDepthOutputHeight();

#endif //NATIVEINFERENCE_INFERENCE_H
