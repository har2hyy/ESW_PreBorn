//============================================================================
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//============================================================================

#include <jni.h>
#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstring>

#include <hpp/inference.h>

#include "android/log.h"

#include "hpp/CheckRuntime.hpp"
#include "hpp/SetBuilderOptions.hpp"
#include "hpp/Util.hpp"
#include "LoadContainer.hpp"
#include "LoadInputTensor.hpp"

#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <android/trace.h>
#include <dlfcn.h>

static std::unique_ptr<zdl::SNPE::SNPE> snpe_depth;
static std::mutex depthMutex;
static zdl::DlSystem::RuntimeList runtimeList;
static bool useUserSuppliedBuffers = true;
static bool useIntBuffer = false;
static int bitWidth = 32;

static zdl::DlSystem::UserBufferMap depthInputMap, depthOutputMap;
static std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> depthInputUserBuffers;
static std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> depthOutputUserBuffers;
static std::unordered_map<std::string, std::vector<float32_t>> depthInputStorage;
static std::unordered_map<std::string, std::vector<float32_t>> depthOutputStorage;

static std::string depthInputTensorName;
static std::string depthOutputTensorName;
static bool depthInputIsNCHW = false;
static bool depthOutputIsNCHW = false;
static int depthInputWidth = 0;
static int depthInputHeight = 0;
static int depthOutputWidth = 0;
static int depthOutputHeight = 0;
static int depthOutputChannels = 1;

static void extractTensorMetadata() {
    if (!snpe_depth) {
        return;
    }

    const auto &inputNamesOpt = snpe_depth->getInputTensorNames();
    if (inputNamesOpt && (*inputNamesOpt).size() > 0) {
        depthInputTensorName = (*inputNamesOpt).at(0);
        auto attrs = snpe_depth->getInputOutputBufferAttributes(depthInputTensorName.c_str());
        if (attrs) {
            const auto &shape = (*attrs)->getDims();
            if (shape.rank() == 4) {
                size_t dim0 = shape[0];
                size_t dim1 = shape[1];
                size_t dim2 = shape[2];
                size_t dim3 = shape[3];
                if (dim1 == 3) {
                    depthInputIsNCHW = true;
                    depthInputWidth = static_cast<int>(dim3);
                    depthInputHeight = static_cast<int>(dim2);
                } else {
                    depthInputIsNCHW = false;
                    depthInputHeight = static_cast<int>(dim1);
                    depthInputWidth = static_cast<int>(dim2);
                }
            }
        }
    }

    const auto &outputNamesOpt = snpe_depth->getOutputTensorNames();
    if (outputNamesOpt && (*outputNamesOpt).size() > 0) {
        depthOutputTensorName = (*outputNamesOpt).at(0);
        auto attrs = snpe_depth->getInputOutputBufferAttributes(depthOutputTensorName.c_str());
        if (attrs) {
            const auto &shape = (*attrs)->getDims();
            if (shape.rank() >= 3) {
                if (shape.rank() == 4) {
                    // Assume NCHW or NHWC
                    if (shape[1] == 1) {
                        depthOutputIsNCHW = true;
                        depthOutputChannels = static_cast<int>(shape[1]);
                        depthOutputHeight = static_cast<int>(shape[2]);
                        depthOutputWidth = static_cast<int>(shape[3]);
                    } else {
                        depthOutputIsNCHW = false;
                        depthOutputHeight = static_cast<int>(shape[1]);
                        depthOutputWidth = static_cast<int>(shape[2]);
                        depthOutputChannels = static_cast<int>(shape[3]);
                    }
                } else if (shape.rank() == 3) {
                    depthOutputIsNCHW = false;
                    depthOutputChannels = static_cast<int>(shape[2]);
                    depthOutputHeight = static_cast<int>(shape[0]);
                    depthOutputWidth = static_cast<int>(shape[1]);
                }
            }
        }
    }
}

static void createUserBuffer(zdl::DlSystem::UserBufferMap &userBufferMap,
                             std::unordered_map<std::string, std::vector<float32_t>> &applicationBuffers,
                             std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> &snpeUserBuffers,
                             std::unique_ptr<zdl::SNPE::SNPE> &snpe,
                             const char *name,
                             bool isTfNBuffer,
                             int bitWidth) {

    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name);
    if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for tensor ") + name);

    const zdl::DlSystem::TensorShape &bufferShape = (*bufferAttributesOpt)->getDims();
    size_t bufferElementSize = isTfNBuffer ? bitWidth / 8 : sizeof(float);

    int num_dims = bufferShape.rank();
    std::vector<size_t> strides(num_dims);
    strides.back() = bufferElementSize;
    size_t stride = strides.back();
    for (int i = num_dims - 1; i > 0; --i) {
        stride *= bufferShape[i];
        strides[i - 1] = stride;
    }

    size_t bufSize = bufferElementSize;
    for (int i = 0; i < num_dims; ++i) {
        bufSize *= bufferShape[i];
    }

    std::unique_ptr<zdl::DlSystem::UserBufferEncoding> userBufferEncoding;
    if (isTfNBuffer) {
        userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingTfN>(
                new zdl::DlSystem::UserBufferEncodingTfN(0, 1.0, bitWidth));
    } else {
        userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingFloat>(
                new zdl::DlSystem::UserBufferEncodingFloat());
    }

    applicationBuffers.emplace(name, std::vector<float32_t>(bufSize / sizeof(float32_t)));
    zdl::DlSystem::IUserBufferFactory &ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
    snpeUserBuffers.push_back(
            ubFactory.createUserBuffer(applicationBuffers.at(name).data(),
                                       bufSize,
                                       strides,
                                       userBufferEncoding.get()));
    if (snpeUserBuffers.back() == nullptr) {
        throw std::runtime_error("Error while creating user buffer");
    }

    userBufferMap.add(name, snpeUserBuffers.back().get());
}

static void createInputBufferMap(zdl::DlSystem::UserBufferMap &inputMap,
                                 std::unordered_map<std::string, std::vector<float32_t>> &applicationBuffers,
                                 std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> &snpeUserBuffers,
                                 std::unique_ptr<zdl::SNPE::SNPE> &snpe,
                                 bool isTfNBuffer,
                                 int bitWidth) {
    const auto &inputNamesOpt = snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    for (const char *name : *inputNamesOpt) {
        createUserBuffer(inputMap, applicationBuffers, snpeUserBuffers, snpe, name, isTfNBuffer, bitWidth);
    }
}

static void createOutputBufferMap(zdl::DlSystem::UserBufferMap &outputMap,
                                  std::unordered_map<std::string, std::vector<float32_t>> &applicationBuffers,
                                  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> &snpeUserBuffers,
                                  std::unique_ptr<zdl::SNPE::SNPE> &snpe,
                                  bool isTfNBuffer,
                                  int bitWidth) {
    const auto &outputNamesOpt = snpe->getOutputTensorNames();
    if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names");
    for (const char *name : *outputNamesOpt) {
        createUserBuffer(outputMap, applicationBuffers, snpeUserBuffers, snpe, name, isTfNBuffer, bitWidth);
    }
}

static void preprocessDepth(std::vector<float32_t> &dest, const cv::Mat &rgba) {
    if (depthInputWidth == 0 || depthInputHeight == 0) {
        return;
    }

    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(depthInputWidth, depthInputHeight), 0, 0, cv::INTER_CUBIC);

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};

    if (depthInputIsNCHW) {
        int channelStride = depthInputWidth * depthInputHeight;
        for (int y = 0; y < depthInputHeight; ++y) {
            const cv::Vec3b *row = resized.ptr<cv::Vec3b>(y);
            for (int x = 0; x < depthInputWidth; ++x) {
                const cv::Vec3b &pix = row[x];
                float b = pix[0] / 255.0f;
                float g = pix[1] / 255.0f;
                float r = pix[2] / 255.0f;
                float rNorm = (r - mean[0]) / std[0];
                float gNorm = (g - mean[1]) / std[1];
                float bNorm = (b - mean[2]) / std[2];
                int offset = y * depthInputWidth + x;
                dest[offset] = rNorm;
                dest[channelStride + offset] = gNorm;
                dest[2 * channelStride + offset] = bNorm;
            }
        }
    } else {
        size_t idx = 0;
        for (int y = 0; y < depthInputHeight; ++y) {
            const cv::Vec3b *row = resized.ptr<cv::Vec3b>(y);
            for (int x = 0; x < depthInputWidth; ++x) {
                const cv::Vec3b &pix = row[x];
                float b = pix[0] / 255.0f;
                float g = pix[1] / 255.0f;
                float r = pix[2] / 255.0f;
                dest[idx++] = (r - mean[0]) / std[0];
                dest[idx++] = (g - mean[1]) / std[1];
                dest[idx++] = (b - mean[2]) / std[2];
            }
        }
    }
}

static bool convertDepthOutput(std::vector<float32_t> &rawOutput, cv::Mat &depthMap) {
    if (depthOutputWidth == 0 || depthOutputHeight == 0) {
        LOGE("Depth output dimensions not configured");
        return false;
    }

    depthMap.create(depthOutputHeight, depthOutputWidth, CV_32FC1);
    int totalPixels = depthOutputWidth * depthOutputHeight;

    if (depthOutputIsNCHW || depthOutputChannels == 1) {
        std::memcpy(depthMap.data, rawOutput.data(), totalPixels * sizeof(float));
    } else {
        for (int y = 0; y < depthOutputHeight; ++y) {
            float *row = depthMap.ptr<float>(y);
            for (int x = 0; x < depthOutputWidth; ++x) {
                int idx = (y * depthOutputWidth + x) * depthOutputChannels;
                row[x] = rawOutput[idx];
            }
        }
    }
    return true;
}

std::string build_depth_network(const uint8_t *dlc_buffer, const size_t dlc_size, const char runtime_arg) {
    std::string outputLogger;
    bool usingInitCaching = false;

    auto container = loadContainerFromBuffer(dlc_buffer, dlc_size);
    if (container == nullptr) {
        LOGE("Error while opening the container file.");
        return "Error while opening the container file.\n";
    }

    runtimeList.clear();
    zdl::DlSystem::Runtime_t selectedRuntime = zdl::DlSystem::Runtime_t::CPU;
    if (runtime_arg == 'D') {
        selectedRuntime = zdl::DlSystem::Runtime_t::DSP;
    } else if (runtime_arg == 'G') {
        selectedRuntime = zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID;
    }

    if (!runtimeList.add(checkRuntime(selectedRuntime))) {
        LOGE("Cannot set runtime");
        return outputLogger + "\nCannot set runtime";
    }

    std::lock_guard<std::mutex> lock(depthMutex);
    snpe_depth = setBuilderOptions(container, selectedRuntime, runtimeList,
                                   useUserSuppliedBuffers, usingInitCaching);

    if (snpe_depth == nullptr) {
        LOGE("SNPE builder failed for depth model");
        return outputLogger + "SNPE Prepare failed";
    }

    // ===== RUNTIME VERIFICATION =====
    // SNPE 2.x doesn't expose getRuntime(), so we infer from initialization logs
    // and runtime list configuration. The selected runtime will execute unless
    // fallback occurs (which SNPE handles internally).
    
    const char* requestedRuntimeStr = nullptr;
    if (selectedRuntime == zdl::DlSystem::Runtime_t::CPU) {
        requestedRuntimeStr = "CPU";
    } else if (selectedRuntime == zdl::DlSystem::Runtime_t::DSP) {
        requestedRuntimeStr = "DSP/NPU";
    } else if (selectedRuntime == zdl::DlSystem::Runtime_t::GPU ||
               selectedRuntime == zdl::DlSystem::Runtime_t::GPU_FLOAT16 ||
               selectedRuntime == zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID) {
        requestedRuntimeStr = "GPU";
    } else if (selectedRuntime == zdl::DlSystem::Runtime_t::AIP_FIXED_TF) {
        requestedRuntimeStr = "AIP";
    } else {
        requestedRuntimeStr = "UNKNOWN";
    }
    
    LOGI("========================================");
    LOGI("SNPE RUNTIME VERIFICATION");
    LOGI("========================================");
    LOGI("Requested runtime: %c (D=DSP, G=GPU, C=CPU)", runtime_arg);
    LOGI("Configured runtime: %s", requestedRuntimeStr ? requestedRuntimeStr : "NONE");
    
    if (selectedRuntime == zdl::DlSystem::Runtime_t::DSP) {
        LOGI("✓ NPU/HTP CONFIGURED");
        LOGI("  If you see this message, the DLC will attempt to run on the NPU.");
        LOGI("  SNPE will automatically fall back to CPU if NPU is unavailable.");
        LOGI("  Monitor system resources to confirm NPU execution:");
        LOGI("    - Check /sys/kernel/debug/rpmh_master_stats for HVX activity");
        LOGI("    - Use 'adb shell top' to verify low CPU usage during inference");
        outputLogger += "\n✓ Configured for NPU/HTP execution";
    } else {
        LOGI("Runtime: %s", requestedRuntimeStr);
        outputLogger += std::string("\nRuntime: ") + requestedRuntimeStr;
    }
    LOGI("========================================");
    // ===== END RUNTIME VERIFICATION =====

    depthInputMap.clear();
    depthOutputMap.clear();
    depthInputUserBuffers.clear();
    depthOutputUserBuffers.clear();
    depthInputStorage.clear();
    depthOutputStorage.clear();

    createInputBufferMap(depthInputMap, depthInputStorage, depthInputUserBuffers,
                         snpe_depth, useIntBuffer, bitWidth);
    createOutputBufferMap(depthOutputMap, depthOutputStorage, depthOutputUserBuffers,
                          snpe_depth, useIntBuffer, bitWidth);

    extractTensorMetadata();

    outputLogger += "\nDepth model ready";
    return outputLogger;
}

bool execute_depth(cv::Mat &rgbaImage, int orig_width, int orig_height,
                   std::vector<float> &normalizedDepth,
                   float &minValue, float &maxValue) {
    std::lock_guard<std::mutex> lock(depthMutex);
    if (!snpe_depth) {
        LOGE("Depth network not initialized");
        return false;
    }

    if (depthInputStorage.find(depthInputTensorName) == depthInputStorage.end()) {
        LOGE("Depth input tensor storage missing");
        return false;
    }

    preprocessDepth(depthInputStorage.at(depthInputTensorName), rgbaImage);

    bool execStatus = snpe_depth->execute(depthInputMap, depthOutputMap);
    if (!execStatus) {
        LOGE("Depth network execution failed");
        return false;
    }

    auto &rawOutput = depthOutputStorage.at(depthOutputTensorName);
    cv::Mat depthModel;
    if (!convertDepthOutput(rawOutput, depthModel)) {
        return false;
    }

    cv::Mat depthResized;
    cv::resize(depthModel, depthResized, cv::Size(orig_width, orig_height), 0, 0, cv::INTER_CUBIC);

    double minRaw, maxRaw;
    cv::minMaxLoc(depthResized, &minRaw, &maxRaw);
    minValue = static_cast<float>(minRaw);
    maxValue = static_cast<float>(maxRaw);
    float denom = (maxValue - minValue);
    if (denom < 1e-6f) {
        denom = 1.0f;
    }

    normalizedDepth.resize(orig_width * orig_height);
    for (int y = 0; y < orig_height; ++y) {
        const float *row = depthResized.ptr<float>(y);
        for (int x = 0; x < orig_width; ++x) {
            float value = (row[x] - minValue) / denom;
            value = std::max(0.0f, std::min(1.0f, value));
            normalizedDepth[y * orig_width + x] = value;
        }
    }

    return true;
}

int getDepthInputWidth() { return depthInputWidth; }
int getDepthInputHeight() { return depthInputHeight; }
int getDepthOutputWidth() { return depthOutputWidth; }
int getDepthOutputHeight() { return depthOutputHeight; }


