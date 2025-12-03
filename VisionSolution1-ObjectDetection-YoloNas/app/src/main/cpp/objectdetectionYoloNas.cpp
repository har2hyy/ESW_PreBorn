//============================================================================
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//============================================================================

#include <opencv2/core.hpp>
using namespace cv;
#include <jni.h>
#include <string>
#include <vector>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "hpp/inference.h"
#include "hpp/Util.hpp"

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"

namespace {
    constexpr const char* kDepthDlcAsset = "depth_anything_v2.dlc";
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_qc_objectdetectionYoloNas_DepthSnpeBridge_nativeInit(
        JNIEnv* env,
        jobject /* this */,
        jobject asset_manager,
        jstring native_lib_path,
        jchar runtime_char) {

    const char *nativePath = env->GetStringUTFChars(native_lib_path, nullptr);
    std::string initLog;
    if (!SetAdspLibraryPath(nativePath)) {
        initLog = "Failed to set ADSP library path";
        env->ReleaseStringUTFChars(native_lib_path, nativePath);
        return env->NewStringUTF(initLog.c_str());
    }
    env->ReleaseStringUTFChars(native_lib_path, nativePath);

    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    AAsset* asset = AAssetManager_open(mgr, kDepthDlcAsset, AASSET_MODE_UNKNOWN);
    if (asset == nullptr) {
        initLog = "Failed to open depth DLC";
        return env->NewStringUTF(initLog.c_str());
    }

    const off_t dlcSize = AAsset_getLength(asset);
    std::vector<char> buffer(static_cast<size_t>(dlcSize));
    AAsset_read(asset, buffer.data(), dlcSize);
    AAsset_close(asset);

    initLog = build_depth_network(reinterpret_cast<const uint8_t*>(buffer.data()), dlcSize, runtime_char);
    return env->NewStringUTF(initLog.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_qc_objectdetectionYoloNas_DepthSnpeBridge_nativeInferDepth(
        JNIEnv* env,
        jobject /* this */,
        jintArray rgbaPixels,
        jint width,
        jint height,
        jfloatArray depthBuffer,
        jfloatArray statsBuffer) {

    const int pixelCount = width * height;
    if (env->GetArrayLength(depthBuffer) < pixelCount || env->GetArrayLength(statsBuffer) < 2) {
        LOGE("Output buffers too small");
        return JNI_FALSE;
    }

    jint *pixels = env->GetIntArrayElements(rgbaPixels, nullptr);
    if (pixels == nullptr) {
        LOGE("Failed to access pixel buffer");
        return JNI_FALSE;
    }

    cv::Mat rgba(height, width, CV_8UC4);
    unsigned char *dst = rgba.data;
    for (int i = 0; i < pixelCount; ++i) {
        unsigned int color = static_cast<unsigned int>(pixels[i]);
        dst[4 * i + 0] = (color >> 16) & 0xFF; // R
        dst[4 * i + 1] = (color >> 8) & 0xFF;  // G
        dst[4 * i + 2] = (color) & 0xFF;       // B
        dst[4 * i + 3] = (color >> 24) & 0xFF; // A
    }
    env->ReleaseIntArrayElements(rgbaPixels, pixels, 0);

    std::vector<float> normalized;
    float minValue = 0.f;
    float maxValue = 0.f;

    if (!execute_depth(rgba, width, height, normalized, minValue, maxValue)) {
        return JNI_FALSE;
    }

    env->SetFloatArrayRegion(depthBuffer, 0, pixelCount, normalized.data());
    float stats[2] = {minValue, maxValue};
    env->SetFloatArrayRegion(statsBuffer, 0, 2, stats);
    return JNI_TRUE;
}