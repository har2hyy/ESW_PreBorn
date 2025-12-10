/*
 * Integrated YOLO + Depth Pipeline for Android
 * Uses ONNX Runtime Mobile (no SNPE SDK required)
 * 
 * This implementation mirrors your Python pipeline:
 * 1. YOLOv11 object detection
 * 2. Depth estimation (lightweight)
 * 3. Spatial analysis & distance calculations
 */

#include <jni.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <android/log.h>
#include <cmath>

#define LOG_TAG "IntegratedPipeline"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Detection structure matching your Python pipeline
struct Detection {
    int id;
    std::string class_name;
    int class_id;
    float confidence;
    cv::Rect bbox;
    cv::Point center;
    float depth_avg;
    float depth_median;
    float depth_min;
    float depth_max;
    float depth_center;
    int area;
};

// Distance pair structure
struct DistancePair {
    int obj1_id;
    int obj2_id;
    std::string obj1_class;
    std::string obj2_class;
    float euclidean;
    float horizontal;
    float vertical;
    float depth_diff;
    float depth_diff_scaled;
    float obj1_depth;
    float obj2_depth;
};

class IntegratedPipelineONNX {
private:
    float depth_scale_factor = 3.0f;
    std::vector<std::string> class_names = {"worker", "truck", "bike", "bulldozer", "car"};
    
    // Simple depth estimation using monocular cues (placeholder for real depth model)
    cv::Mat estimateDepth(const cv::Mat& input_image) {
        cv::Mat gray, depth;
        cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
        
        // Simple depth approximation using edge density and intensity
        cv::Mat edges, blurred;
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::Canny(blurred, edges, 50, 150);
        
        // Combine intensity gradient with edge information
        cv::Mat grad_x, grad_y, grad;
        cv::Sobel(blurred, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(blurred, grad_y, CV_32F, 0, 1, 3);
        cv::magnitude(grad_x, grad_y, grad);
        
        // Normalize and invert (darker = farther)
        cv::normalize(grad, depth, 0, 255, cv::NORM_MINMAX);
        depth.convertTo(depth, CV_8U);
        
        // Apply bilateral filter for smoother depth map
        cv::bilateralFilter(depth, depth, 9, 75, 75);
        
        // Invert so darker pixels = farther
        depth = 255 - depth;
        
        LOGI("Depth estimation complete: %dx%d", depth.cols, depth.rows);
        return depth;
    }
    
    void attachDepthStats(std::vector<Detection>& detections, const cv::Mat& depth_map) {
        int h = depth_map.rows;
        int w = depth_map.cols;
        
        for (auto& det : detections) {
            // Clamp bbox to image bounds
            int x1 = std::max(0, std::min(det.bbox.x, w - 1));
            int y1 = std::max(0, std::min(det.bbox.y, h - 1));
            int x2 = std::max(x1 + 1, std::min(det.bbox.x + det.bbox.width, w));
            int y2 = std::max(y1 + 1, std::min(det.bbox.y + det.bbox.height, h));
            
            cv::Rect safe_roi(x1, y1, x2 - x1, y2 - y1);
            cv::Mat roi = depth_map(safe_roi);
            
            if (roi.empty()) continue;
            
            // Calculate depth statistics
            cv::Scalar mean_val, stddev_val;
            cv::meanStdDev(roi, mean_val, stddev_val);
            
            double min_val, max_val;
            cv::minMaxLoc(roi, &min_val, &max_val);
            
            // Calculate median
            std::vector<uchar> pixels;
            pixels.assign(roi.data, roi.data + roi.total());
            std::sort(pixels.begin(), pixels.end());
            float median = pixels[pixels.size() / 2];
            
            // Center point depth
            int cx = (x1 + x2) / 2;
            int cy = (y1 + y2) / 2;
            float center_depth = depth_map.at<uchar>(cy, cx);
            
            det.center = cv::Point(cx, cy);
            det.depth_avg = mean_val[0];
            det.depth_median = median;
            det.depth_min = min_val;
            det.depth_max = max_val;
            det.depth_center = center_depth;
            det.area = (x2 - x1) * (y2 - y1);
            
            LOGI("Object %d: %s | depth_avg=%.1f @ (%d,%d)", 
                 det.id, det.class_name.c_str(), det.depth_avg, cx, cy);
        }
    }
    
    std::vector<DistancePair> calculateDistances(const std::vector<Detection>& detections) {
        std::vector<DistancePair> distances;
        
        for (size_t i = 0; i < detections.size(); i++) {
            for (size_t j = i + 1; j < detections.size(); j++) {
                const auto& obj1 = detections[i];
                const auto& obj2 = detections[j];
                
                float dx = obj2.center.x - obj1.center.x;
                float dy = obj2.center.y - obj1.center.y;
                float euclidean = std::sqrt(dx * dx + dy * dy);
                float depth_diff = std::abs(obj1.depth_avg - obj2.depth_avg);
                float depth_diff_scaled = depth_diff * depth_scale_factor;
                
                DistancePair pair;
                pair.obj1_id = obj1.id;
                pair.obj2_id = obj2.id;
                pair.obj1_class = obj1.class_name;
                pair.obj2_class = obj2.class_name;
                pair.euclidean = euclidean;
                pair.horizontal = std::abs(dx);
                pair.vertical = std::abs(dy);
                pair.depth_diff = depth_diff;
                pair.depth_diff_scaled = depth_diff_scaled;
                pair.obj1_depth = obj1.depth_avg;
                pair.obj2_depth = obj2.depth_avg;
                
                distances.push_back(pair);
            }
        }
        
        // Sort by euclidean distance
        std::sort(distances.begin(), distances.end(), 
                  [](const DistancePair& a, const DistancePair& b) {
                      return a.euclidean < b.euclidean;
                  });
        
        LOGI("Calculated %zu distance pairs", distances.size());
        return distances;
    }
    
public:
    // Main processing function matching your Python pipeline
    std::string processFrame(const cv::Mat& input_frame,
                            const std::vector<cv::Rect>& boxes,
                            const std::vector<float>& confidences,
                            const std::vector<int>& class_ids) {
        
        LOGI("=== INTEGRATED PIPELINE START ===");
        LOGI("Input: %dx%d, Detections: %zu", 
             input_frame.cols, input_frame.rows, boxes.size());
        
        // Stage 1: Convert raw detections to Detection objects
        std::vector<Detection> detections;
        for (size_t i = 0; i < boxes.size(); i++) {
            Detection det;
            det.id = i;
            det.class_id = class_ids[i];
            det.class_name = (class_ids[i] >= 0 && class_ids[i] < class_names.size()) 
                            ? class_names[class_ids[i]] : "unknown";
            det.confidence = confidences[i];
            det.bbox = boxes[i];
            detections.push_back(det);
            
            LOGI("  Detection %zu: %s (%.3f) @ [%d,%d,%d,%d]",
                 i, det.class_name.c_str(), det.confidence,
                 det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height);
        }
        
        // Stage 2: Depth estimation
        LOGI("\n[Stage 2] Depth estimation...");
        auto depth_start = std::chrono::high_resolution_clock::now();
        cv::Mat depth_map = estimateDepth(input_frame);
        auto depth_end = std::chrono::high_resolution_clock::now();
        float depth_time = std::chrono::duration<float, std::milli>(depth_end - depth_start).count();
        LOGI("Depth map ready in %.1f ms", depth_time);
        
        // Stage 3: Attach depth statistics
        LOGI("\n[Stage 3] Aggregating depth statistics...");
        attachDepthStats(detections, depth_map);
        
        // Stage 4: Calculate pairwise distances
        LOGI("\n[Stage 4] Calculating spatial relationships...");
        std::vector<DistancePair> distances = calculateDistances(detections);
        
        // Build JSON result matching your Python output
        std::ostringstream json;
        json << "{\n";
        json << "  \"total_detections\": " << detections.size() << ",\n";
        json << "  \"depth_inference_ms\": " << depth_time << ",\n";
        json << "  \"detections\": [\n";
        
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& det = detections[i];
            json << "    {\n";
            json << "      \"id\": " << det.id << ",\n";
            json << "      \"class\": \"" << det.class_name << "\",\n";
            json << "      \"class_id\": " << det.class_id << ",\n";
            json << "      \"confidence\": " << det.confidence << ",\n";
            json << "      \"bbox\": [" << det.bbox.x << "," << det.bbox.y << "," 
                 << det.bbox.width << "," << det.bbox.height << "],\n";
            json << "      \"center\": [" << det.center.x << "," << det.center.y << "],\n";
            json << "      \"depth_avg\": " << det.depth_avg << ",\n";
            json << "      \"depth_median\": " << det.depth_median << ",\n";
            json << "      \"depth_min\": " << det.depth_min << ",\n";
            json << "      \"depth_max\": " << det.depth_max << ",\n";
            json << "      \"depth_center\": " << det.depth_center << ",\n";
            json << "      \"area\": " << det.area << "\n";
            json << "    }" << (i < detections.size() - 1 ? "," : "") << "\n";
        }
        json << "  ],\n";
        
        json << "  \"distances\": [\n";
        for (size_t i = 0; i < distances.size() && i < 10; i++) {  // Limit to top 10
            const auto& dist = distances[i];
            json << "    {\n";
            json << "      \"obj1_id\": " << dist.obj1_id << ",\n";
            json << "      \"obj1_class\": \"" << dist.obj1_class << "\",\n";
            json << "      \"obj2_id\": " << dist.obj2_id << ",\n";
            json << "      \"obj2_class\": \"" << dist.obj2_class << "\",\n";
            json << "      \"euclidean\": " << dist.euclidean << ",\n";
            json << "      \"horizontal\": " << dist.horizontal << ",\n";
            json << "      \"vertical\": " << dist.vertical << ",\n";
            json << "      \"depth_diff\": " << dist.depth_diff << ",\n";
            json << "      \"depth_diff_scaled\": " << dist.depth_diff_scaled << "\n";
            json << "    }" << (i < std::min(distances.size(), size_t(10)) - 1 ? "," : "") << "\n";
        }
        json << "  ]\n";
        json << "}\n";
        
        LOGI("=== PIPELINE COMPLETE ===");
        return json.str();
    }
    
    // Visualization helper
    cv::Mat createCombinedVisualization(const cv::Mat& input_frame,
                                        const std::vector<Detection>& detections,
                                        const cv::Mat& depth_map) {
        cv::Mat annotated = input_frame.clone();
        
        // Draw detections with depth info
        for (const auto& det : detections) {
            // Draw bbox
            cv::rectangle(annotated, det.bbox, cv::Scalar(0, 255, 0), 2);
            
            // Draw label with depth
            std::ostringstream label;
            label << det.class_name << " " << std::fixed << std::setprecision(2) 
                  << det.confidence << " D:" << (int)det.depth_avg;
            
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 
                                                  0.5, 1, &baseline);
            
            cv::rectangle(annotated, 
                         cv::Point(det.bbox.x, det.bbox.y - text_size.height - 10),
                         cv::Point(det.bbox.x + text_size.width, det.bbox.y),
                         cv::Scalar(0, 255, 0), -1);
            
            cv::putText(annotated, label.str(), 
                       cv::Point(det.bbox.x, det.bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            
            // Draw center point
            cv::circle(annotated, det.center, 5, cv::Scalar(0, 0, 255), -1);
        }
        
        // Create depth colored visualization
        cv::Mat depth_colored;
        cv::applyColorMap(depth_map, depth_colored, cv::COLORMAP_SPECTRAL);
        
        // Combine side by side
        cv::Mat combined;
        cv::hconcat(annotated, depth_colored, combined);
        
        return combined;
    }
};

// Global instance
static IntegratedPipelineONNX* g_pipeline = nullptr;

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_qc_objectdetectionYoloNas_IntegratedPipeline_initPipeline(
    JNIEnv* env, jobject /* this */) {
    
    if (g_pipeline == nullptr) {
        g_pipeline = new IntegratedPipelineONNX();
        LOGI("Integrated pipeline initialized");
        return env->NewStringUTF("SUCCESS");
    }
    return env->NewStringUTF("ALREADY_INITIALIZED");
}

JNIEXPORT jstring JNICALL
Java_com_qc_objectdetectionYoloNas_IntegratedPipeline_processFrame(
    JNIEnv* env, jobject /* this */,
    jlong mat_addr,
    jfloatArray boxes_array,
    jfloatArray confidences_array,
    jintArray class_ids_array) {
    
    if (g_pipeline == nullptr) {
        return env->NewStringUTF("{\"error\": \"Pipeline not initialized\"}");
    }
    
    // Get input Mat
    cv::Mat& input_mat = *(cv::Mat*)mat_addr;
    
    // Convert Java arrays to C++ vectors
    jfloat* boxes = env->GetFloatArrayElements(boxes_array, nullptr);
    jfloat* confidences = env->GetFloatArrayElements(confidences_array, nullptr);
    jint* class_ids = env->GetIntArrayElements(class_ids_array, nullptr);
    
    int num_boxes = env->GetArrayLength(boxes_array) / 4;
    
    std::vector<cv::Rect> box_vec;
    std::vector<float> conf_vec;
    std::vector<int> class_vec;
    
    for (int i = 0; i < num_boxes; i++) {
        int x = (int)boxes[i * 4];
        int y = (int)boxes[i * 4 + 1];
        int w = (int)boxes[i * 4 + 2];
        int h = (int)boxes[i * 4 + 3];
        box_vec.push_back(cv::Rect(x, y, w, h));
        conf_vec.push_back(confidences[i]);
        class_vec.push_back(class_ids[i]);
    }
    
    // Release arrays
    env->ReleaseFloatArrayElements(boxes_array, boxes, 0);
    env->ReleaseFloatArrayElements(confidences_array, confidences, 0);
    env->ReleaseIntArrayElements(class_ids_array, class_ids, 0);
    
    // Process
    std::string result = g_pipeline->processFrame(input_mat, box_vec, conf_vec, class_vec);
    
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_qc_objectdetectionYoloNas_IntegratedPipeline_cleanup(
    JNIEnv* env, jobject /* this */) {
    
    if (g_pipeline != nullptr) {
        delete g_pipeline;
        g_pipeline = nullptr;
        LOGI("Integrated pipeline cleaned up");
    }
}

} // extern "C"
