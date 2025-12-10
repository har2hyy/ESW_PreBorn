package com.qc.objectdetectionYoloNas;

import android.graphics.Bitmap;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.json.JSONObject;

/**
 * Integrated YOLO + Depth Pipeline for Android
 * Mirrors the Python integrated_pipeline_onnx.py functionality
 */
public class IntegratedPipeline {
    
    static {
        System.loadLibrary("objectdetectionYoloNas");
    }
    
    // Native methods
    private native String initPipeline();
    private native String processFrame(long matAddr, float[] boxes, float[] confidences, int[] classIds);
    private native void cleanup();
    
    private boolean isInitialized = false;
    
    /**
     * Initialize the integrated pipeline
     */
    public boolean initialize() {
        if (!isInitialized) {
            String result = initPipeline();
            isInitialized = result.equals("SUCCESS") || result.equals("ALREADY_INITIALIZED");
            return isInitialized;
        }
        return true;
    }
    
    /**
     * Process a frame with detections through the integrated pipeline
     * 
     * @param inputMat OpenCV Mat of the input frame
     * @param detections List of DetectionWithDepth objects from YOLO
     * @return JSON string with depth-enhanced analysis
     */
    public String process(Mat inputMat, java.util.List<DetectionWithDepth> detections) {
        if (!isInitialized) {
            return "{\"error\": \"Pipeline not initialized\"}";
        }
        
        // Convert detections to arrays for JNI
        int numDetections = detections.size();
        float[] boxes = new float[numDetections * 4];
        float[] confidences = new float[numDetections];
        int[] classIds = new int[numDetections];
        
        for (int i = 0; i < numDetections; i++) {
            DetectionWithDepth det = detections.get(i);
            boxes[i * 4] = det.getX();
            boxes[i * 4 + 1] = det.getY();
            boxes[i * 4 + 2] = det.getWidth();
            boxes[i * 4 + 3] = det.getHeight();
            confidences[i] = det.getConfidence();
            classIds[i] = det.getClassId();
        }
        
        // Call native processing
        return processFrame(inputMat.getNativeObjAddr(), boxes, confidences, classIds);
    }
    
    /**
     * Cleanup resources
     */
    public void release() {
        if (isInitialized) {
            cleanup();
            isInitialized = false;
        }
    }
    
    @Override
    protected void finalize() throws Throwable {
        release();
        super.finalize();
    }
}
