package com.qc.objectdetectionYoloNas;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class TFLiteRunner {
    private static final String MODEL_FILE = "best_float32.tflite";
    private static final int MODEL_INPUT_W = 320;
    private static final int MODEL_INPUT_H = 320;
    private static final int OUTPUT_SIZE_N = 2100;

    private final Interpreter tflite;
    private int actualInputWidth;
    private int actualInputHeight;
    private int actualOutputN;
    private int actualOutputFeatures;

    public static class Det {
        public final float x1, y1, x2, y2, score;
        public final int cls;
        public Det(float x1, float y1, float x2, float y2, float score, int cls) {
            this.x1=x1; this.y1=y1; this.x2=x2; this.y2=y2; this.score=score; this.cls=cls;
        }
        public String toString() {
            return String.format("Det{[%.1f,%.1f,%.1f,%.1f], score=%.3f, cls=%d}", x1, y1, x2, y2, score, cls);
        }
    }

    public TFLiteRunner(Context ctx) throws IOException {
        Interpreter.Options opts = new Interpreter.Options();
        opts.setNumThreads(4);
        opts.setUseNNAPI(false);
        
        tflite = new Interpreter(loadModelFile(ctx, MODEL_FILE), opts);
        
        if (tflite.getInputTensorCount() > 0) {
            int[] inputShape = tflite.getInputTensor(0).shape();
            actualInputHeight = inputShape[1];
            actualInputWidth = inputShape[2];
            android.util.Log.d("TFLiteRunner", "Input: [" + inputShape[0] + ", " + inputShape[1] + ", " + inputShape[2] + ", " + inputShape[3] + "]");
        } else {
            actualInputWidth = MODEL_INPUT_W;
            actualInputHeight = MODEL_INPUT_H;
        }
        
        if (tflite.getOutputTensorCount() > 0) {
            int[] outputShape = tflite.getOutputTensor(0).shape();
            actualOutputN = outputShape[1];
            actualOutputFeatures = outputShape[2];
            android.util.Log.d("TFLiteRunner", "Output: [" + outputShape[0] + ", " + outputShape[1] + ", " + outputShape[2] + "]");
        } else {
            actualOutputN = OUTPUT_SIZE_N;
            actualOutputFeatures = 6;
        }
        
        tflite.allocateTensors();
        android.util.Log.d("TFLiteRunner", "Model ready: " + actualInputWidth + "x" + actualInputHeight);
    }

    private MappedByteBuffer loadModelFile(Context context, String assetName) throws IOException {
        AssetFileDescriptor fd = context.getAssets().openFd(assetName);
        FileInputStream inputStream = new FileInputStream(fd.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fd.getStartOffset();
        long declaredLength = fd.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private float[][][][] bitmapToInput(Bitmap src) {
        Bitmap resized = Bitmap.createScaledBitmap(src, actualInputWidth, actualInputHeight, true);
        float[][][][] input = new float[1][actualInputHeight][actualInputWidth][3];
        int[] px = new int[actualInputWidth * actualInputHeight];
        resized.getPixels(px, 0, actualInputWidth, 0, 0, actualInputWidth, actualInputHeight);
        
        int i = 0;
        for (int y = 0; y < actualInputHeight; y++) {
            for (int x = 0; x < actualInputWidth; x++) {
                int c = px[i++];
                input[0][y][x][0] = ((c >> 16) & 0xFF) / 255.0f;
                input[0][y][x][1] = ((c >> 8) & 0xFF) / 255.0f;
                input[0][y][x][2] = (c & 0xFF) / 255.0f;
            }
        }
        return input;
    }

    private float sigmoid(float x) {
        return (float)(1.0 / (1.0 + Math.exp(-x)));
    }

    public List<Det> detect(Bitmap frame, float scoreThresh) {
        try {
            float[][][][] input = bitmapToInput(frame);
            float[][][] out = new float[1][actualOutputN][actualOutputFeatures];
            tflite.run(input, out);
            
            android.util.Log.d("TFLiteRunner", "Processing output shape [" + actualOutputN + ", " + actualOutputFeatures + "] with threshold " + scoreThresh);

            ArrayList<Det> dets = new ArrayList<>();
            
            // Based on Python reference: YOLOv11 output format [1, 9, 21504]
            // where 9 = 4 (bbox) + num_classes, 21504 = total anchors
            // The output is in format [batch, channels, anchors]
            
            int numClasses = actualOutputN - 4; // 9 - 4 = 5 classes
            int totalAnchors = actualOutputFeatures; // 21504
            
            android.util.Log.d("TFLiteRunner", "Detected " + numClasses + " classes, " + totalAnchors + " anchors");
            
            // Generate anchor points for 3 scales (following Python reference)
            int[] strides = {8, 16, 32};
            int anchorIdx = 0;
            
            for (int stride : strides) {
                int gridH = actualInputHeight / stride;
                int gridW = actualInputWidth / stride;
                
                for (int y = 0; y < gridH; y++) {
                    for (int x = 0; x < gridW; x++) {
                        if (anchorIdx >= totalAnchors) break;
                        
                        // Extract predictions for this anchor
                        float bboxX = out[0][0][anchorIdx]; // x offset
                        float bboxY = out[0][1][anchorIdx]; // y offset
                        float bboxW = out[0][2][anchorIdx]; // width  
                        float bboxH = out[0][3][anchorIdx]; // height
                        
                        // Find best class
                        float maxClassProb = 0;
                        int bestClass = 0;
                        
                        for (int cls = 0; cls < numClasses; cls++) {
                            float classLogit = out[0][4 + cls][anchorIdx];
                            float classProb = sigmoid(classLogit); // Apply sigmoid to convert logits to probabilities
                            
                            if (classProb > maxClassProb) {
                                maxClassProb = classProb;
                                bestClass = cls;
                            }
                        }
                        
                        // Apply confidence threshold
                        if (maxClassProb >= scoreThresh) {
                            // Convert to absolute coordinates (following Python reference logic)
                            // The bbox predictions are normalized [0,1] relative to the model input
                            float centerX = bboxX * actualInputWidth;
                            float centerY = bboxY * actualInputHeight;
                            float width = bboxW * actualInputWidth;
                            float height = bboxH * actualInputHeight;
                            
                            // Convert center format to corners
                            float x1 = centerX - width / 2;
                            float y1 = centerY - height / 2;
                            float x2 = centerX + width / 2;
                            float y2 = centerY + height / 2;
                            
                            // Scale to original image size
                            float scaleX = (float) frame.getWidth() / actualInputWidth;
                            float scaleY = (float) frame.getHeight() / actualInputHeight;
                            
                            float scaledX1 = x1 * scaleX;
                            float scaledY1 = y1 * scaleY;
                            float scaledX2 = x2 * scaleX;
                            float scaledY2 = y2 * scaleY;
                            
                            // Clamp to image bounds
                            scaledX1 = Math.max(0, Math.min(frame.getWidth() - 1, scaledX1));
                            scaledY1 = Math.max(0, Math.min(frame.getHeight() - 1, scaledY1));
                            scaledX2 = Math.max(0, Math.min(frame.getWidth(), scaledX2));
                            scaledY2 = Math.max(0, Math.min(frame.getHeight(), scaledY2));
                            
                            // Filter out invalid boxes
                            if (scaledX2 > scaledX1 && scaledY2 > scaledY1 && 
                                (scaledX2 - scaledX1) > 5 && (scaledY2 - scaledY1) > 5) {
                                
                                Det detection = new Det(scaledX1, scaledY1, scaledX2, scaledY2, maxClassProb, bestClass);
                                dets.add(detection);
                                
                                // Log first few high-confidence detections
                                if (dets.size() <= 10 && maxClassProb > 0.6) {
                                    android.util.Log.d("TFLiteRunner", "HIGH CONF: " + detection.toString() + 
                                        " [raw: " + bboxX + "," + bboxY + "," + bboxW + "," + bboxH + "]");
                                }
                            }
                        }
                        
                        anchorIdx++;
                    }
                    if (anchorIdx >= totalAnchors) break;
                }
                if (anchorIdx >= totalAnchors) break;
            }
            
            android.util.Log.d("TFLiteRunner", "Found " + dets.size() + " detections above threshold " + scoreThresh);
            return dets;
            
        } catch (Exception e) {
            android.util.Log.e("TFLiteRunner", "Detection failed", e);
            return new ArrayList<>();
        }
    }

    public List<Det> detectWithNMS(Bitmap frame, float scoreThresh, float nmsThresh) {
        List<Det> allDetections = detect(frame, scoreThresh);
        return applyNMS(allDetections, nmsThresh);
    }

    private List<Det> applyNMS(List<Det> detections, float nmsThreshold) {
        if (detections.isEmpty()) return detections;
        
        // Sort by confidence score (highest first)
        detections.sort((a, b) -> Float.compare(b.score, a.score));
        
        List<Det> selectedDetections = new ArrayList<>();
        boolean[] suppressed = new boolean[detections.size()];
        
        for (int i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;
            
            Det currentDet = detections.get(i);
            selectedDetections.add(currentDet);
            
            // Suppress overlapping detections
            for (int j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;
                
                Det otherDet = detections.get(j);
                float iou = calculateIoU(currentDet, otherDet);
                
                if (iou > nmsThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        
        android.util.Log.d("TFLiteRunner", "NMS: " + detections.size() + " -> " + selectedDetections.size() + " detections");
        return selectedDetections;
    }

    private float calculateIoU(Det a, Det b) {
        float intersectionLeft = Math.max(a.x1, b.x1);
        float intersectionTop = Math.max(a.y1, b.y1);
        float intersectionRight = Math.min(a.x2, b.x2);
        float intersectionBottom = Math.min(a.y2, b.y2);
        
        if (intersectionLeft >= intersectionRight || intersectionTop >= intersectionBottom) {
            return 0.0f;
        }
        
        float intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop);
        float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        float unionArea = areaA + areaB - intersectionArea;
        
        return intersectionArea / unionArea;
    }

    public int getActualInputWidth() { return actualInputWidth; }
    public int getActualInputHeight() { return actualInputHeight; }
    public static int getModelInputW() { return MODEL_INPUT_W; }
    public static int getModelInputH() { return MODEL_INPUT_H; }
    
    public void close() {
        if (tflite != null) tflite.close();
    }
}
