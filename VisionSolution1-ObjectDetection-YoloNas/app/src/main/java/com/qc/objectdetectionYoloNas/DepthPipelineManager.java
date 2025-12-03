package com.qc.objectdetectionYoloNas;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.json.JSONException;

import java.io.IOException;
import java.util.Locale;

/**
 * High-level helper that wires together the SNPE runtime, calibration data and depth
 * post-processing utilities required by the integrated detection+depth pipeline.
 */
public class DepthPipelineManager {
    private static final String TAG = "DepthPipelineMgr";
    private static final String CALIBRATION_ASSET = "camera_intrinsics.json";
    private static final char DEFAULT_RUNTIME = 'D'; // DSP/HTP for QIDK NPU

    private final Context context;
    private final DepthSnpeBridge snpeBridge;
    private CameraCalibration calibration;
    private float depthScaleMeters = 2.0f; // default scaling factor (meters at depth=1.0)

    public DepthPipelineManager(Context context) {
        this.context = context.getApplicationContext();
        this.snpeBridge = new DepthSnpeBridge();
        if (!snpeBridge.isNativeLibraryLoaded()) {
            Log.w(TAG, "Native depth library missing; depth inference will be disabled until the .so is packaged.");
        }
        loadCalibration();
    }

    private void loadCalibration() {
        try {
            calibration = CameraCalibration.loadFromAssets(context, CALIBRATION_ASSET);
            Log.i(TAG, "Loaded calibration from " + CALIBRATION_ASSET);
        } catch (IOException | JSONException e) {
            Log.e(TAG, "Failed to load camera calibration", e);
            calibration = null;
        }
    }

    public void setDepthScaleMeters(float depthScaleMeters) {
        this.depthScaleMeters = depthScaleMeters;
    }

    public float getDepthScaleMeters() {
        return depthScaleMeters;
    }

    public CameraCalibration getCalibration() {
        return calibration;
    }

    public boolean isDepthAvailable() {
        return snpeBridge != null && snpeBridge.isNativeLibraryLoaded();
    }

    public boolean ensureRuntimeReady() {
        return snpeBridge.ensureInitialized(context, DEFAULT_RUNTIME);
    }

    public DepthResult estimateDepth(Bitmap sourceBitmap) {
        if (sourceBitmap == null) {
            return null;
        }
        if (!snpeBridge.isNativeLibraryLoaded()) {
            Log.w(TAG, "Depth estimate requested but native bridge is unavailable");
            return null;
        }
        if (!ensureRuntimeReady()) {
            Log.e(TAG, "Depth runtime not ready");
            return null;
        }

        Bitmap bitmap = sourceBitmap;
        if (sourceBitmap.getConfig() != Bitmap.Config.ARGB_8888) {
            bitmap = sourceBitmap.copy(Bitmap.Config.ARGB_8888, false);
        }

        final int width = bitmap.getWidth();
        final int height = bitmap.getHeight();
        int[] rgbaPixels = new int[width * height];
        bitmap.getPixels(rgbaPixels, 0, width, 0, 0, width, height);

        float[] depthBuffer = new float[rgbaPixels.length];
        float[] statsBuffer = new float[2];

        long start = System.currentTimeMillis();
        boolean success = snpeBridge.inferDepth(rgbaPixels, width, height, depthBuffer, statsBuffer);
        long duration = System.currentTimeMillis() - start;

        if (!success) {
            Log.e(TAG, "Depth inference failed");
            return null;
        }

        Log.i(TAG, String.format(Locale.US, "Depth inference done in %d ms (min=%.4f, max=%.4f)",
                duration, statsBuffer[0], statsBuffer[1]));
        return new DepthResult(depthBuffer, width, height, statsBuffer[0], statsBuffer[1], duration);
    }

    public float sampleDepthMeters(DepthResult depthResult, float x, float y) {
        if (depthResult == null) {
            return Float.NaN;
        }
        float normalized = depthResult.sampleNormalizedDepth(x, y);
        if (normalized < 0f) {
            return Float.NaN;
        }
        return normalized * depthScaleMeters;
    }

    public CameraCalibration.Point3D projectTo3D(DepthResult depthResult, float x, float y) {
        if (calibration == null) {
            return null;
        }
        float depthMeters = sampleDepthMeters(depthResult, x, y);
        if (Float.isNaN(depthMeters)) {
            return null;
        }
        return calibration.pixelToPoint(x, y, depthMeters);
    }

    public static class DepthResult {
        public final float[] normalizedDepth;
        public final int width;
        public final int height;
        public final float minValue;
        public final float maxValue;
        public final long inferenceTimeMs;

        DepthResult(float[] normalizedDepth, int width, int height,
                    float minValue, float maxValue, long inferenceTimeMs) {
            this.normalizedDepth = normalizedDepth;
            this.width = width;
            this.height = height;
            this.minValue = minValue;
            this.maxValue = maxValue;
            this.inferenceTimeMs = inferenceTimeMs;
        }

        public float sampleNormalizedDepth(float x, float y) {
            if (normalizedDepth == null || width == 0 || height == 0) {
                return -1f;
            }
            int ix = Math.min(Math.max(Math.round(x), 0), width - 1);
            int iy = Math.min(Math.max(Math.round(y), 0), height - 1);
            int idx = iy * width + ix;
            if (idx < 0 || idx >= normalizedDepth.length) {
                return -1f;
            }
            return normalizedDepth[idx];
        }
    }
}
