package com.qc.objectdetectionYoloNas;

import android.content.Context;
import android.content.res.AssetManager;
import android.text.TextUtils;
import android.util.Log;

/**
 * Thin JNI bridge around the SNPE-powered Depth Anything V2 DLC.
 * Handles native initialization and inference calls while ensuring threads
 * do not attempt inference before the runtime is ready.
 */
public class DepthSnpeBridge {
    private static final String TAG = "DepthSnpeBridge";

    private static boolean nativeLibraryLoaded;
    private static UnsatisfiedLinkError nativeLoadError;

    static {
        try {
            System.loadLibrary("objectdetectionYoloNas");
            nativeLibraryLoaded = true;
            Log.i(TAG, "Native depth bridge loaded");
        } catch (UnsatisfiedLinkError e) {
            nativeLoadError = e;
            nativeLibraryLoaded = false;
            Log.e(TAG, "Unable to load native depth library; depth inference disabled", e);
        }
    }

    private boolean initialized;
    private String lastInitLog = "";

    public synchronized boolean ensureInitialized(Context context, char runtime) {
        if (!nativeLibraryLoaded) {
            lastInitLog = nativeLoadError != null ? nativeLoadError.getMessage() : "Native depth library unavailable";
            Log.w(TAG, "Depth runtime requested but native library failed to load: " + lastInitLog);
            return false;
        }
        if (initialized) {
            return true;
        }
        lastInitLog = nativeInit(context.getAssets(), context.getApplicationInfo().nativeLibraryDir, runtime);
        initialized = !TextUtils.isEmpty(lastInitLog);
        if (!initialized) {
            Log.e(TAG, "Failed to initialize SNPE runtime: " + lastInitLog);
        } else {
            // Parse runtime status from initialization log
            String runtimeStatus = "UNKNOWN";
            if (lastInitLog.contains("NPU/HTP")) {
                runtimeStatus = "NPU/HTP ✓";
            } else if (lastInitLog.contains("GPU")) {
                runtimeStatus = "GPU";
            } else if (lastInitLog.contains("CPU")) {
                runtimeStatus = "CPU";
            }
            
            Log.i(TAG, "========================================");
            Log.i(TAG, "DEPTH MODEL RUNTIME STATUS");
            Log.i(TAG, "========================================");
            Log.i(TAG, "Runtime: " + runtimeStatus);
            Log.i(TAG, "Init log: " + lastInitLog);
            Log.i(TAG, "========================================");
        }
        return initialized;
    }

    public synchronized boolean inferDepth(int[] rgbaPixels, int width, int height,
                                           float[] depthBuffer, float[] statsBuffer) {
        if (!initialized) {
            Log.w(TAG, "inferDepth called before initialization");
            return false;
        }
        return nativeInferDepth(rgbaPixels, width, height, depthBuffer, statsBuffer);
    }

    public boolean isNativeLibraryLoaded() {
        return nativeLibraryLoaded;
    }

    public String getNativeLoadErrorMessage() {
        return nativeLoadError != null ? nativeLoadError.getMessage() : null;
    }

    public boolean isInitialized() {
        return initialized;
    }

    public String getLastInitLog() {
        return lastInitLog;
    }

    private native String nativeInit(AssetManager assetManager, String nativeLibDir, char runtime);
    private native boolean nativeInferDepth(int[] rgbaPixels, int width, int height,
                                            float[] depthBuffer, float[] statsBuffer);
}
