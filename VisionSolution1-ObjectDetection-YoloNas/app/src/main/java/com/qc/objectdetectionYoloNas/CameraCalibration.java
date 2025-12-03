package com.qc.objectdetectionYoloNas;

import android.content.Context;
import android.content.res.AssetManager;

import androidx.annotation.NonNull;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * Lightweight camera calibration helper that mirrors the Python PixelTo3DConverter implementation.
 * It loads calibration parameters from {@code camera_intrinsics.json} bundled inside assets and
 * exposes helpers to convert 2D pixel coordinates and depth estimates into real-world coordinates.
 */
public class CameraCalibration {

    public static class Point3D {
        public final float x;
        public final float y;
        public final float z;

        public Point3D(float x, float y, float z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        @NonNull
        @Override
        public String toString() {
            return String.format(Locale.US, "(%.3f, %.3f, %.3f)m", x, y, z);
        }
    }

    private final float fx;
    private final float fy;
    private final float cx;
    private final float cy;
    private final float[] distortionCoeffs;
    private final int imageWidth;
    private final int imageHeight;

    private CameraCalibration(float fx, float fy, float cx, float cy,
                              float[] distortionCoeffs,
                              int imageWidth, int imageHeight) {
        this.fx = fx;
        this.fy = fy;
        this.cx = cx;
        this.cy = cy;
        this.distortionCoeffs = distortionCoeffs;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
    }

    public static CameraCalibration loadFromAssets(Context context, String assetName)
            throws IOException, JSONException {
        AssetManager assets = context.getAssets();
        try (InputStream inputStream = assets.open(assetName);
             InputStreamReader reader = new InputStreamReader(inputStream, StandardCharsets.UTF_8);
             BufferedReader bufferedReader = new BufferedReader(reader)) {

            StringBuilder builder = new StringBuilder();
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                builder.append(line);
            }
            JSONObject root = new JSONObject(builder.toString());

            JSONObject focal = root.getJSONObject("focal_length");
            JSONObject principal = root.getJSONObject("principal_point");
            JSONObject imageSize = root.getJSONObject("image_size");

            float fx = (float) focal.getDouble("fx");
            float fy = (float) focal.getDouble("fy");
            float cx = (float) principal.getDouble("cx");
            float cy = (float) principal.getDouble("cy");
            int width = imageSize.getInt("width");
            int height = imageSize.getInt("height");

            JSONArray dist = root.getJSONArray("distortion_coefficients");
            float[] distortionCoeffs = new float[dist.length()];
            for (int i = 0; i < dist.length(); i++) {
                distortionCoeffs[i] = (float) dist.getDouble(i);
            }

            return new CameraCalibration(fx, fy, cx, cy, distortionCoeffs, width, height);
        }
    }

    public Point3D pixelToPoint(float u, float v, float depthMeters) {
        float x = (u - cx) * depthMeters / fx;
        float y = (v - cy) * depthMeters / fy;
        return new Point3D(x, y, depthMeters);
    }

    public float distance(Point3D a, Point3D b) {
        float dx = b.x - a.x;
        float dy = b.y - a.y;
        float dz = b.z - a.z;
        return (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    public float horizontalDistance(Point3D a, Point3D b) {
        float dx = b.x - a.x;
        float dz = b.z - a.z;
        return (float) Math.sqrt(dx * dx + dz * dz);
    }

    public float getFx() {
        return fx;
    }

    public float getFy() {
        return fy;
    }

    public float getCx() {
        return cx;
    }

    public float getCy() {
        return cy;
    }

    public int getImageWidth() {
        return imageWidth;
    }

    public int getImageHeight() {
        return imageHeight;
    }

    public float[] getDistortionCoeffs() {
        return distortionCoeffs;
    }
}
