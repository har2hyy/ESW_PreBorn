package com.qc.objectdetectionYoloNas;

import java.util.ArrayList;

/**
 * Updated RectangleBox class to include center coordinates and safety status.
 */
public class RectangleBox {

    // --- Your original fields ---
    public float top;
    public float bottom;
    public float left;
    public float right;
    public int classId;
    public float confidence;
    public int fps;
    public String processing_time;
    public String label;

    // --- NEW fields for alert logic and Firebase ---
    public float centerX;
    public float centerY;
    public boolean isUnsafe = false; // Default to safe

    // Your existing default constructor is perfect for Firebase
    public RectangleBox() {}

    // Your existing parameterized constructor
    public RectangleBox(float x1, float y1, float x2, float y2, int classId, float confidence) {
        this.left = x1;
        this.top = y1;
        this.right = x2;
        this.bottom = y2;
        this.classId = classId;
        this.confidence = confidence;
        
        // --- NEW: Calculate center point on creation ---
        this.calculateCenter();
    }

    // --- NEW: A helper method to calculate the center ---
    public void calculateCenter() {
        this.centerX = (left + right) / 2;
        this.centerY = (top + bottom) / 2;
    }

    public static ArrayList<RectangleBox> createBoxes(int num) {
        final ArrayList<RectangleBox> boxes = new ArrayList<>();
        for (int i = 0; i < num; ++i) {
            boxes.add(new RectangleBox());
        }
        return boxes;
    }
}