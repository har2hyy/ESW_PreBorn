package com.qc.objectdetectionYoloNas;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

/**
 * Fullscreen image viewer activity
 */
public class FullscreenImageActivity extends AppCompatActivity {

    private ZoomableImageView fullscreenImageView;
    private Button closeButton;
    private TextView infoText;
    
    public static Bitmap imageToDisplay = null; // Static reference to pass image data

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        // The theme already handles fullscreen mode and action bar hiding
        setContentView(R.layout.activity_fullscreen_image);
        
        initializeViews();
        setupClickListeners();
        displayImage();
    }

    private void initializeViews() {
        fullscreenImageView = findViewById(R.id.fullscreenImageView);
        closeButton = findViewById(R.id.closeButton);
        infoText = findViewById(R.id.fullscreenInfoText);
    }

    private void setupClickListeners() {
        // Close button
        closeButton.setOnClickListener(v -> finish());
        
        // Tap on image to close (handled by ZoomableImageView)
        fullscreenImageView.setOnClickListener(v -> finish());
    }

    private void displayImage() {
        if (imageToDisplay != null) {
            fullscreenImageView.setImageBitmap(imageToDisplay);
            infoText.setText("Pinch to zoom • Drag to pan • Double-tap to reset • Tap to close • " + 
                imageToDisplay.getWidth() + "×" + imageToDisplay.getHeight() + " pixels");
        } else {
            infoText.setText("No image to display");
            finish(); // Close if no image
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Clear static reference to prevent memory leaks
        imageToDisplay = null;
    }

    @Override
    public void onBackPressed() {
        super.onBackPressed();
        finish();
    }
}