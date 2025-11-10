package com.qc.objectdetectionYoloNas;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int GALLERY_REQUEST_CODE = 1001;
    private static final int PERMISSION_REQUEST_CODE = 1002;

    private ImageView imageView;
    private TextView resultsTextView;
    private Button selectImageButton;
    private Button runInferenceButton;
    private SeekBar confidenceSlider;
    private TextView confidenceValueText;

    private TFLiteRunner tfliteRunner;
    private Bitmap selectedBitmap;
    private float currentConfidenceThreshold = 0.45f;

    private DatabaseReference detectionsRef;
    private DatabaseReference thresholdRef;
    private float distanceThreshold = 150f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);

        initializeViews();
        initializeTFLite();
        initializeFirebase();
        checkPermissions();
    }

    private void initializeFirebase() {
        FirebaseDatabase database = FirebaseDatabase.getInstance();
        detectionsRef = database.getReference("detections");
        thresholdRef = database.getReference("config/threshold");

        thresholdRef.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot dataSnapshot) {
                Float newThreshold = dataSnapshot.getValue(Float.class);
                if (newThreshold != null) {
                    distanceThreshold = newThreshold;
                    Log.i(TAG, "Updated distance threshold from Firebase: " + distanceThreshold);
                    Toast.makeText(MainActivity.this, "Safety distance updated to " + (int)distanceThreshold + "px", Toast.LENGTH_SHORT).show();
                }
            }
            @Override
            public void onCancelled(@NonNull DatabaseError databaseError) {
                Log.w(TAG, "Failed to read threshold from Firebase.", databaseError.toException());
            }
        });
    }

    private void runInference() {
        if (selectedBitmap == null || tfliteRunner == null) {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show();
            return;
        }

        runInferenceButton.setEnabled(false);
        resultsTextView.setText("Running detection...");

        new Thread(() -> {
            try {
                long startTime = System.currentTimeMillis();
                List<TFLiteRunner.Det> detections = tfliteRunner.detectWithNMS(selectedBitmap, currentConfidenceThreshold, 0.4f);
                long inferenceTime = System.currentTimeMillis() - startTime;
                
                Bitmap resultBitmap = processAndDrawAlerts(selectedBitmap.copy(Bitmap.Config.ARGB_8888, true), detections);

                runOnUiThread(() -> {
                    imageView.setImageBitmap(resultBitmap);
                    displayResults(detections, inferenceTime);
                    runInferenceButton.setEnabled(true);
                });

            } catch (Exception e) {
                Log.e(TAG, "Error during inference", e);
                runOnUiThread(() -> {
                    resultsTextView.setText("Error during detection: " + e.getMessage());
                    runInferenceButton.setEnabled(true);
                });
            }
        }).start();
    }
    
    // --- THIS METHOD IS NOW COMPLETE ---
    private Bitmap processAndDrawAlerts(Bitmap bitmap, List<TFLiteRunner.Det> detections) {
        Canvas canvas = new Canvas(bitmap);
        Paint alertPaint = new Paint();
        alertPaint.setColor(Color.RED);
        alertPaint.setStyle(Paint.Style.FILL);
        alertPaint.setTextSize(40f);
        alertPaint.setTypeface(Typeface.create(Typeface.DEFAULT, Typeface.BOLD));
        
        Paint safePaint = new Paint();
        safePaint.setColor(Color.GREEN);
        safePaint.setStyle(Paint.Style.FILL);

        List<RectangleBox> workers = new ArrayList<>();
        List<RectangleBox> vehicles = new ArrayList<>();
        List<RectangleBox> allObjectsForUpload = new ArrayList<>();

        // 1. Extract center points and sort all detected objects using your RectangleBox class
        for (TFLiteRunner.Det det : detections) {
            RectangleBox box = new RectangleBox(det.x1, det.y1, det.x2, det.y2, det.cls, det.score);
            box.label = (det.cls == 0) ? "worker" : (det.cls == 1) ? "truck" : "vehicle"; // Assuming class IDs

            allObjectsForUpload.add(box);

            if (box.label.equals("worker")) {
                workers.add(box);
            } else {
                vehicles.add(box);
            }
        }

        // 2. Check distances and mark unsafe workers
        for (RectangleBox worker : workers) {
            for (RectangleBox vehicle : vehicles) {
                float dx = worker.centerX - vehicle.centerX;
                float dy = worker.centerY - vehicle.centerY;
                double distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < distanceThreshold) {
                    worker.isUnsafe = true;
                    break;
                }
            }
        }
        
        // 3. Draw dots on the canvas based on safety status
        for (RectangleBox box : allObjectsForUpload) {
             if (box.label.equals("worker")) {
                 if (box.isUnsafe) {
                    canvas.drawCircle(box.centerX, box.centerY, 25f, alertPaint);
                    canvas.drawText("ALERT!", box.centerX + 30, box.centerY, alertPaint);
                 } else {
                    canvas.drawCircle(box.centerX, box.centerY, 15f, safePaint);
                 }
             } else {
                 // Optionally draw blue dots for vehicles
                 Paint vehiclePaint = new Paint();
                 vehiclePaint.setColor(Color.BLUE);
                 canvas.drawCircle(box.centerX, box.centerY, 15f, vehiclePaint);
             }
        }

        // 4. Upload all object data to Firebase
        if (detectionsRef != null) {
            detectionsRef.setValue(allObjectsForUpload);
        }

        return bitmap;
    }

    // --- (The rest of your methods remain unchanged) ---
    // Note: I left all your other methods exactly as they were.
    // ... initializeViews(), onActivityResult(), displayResults(), etc. ...
    private void initializeViews() {
        imageView = findViewById(R.id.imageView);
        resultsTextView = findViewById(R.id.resultsTextView);
        selectImageButton = findViewById(R.id.selectImageButton);
        runInferenceButton = findViewById(R.id.runInferenceButton);
        confidenceSlider = findViewById(R.id.confidenceSlider);
        confidenceValueText = findViewById(R.id.confidenceValueText);
        selectImageButton.setOnClickListener(v -> openGallery());
        runInferenceButton.setOnClickListener(v -> runInference());
        runInferenceButton.setEnabled(false);
        imageView.setOnClickListener(v -> openFullscreenImage());
        setupConfidenceSlider();
    }
    private void initializeTFLite() {
        try {
            tfliteRunner = new TFLiteRunner(this);
            Log.d(TAG, "TFLite model loaded successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize TFLite model", e);
            Toast.makeText(this, "Failed to load AI model: " + e.getMessage(), Toast.LENGTH_LONG).show();
            if (runInferenceButton != null) {
                runInferenceButton.setEnabled(false);
            }
        }
    }
    private void checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_MEDIA_IMAGES}, PERMISSION_REQUEST_CODE);
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
            }
        }
    }
    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        startActivityForResult(intent, GALLERY_REQUEST_CODE);
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == GALLERY_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            Uri selectedImageUri = data.getData();
            try {
                selectedBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), selectedImageUri);
                imageView.setImageBitmap(selectedBitmap);
                runInferenceButton.setEnabled(true);
                resultsTextView.setText("Image selected. Tap 'Run Detection' to analyze.");
            } catch (IOException e) {
                Log.e(TAG, "Error loading image", e);
                Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show();
            }
        }
    }
    private void setupConfidenceSlider() {
        // Set initial slider position to match currentConfidenceThreshold (0.45)
        // Formula: position = (confidence - 0.45) / 0.50 * 100
        int initialPosition = (int) ((currentConfidenceThreshold - 0.45f) / 0.50f * 100);
        confidenceSlider.setProgress(initialPosition);
        
        // Initialize the confidence text display
        confidenceValueText.setText(String.format("%.2f", currentConfidenceThreshold));
        
        confidenceSlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                // Map progress (0-100) to confidence range (0.45-0.95)
                // Formula: confidence = 0.45 + (progress/100) * 0.50
                currentConfidenceThreshold = 0.45f + (progress / 100.0f) * 0.50f;
                confidenceValueText.setText(String.format("%.2f", currentConfidenceThreshold));
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                if (selectedBitmap != null && tfliteRunner != null) {
                    Toast.makeText(MainActivity.this,
                            "Confidence changed to " + String.format("%.2f", currentConfidenceThreshold),
                            Toast.LENGTH_SHORT).show();
                }
            }
        });
    }
    private void displayResults(List<TFLiteRunner.Det> detections, long inferenceTime) {
        String[] classNames = {"worker", "truck", "class2", "class3", "class4"};
        StringBuilder results = new StringBuilder();
        results.append("Detection Results:\n");
        results.append("Confidence Threshold: ").append(String.format("%.2f", currentConfidenceThreshold)).append("\n");
        results.append("Inference Time: ").append(inferenceTime).append("ms\n");
        results.append("Objects Found: ").append(detections.size()).append("\n\n");
        for (int i = 0; i < detections.size(); i++) {
            TFLiteRunner.Det det = detections.get(i);
            String className = (det.cls < classNames.length) ?
                    classNames[det.cls] : ("class" + det.cls);
            results.append(String.format("Object %d:\n", i + 1));
            results.append(String.format("  Class: %s (%d)\n", className, det.cls));
            results.append(String.format("  Confidence: %.2f\n", det.score));
            float centerX = (det.x1 + det.x2) / 2;
            float centerY = (det.y1 + det.y2) / 2;
            results.append(String.format("  Center: [%.1f, %.1f]\n\n", centerX, centerY));
        }
        if (detections.isEmpty()) {
            results.append("No objects detected above confidence threshold.");
        }
        resultsTextView.setText(results.toString());
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Permission required to access images", Toast.LENGTH_LONG).show();
            }
        }
    }
    private void openFullscreenImage() {
        if (imageView.getDrawable() == null) {
            Toast.makeText(this, "No image to display", Toast.LENGTH_SHORT).show();
            return;
        }
        try {
            imageView.setDrawingCacheEnabled(true);
            imageView.buildDrawingCache();
            Bitmap currentImage = Bitmap.createBitmap(imageView.getDrawingCache());
            imageView.setDrawingCacheEnabled(false);
            FullscreenImageActivity.imageToDisplay = currentImage;
            Intent intent = new Intent(this, FullscreenImageActivity.class);
            startActivity(intent);
        } catch (Exception e) {
            Log.e(TAG, "Error opening fullscreen image", e);
            Toast.makeText(this, "Error opening fullscreen view", Toast.LENGTH_SHORT).show();
        }
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tfliteRunner != null) {
            tfliteRunner.close();
        }
    }
}