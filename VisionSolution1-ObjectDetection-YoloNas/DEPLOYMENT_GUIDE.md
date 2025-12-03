# QIDK Standalone Deployment Guide

## Overview
Deploy the Object Detection + Depth Estimation app to QIDK as a standalone application.

---

## OPTION 1: Quick Deploy (Debug APK) - Recommended for Testing

### Step 1: Install Debug APK
```bash
# Connect QIDK via USB
adb devices

# Install the debug APK
./gradlew installDebug

# Grant permissions
adb shell pm grant com.qc.objectdetectionYoloNas android.permission.CAMERA
adb shell pm grant com.qc.objectdetectionYoloNas android.permission.WRITE_EXTERNAL_STORAGE
adb shell pm grant com.qc.objectdetectionYoloNas android.permission.READ_EXTERNAL_STORAGE
```

### Step 2: Disconnect and Run
- Disconnect USB cable
- Launch app from QIDK home screen or app drawer
- App icon: "Object Detection YoloNas"

---

## OPTION 2: Release APK (Production Ready)

### Step 1: Generate Signing Key (One-time setup)
```bash
# Create keystore directory
mkdir -p ~/.android

# Generate release keystore
keytool -genkey -v -keystore ~/.android/release.keystore \
  -alias objectdetection_release \
  -keyalg RSA -keysize 2048 -validity 10000

# Enter password and details when prompted
# REMEMBER: Store password securely!
```

### Step 2: Configure Signing in build.gradle
Add this to `app/build.gradle` inside `android {}` block:

```gradle
signingConfigs {
    release {
        storeFile file(System.getProperty("user.home") + "/.android/release.keystore")
        storePassword "YOUR_KEYSTORE_PASSWORD"  // Replace with actual password
        keyAlias "objectdetection_release"
        keyPassword "YOUR_KEY_PASSWORD"  // Replace with actual password
    }
}

buildTypes {
    release {
        minifyEnabled false
        proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
        signingConfig signingConfigs.release  // Add this line
    }
}
```

**SECURITY TIP**: Never commit passwords to Git! Use environment variables:
```gradle
storePassword System.getenv("KEYSTORE_PASSWORD")
keyPassword System.getenv("KEY_PASSWORD")
```

### Step 3: Build Release APK
```bash
# Set environment variables (if using secure method)
export KEYSTORE_PASSWORD="your_password"
export KEY_PASSWORD="your_password"

# Build signed release APK
./gradlew assembleRelease

# APK location:
# app/build/outputs/apk/release/app-release.apk
```

### Step 4: Install on QIDK
```bash
# Install release APK
adb install app/build/outputs/apk/release/app-release.apk

# OR update existing installation
adb install -r app/build/outputs/apk/release/app-release.apk

# Grant permissions
adb shell pm grant com.qc.objectdetectionYoloNas android.permission.CAMERA
adb shell pm grant com.qc.objectdetectionYoloNas android.permission.WRITE_EXTERNAL_STORAGE
adb shell pm grant com.qc.objectdetectionYoloNas android.permission.READ_EXTERNAL_STORAGE
```

---

## OPTION 3: Wireless Distribution (For Field Deployment)

### Method A: Transfer APK via File Manager
```bash
# Copy APK to QIDK internal storage
adb push app/build/outputs/apk/release/app-release.apk /sdcard/Download/

# On QIDK device:
# 1. Open File Manager
# 2. Navigate to Download folder
# 3. Tap app-release.apk
# 4. Tap "Install"
# 5. Allow "Install from unknown sources" if prompted
```

### Method B: Share via Cloud/USB Drive
1. Copy APK to USB drive or upload to cloud (Google Drive, Dropbox)
2. On QIDK: Download/copy APK to device
3. Install via File Manager as above

### Method C: QR Code Distribution (Advanced)
1. Upload APK to web server
2. Generate QR code linking to APK URL
3. Scan QR code on QIDK to download and install

---

## Verification Steps

### 1. Check Installation
```bash
# Verify app is installed
adb shell pm list packages | grep objectdetectionYoloNas

# Expected output:
# package:com.qc.objectdetectionYoloNas
```

### 2. Test Standalone Launch
```bash
# Launch app via ADB (simulates user tap)
adb shell am start -n com.qc.objectdetectionYoloNas/.MainActivity

# Check logs
adb logcat -s SNPE_INF:I DepthSnpeBridge:I MainActivity:I
```

### 3. Verify NPU Execution
Look for these logs on launch:
```
I SNPE_INF: ✓ NPU/HTP CONFIGURED
I DepthSnpeBridge: Runtime: NPU/HTP ✓
```

### 4. Test Offline Operation
1. Disconnect USB cable completely
2. Turn off QIDK
3. Power on QIDK (standalone)
4. Launch app from home screen
5. Point camera at objects (worker, truck, car, etc.)
6. Verify detection boxes and depth colors appear

---

## Troubleshooting

### App Crashes on Launch
```bash
# Check crash logs
adb logcat *:E | grep objectdetectionYoloNas

# Common fixes:
# - Reinstall with: adb install -r <apk>
# - Clear app data: adb shell pm clear com.qc.objectdetectionYoloNas
# - Verify SNPE libs in APK: unzip -l app-release.apk | grep libSNPE
```

### Permissions Not Granted
```bash
# Reset permissions and grant again
adb shell pm reset-permissions com.qc.objectdetectionYoloNas
adb shell pm grant com.qc.objectdetectionYoloNas android.permission.CAMERA
```

### Firebase Connection Issues
- App works offline but won't sync alerts
- Check QIDK WiFi connection
- Verify `google-services.json` is in `app/` folder
- Check Firebase console for API key restrictions

### Models Not Loading
```bash
# Verify models are packaged in APK
unzip -l app/build/outputs/apk/release/app-release.apk | grep -E "\.dlc|\.tflite|\.onnx"

# Should see:
# assets/depth_anything_v2_vitb_quantized.dlc
# assets/yolo11n.tflite (or similar)
```

---

## Performance Optimization

### Reduce APK Size
1. Remove unused architectures (keep only arm64-v8a):
   ```gradle
   ndk { abiFilters 'arm64-v8a' }  // Already configured
   ```

2. Enable ProGuard (release builds):
   ```gradle
   buildTypes {
       release {
           minifyEnabled true  // Change from false
           shrinkResources true
       }
   }
   ```

### Improve Startup Time
- Models load on-demand (already implemented)
- Firebase initializes asynchronously (already implemented)
- Consider preloading models in splash screen

---

## Distribution Checklist

Before deploying to production QIDK devices:

- [ ] Test all detection classes (worker, truck, car, bike, bulldozer)
- [ ] Verify depth estimation accuracy with known distances
- [ ] Test in various lighting conditions (bright, dim, outdoor, indoor)
- [ ] Confirm Firebase alerts are received (if online)
- [ ] Measure battery consumption during continuous operation
- [ ] Test app recovery after device reboot
- [ ] Verify NPU execution (not CPU fallback)
- [ ] Check inference latency (<100ms per frame target)
- [ ] Test with disconnected USB (full standalone mode)
- [ ] Document expected behavior and known limitations

---

## Quick Reference

| Action | Command |
|--------|---------|
| Build Debug | `./gradlew assembleDebug` |
| Build Release | `./gradlew assembleRelease` |
| Install Debug | `./gradlew installDebug` |
| Install Release | `adb install app/build/outputs/apk/release/app-release.apk` |
| Uninstall | `adb uninstall com.qc.objectdetectionYoloNas` |
| Launch App | `adb shell am start -n com.qc.objectdetectionYoloNas/.MainActivity` |
| Clear Data | `adb shell pm clear com.qc.objectdetectionYoloNas` |
| View Logs | `adb logcat -s SNPE_INF DepthSnpeBridge MainActivity` |

---

## Support

For issues:
1. Check logcat output: `adb logcat *:E`
2. Verify SNPE version: Look for "SNPE Version = 2.40.0..." in logs
3. Test on emulator first (if available)
4. Check QIDK system requirements (Android 10+, Snapdragon 8 Gen 2)

