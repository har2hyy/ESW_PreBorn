# App Relocation Guide
## How to Copy the App to Another Folder/System

This guide explains everything you need to change when copying the Android app to a new location.

---

## Quick Summary

**Good News**: The app is **99% portable**! Almost all paths are **relative**, not absolute.

**Only 1 File Needs Updating**: `local.properties`

---

## Step-by-Step Relocation Process

### 1. Copy the Entire Project Folder

```bash
# Example: Copy to new location
cp -r VisionSolution1-ObjectDetection-YoloNas /path/to/new/location/

# Or compress and transfer
tar -czf app.tar.gz VisionSolution1-ObjectDetection-YoloNas
# Transfer to new system
tar -xzf app.tar.gz
```

### 2. Update `local.properties`

**File**: `local.properties` (in project root)

**What it contains**:
```properties
sdk.dir=/home/harshyy/Android/Sdk
```

**What to change**:
- Replace with YOUR Android SDK path
- Common locations:
  - Linux: `/home/YOUR_USERNAME/Android/Sdk`
  - macOS: `/Users/YOUR_USERNAME/Library/Android/sdk`
  - Windows: `C:\\Users\\YOUR_USERNAME\\AppData\\Local\\Android\\Sdk`

**How to update**:

#### Option A: Manual Edit
```bash
cd /path/to/new/location/VisionSolution1-ObjectDetection-YoloNas
nano local.properties

# Change this line:
sdk.dir=/home/harshyy/Android/Sdk

# To your SDK path:
sdk.dir=/home/YOUR_USERNAME/Android/Sdk
```

#### Option B: Delete and Regenerate
```bash
# Delete old file
rm local.properties

# Open in Android Studio
# It will auto-create local.properties with correct SDK path
```

#### Option C: Auto-detect (Linux/Mac)
```bash
# Find SDK automatically
SDK_PATH=$(find ~/Library/Android ~/Android -name "platform-tools" 2>/dev/null | head -1 | sed 's/\/platform-tools//')

# Update local.properties
echo "sdk.dir=$SDK_PATH" > local.properties
```

### 3. Clean Build Artifacts (Recommended)

```bash
# Remove old build files
./gradlew clean

# Or manually delete
rm -rf .gradle/
rm -rf app/build/
rm -rf build/
```

### 4. Verify and Build

```bash
# Test build
./gradlew assembleDebug

# If successful, you're done! ✅
```

---

## Files That DON'T Need Changing

### ✅ Already Using Relative Paths

These files use **relative paths** based on project structure - no changes needed:

#### `app/build.gradle`
```gradle
// All asset paths are relative to app/src/main/
android {
    sourceSets {
        main {
            jniLibs.srcDirs = ['src/main/jniLibs']
            assets.srcDirs = ['src/main/assets']
        }
    }
}
```

#### `app/src/main/cpp/CMakeLists.txt`
```cmake
# All paths relative to project root
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
set(OpenCV_DIR ${PROJECT_ROOT}/sdk/native/jni)
set(SNPE_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})
```

#### Java Code (MainActivity, DepthPipelineManager, etc.)
```java
// Assets loaded from APK - no file paths
InputStream stream = context.getAssets().open("camera_intrinsics.json");
```

#### Native Code (inference.cpp)
```cpp
// DLC loaded from Android app directory - handled by SNPE runtime
// No hardcoded paths
```

---

## Common Scenarios

### Scenario 1: Copy to Different User on Same System

```bash
# Copy project
cp -r /home/harshyy/Desktop/ESW/ESW_PreBorn/VisionSolution1-ObjectDetection-YoloNas \
     /home/newuser/Projects/

# Update local.properties
cd /home/newuser/Projects/VisionSolution1-ObjectDetection-YoloNas
echo "sdk.dir=/home/newuser/Android/Sdk" > local.properties

# Build
./gradlew assembleDebug
```

### Scenario 2: Copy to Different Computer

```bash
# On old computer
cd /home/harshyy/Desktop/ESW/ESW_PreBorn
tar -czf app.tar.gz VisionSolution1-ObjectDetection-YoloNas
scp app.tar.gz user@newcomputer:/tmp/

# On new computer
cd ~/Projects
tar -xzf /tmp/app.tar.gz
cd VisionSolution1-ObjectDetection-YoloNas

# Update SDK path
echo "sdk.dir=/home/YOUR_USERNAME/Android/Sdk" > local.properties

# Verify Android SDK exists
ls $HOME/Android/Sdk/platform-tools/adb
# If not found, install Android SDK first

# Build
./gradlew assembleDebug
```

### Scenario 3: Move to Windows

```bash
# On Linux (create archive)
tar -czf app.tar.gz VisionSolution1-ObjectDetection-YoloNas

# Transfer to Windows (USB, cloud, etc.)

# On Windows (extract)
# Use 7-Zip or Windows tar
tar -xzf app.tar.gz

# Update local.properties (in project root)
# Use Windows path format:
sdk.dir=C:\\Users\\YourUsername\\AppData\\Local\\Android\\Sdk

# Build (use gradlew.bat instead of ./gradlew)
gradlew.bat assembleDebug
```

### Scenario 4: Share via Git Repository

```bash
# .gitignore already excludes local.properties
# So each developer gets their own

# Clone repo
git clone https://github.com/har2hyy/ESW_PreBorn.git
cd VisionSolution1-ObjectDetection-YoloNas

# Create local.properties (each dev does this)
echo "sdk.dir=$HOME/Android/Sdk" > local.properties

# Build
./gradlew assembleDebug
```

---

## What About Firebase?

### `google-services.json`

**Location**: `app/google-services.json`

**Is it portable?**: ✅ **YES** - No changes needed!

**Why**: This file contains:
- Project ID
- API keys
- Firebase URLs

These are **project-specific**, not **path-specific**.

**When to change**:
- Only if deploying to a **different Firebase project**
- Download new `google-services.json` from Firebase Console
- Replace the old file

**Current file is fine if**:
- Using the same Firebase project
- Same `applicationId` in `build.gradle`

---

## Checklist for Relocation

Before building in new location:

- [ ] **Copied entire project folder**
  - Include all subfolders (app, gradle, sdk, etc.)
  - Don't skip hidden files (.gradle, .idea - optional)

- [ ] **Updated `local.properties`**
  - Set correct Android SDK path for new system
  - Verify SDK path exists: `ls /path/to/Android/Sdk`

- [ ] **Android SDK installed**
  - Platform tools
  - NDK r21e (if rebuilding native code)
  - CMake 3.18.1+ (if rebuilding native code)

- [ ] **Java/JDK installed**
  - JDK 11 or higher
  - Check: `java -version`

- [ ] **Clean build (optional but recommended)**
  - Run: `./gradlew clean`

- [ ] **Test build**
  - Run: `./gradlew assembleDebug`
  - APK should generate successfully

- [ ] **Verify assets included**
  - Unzip APK and check for:
    - `assets/camera_intrinsics.json`
    - `assets/depth_anything_v2_vitb_quantized.dlc`
    - `assets/yolo11n.tflite`
  - Check: `unzip -l app/build/outputs/apk/debug/app-debug.apk | grep assets`

---

## Troubleshooting New Location

### Error: "SDK location not found"

**Cause**: `local.properties` has wrong path or doesn't exist

**Fix**:
```bash
# Check if file exists
ls -la local.properties

# Check SDK path is correct
cat local.properties

# Find your SDK
find ~ -name "platform-tools" 2>/dev/null

# Update with correct path
echo "sdk.dir=/correct/path/to/Android/Sdk" > local.properties
```

### Error: "NDK not found"

**Cause**: NDK not installed or wrong version

**Fix**:
```bash
# Install NDK via Android Studio:
# Tools → SDK Manager → SDK Tools → NDK (21.4.7075529)

# Or use sdkmanager
sdkmanager "ndk;21.4.7075529"
```

### Error: "Could not find OpenCV"

**Cause**: OpenCV SDK missing from project

**Fix**:
- Verify `sdk/` folder exists in project root
- Contains `native/` and `java/` subfolders
- If missing, you didn't copy the full project

### Build Works, but APK Crashes on Device

**Likely Cause**: Models or libraries not packaged

**Fix**:
```bash
# Check assets are in APK
unzip -l app/build/outputs/apk/debug/app-debug.apk | grep -E "\.dlc|\.tflite|\.json"

# Check native libs are in APK
unzip -l app/build/outputs/apk/debug/app-debug.apk | grep libSNPE.so

# If missing, rebuild from scratch
./gradlew clean assembleDebug
```

---

## Files You Can Safely Delete Before Transfer

To reduce archive size:

```bash
# Build artifacts (will regenerate)
rm -rf .gradle/
rm -rf app/build/
rm -rf build/

# IDE files (will regenerate, or use your own)
rm -rf .idea/
rm local.properties

# Gradle wrapper cache (optional, but recommended to keep)
# rm -rf .gradle/wrapper/

# After deletion, create archive:
tar -czf app-minimal.tar.gz VisionSolution1-ObjectDetection-YoloNas
```

**Keep these essential files**:
- All source code (`.java`, `.cpp`, `.h`)
- Gradle build files (`.gradle`, `gradlew`, `gradlew.bat`)
- Assets (`app/src/main/assets/`)
- Native libraries (`app/src/main/jniLibs/`)
- OpenCV SDK (`sdk/`)
- Config files (`AndroidManifest.xml`, `build.gradle`, etc.)

---

## IDE-Specific Notes

### Android Studio

**Opening Project in New Location**:
1. File → Open
2. Navigate to new project location
3. Select project root folder (contains `build.gradle`)
4. Click OK
5. Android Studio will:
   - Auto-create `local.properties` (if missing)
   - Sync Gradle
   - Index project

**No manual changes needed!** ✅

### VS Code

**Opening Project**:
1. File → Open Folder
2. Select project root
3. Install extensions (if prompted):
   - Java Extension Pack
   - Gradle for Java

**Manual step**:
- Create `local.properties` with your SDK path

### Command Line Only

**Building without IDE**:
```bash
# Create local.properties
echo "sdk.dir=$HOME/Android/Sdk" > local.properties

# Build
./gradlew assembleDebug

# Install
./gradlew installDebug
```

---

## Platform-Specific Path Formats

### Linux
```properties
sdk.dir=/home/username/Android/Sdk
```

### macOS
```properties
sdk.dir=/Users/username/Library/Android/sdk
```

### Windows
```properties
# Use double backslashes or forward slashes
sdk.dir=C:\\Users\\Username\\AppData\\Local\\Android\\Sdk
# OR
sdk.dir=C:/Users/Username/AppData/Local/Android/Sdk
```

---

## Quick Reference: Path Types in Project

| File | Path Type | Needs Change? |
|------|-----------|---------------|
| `local.properties` | Absolute (SDK) | ✅ **YES** |
| `build.gradle` | Relative | ❌ No |
| `CMakeLists.txt` | Relative | ❌ No |
| `google-services.json` | Config (not path) | ❌ No |
| Java source code | Asset names (no paths) | ❌ No |
| Native code | APK-relative | ❌ No |
| `.idea/workspace.xml` | IDE cache | ❌ No (auto-regenerates) |

---

## Advanced: Automated Relocation Script

```bash
#!/bin/bash
# relocate_app.sh - Automate app relocation

set -e  # Exit on error

# 1. Detect Android SDK
echo "Detecting Android SDK..."
SDK_PATH=$(find ~/Library/Android ~/Android -name "platform-tools" 2>/dev/null | head -1 | sed 's/\/platform-tools//')

if [ -z "$SDK_PATH" ]; then
    echo "❌ Android SDK not found!"
    echo "Please install Android SDK or set path manually."
    exit 1
fi

echo "✅ Found SDK at: $SDK_PATH"

# 2. Update local.properties
echo "Updating local.properties..."
echo "sdk.dir=$SDK_PATH" > local.properties
echo "✅ local.properties updated"

# 3. Clean old build
echo "Cleaning old build artifacts..."
./gradlew clean
echo "✅ Cleaned"

# 4. Test build
echo "Testing build..."
./gradlew assembleDebug
echo "✅ Build successful!"

echo ""
echo "========================================="
echo "✅ App successfully relocated!"
echo "APK location:"
echo "  app/build/outputs/apk/debug/app-debug.apk"
echo "========================================="
```

**Usage**:
```bash
chmod +x relocate_app.sh
./relocate_app.sh
```

---

## Summary

**TL;DR - Relocation Steps**:

1. Copy entire project folder to new location
2. Update `local.properties` with your Android SDK path
3. Run `./gradlew assembleDebug`
4. Done! ✅

**That's it!** The app is designed to be portable. All other paths are relative or asset-based.

---

## Support

If you encounter issues after relocation:

1. **Check `local.properties`**: Correct SDK path?
2. **Verify SDK installed**: `ls /path/to/Android/Sdk/platform-tools/adb`
3. **Clean build**: `./gradlew clean`
4. **Rebuild**: `./gradlew assembleDebug`
5. **Check logs**: Look for missing files or path errors

**Still stuck?** Check the error message - it usually indicates exactly what's wrong.
