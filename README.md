# Age + Gender Estimation in Android with TensorFlow

Contents

Exploring the Android project
 * [Overview](#overview)
 * [Usage](#usage)
 * [Project Configuration ( + Firebase Services )](#project-configuration)
 * [TensorFlow Lite models](#tensorflow-lite-models)
 * [NNAPI and `GpuDelegate` compatibility](#nnapi-and-gpudelegate-compatibility)



### 👉🏻 Overview

The following GIF displays the basic functionality of the app. You may observe the four options provided for "Choose Model" and also the NNAPI and GPU Delegate options. The inference time may change depending on the device you're using.

![working_of_app](images/app_working_s.gif) 

### 👉🏻 Usage

* Open the app. A dialog box saying "Initialize a Model" pops up. Select any one of the four models.
* Select the additional options, like "Use NNAPI" and "Use GPU". If any of these options are not available on your device, a message will be shown for the same. See [NNAPI and `GpuDelegate` compatibility](#nnapi-and-gpudelegate-compatibility)
* Once the models are initialized, tap "Take Photo" to open the default camera app.
* If none of the faces are identified in the picture, a dialog will be displayed, prompting you to take another picture. If everything goes fine, the results appear on the screen in a couple of seconds.
* Tap "Reinitialize" to try another model provided in the app.

> **Note: If the picture clicked by the user contains multiple faces, results will be shown for a single face only. This is a drawback and we'll try to improve it in the next releases.**

### 👉🏻 Project Configuration

The project has been configured with the below settings, which can be easily found in the app's `build.gradle` file.
```
// SDK Info
compileSdkVersion 30  
buildToolsVersion "30.0.0"  

// App info
applicationId "com.ml.projects.age_genderdetection"  
minSdkVersion 23  
targetSdkVersion 30  

// Version info
versionCode 1  
versionName "1.0"
```

The versions of the Android Gradle plugin and the Gradle Version are mentioned below.

```
Android Gradle Plugin Version: 4.1.3
Gradle Version: 6.5
```

As mentioned, the project utilizes Firebase, and specifically [MLKit FaceDetector](https://firebase.google.com/docs/ml-kit/detect-faces). To connect the app with Firebase, follow these [instructions](https://firebase.google.com/docs/android/setup#register-app). Download the `google-services.json` file and place it in the `app` folder. The Firebase library required for the face detection functionality is added in the app's `build.gradle`,

```
implementation 'com.google.android.gms:play-services-mlkit-face-detection:16.1.5'
```

> Why isn't the `google-services.json` file shared alongside the code in this repo?
> That's because the file contains credentials unique for every user. ( You may need to create a Firebase project. Follow the instructions cited above )

###  👉🏻 TensorFlow Lite models

To enable TFLite capabilities in our Android project, we add the below libraries to our project,

```
implementation 'org.tensorflow:tensorflow-lite:2.4.0'  
implementation 'org.tensorflow:tensorflow-lite-gpu:2.4.0'  
implementation 'org.tensorflow:tensorflow-lite-support:0.1.0'
```
> See the [packages on JFrong Bintray](https://bintray.com/google/tensorflow).

All TFLite models are placed in the app's `assets` folder. In order to disable the compression performed on files present in the `assets` folder, we add the following flag in app's `build.gradle` ,

```
android {  
  ...
  aaptOptions {  
     noCompress "tflite"  
  }
  ...
}  
```
The names of these models are stored in the `modelFilenames` array in `MainActivity.kt`,

```
private val modelFilenames = arrayOf(  
    arrayOf("model_v6_age_q.tflite", "model_v6_gender_q.tflite"),  
    arrayOf("model_v6_age_nonq.tflite", "model_v6_gender_nonq.tflite"),  
    arrayOf("model_v6_lite_age_q.tflite", "model_v6_lite_gender_q.tflite"),  
    arrayOf("model_v6_lite_age_nonq.tflite", "model_v6_lite_gender_nonq.tflite"),  
)
```
Whenever the user selects a particular model, we use the `FileUtil.loadMappedFile()` method to get `MappedByteBuffer` of the model, which is then passed to the constructor of  `Interpreter`,

```
ageModelInterpreter = Interpreter(FileUtil.loadMappedFile( applicationContext , "model_v6_age_q.tflite"), options )
```
> Note: The method `FileUtil.loadMappedBuffer()` comes from the `tf-lite-support` library, which helps us parse TFLite models and also to preprocess model inputs.

### 👉🏻 NNAPI and `GpuDelegate` compatibility

The app offers acceleration through the means of NNAPI and `GpuDelegate` provided by TensorFlow Lite.

* [TensorFlow Lite NNAPI delegate](nsorflow.org/lite/performance/nnapi)
* [TensorFlow Lite GPU delegate](https://www.tensorflow.org/lite/performance/gpu)

As mentioned in the docs, NNAPI is compatible for Android devices running Android Pie ( API level 27 ) and above. The app checks this compatibility in `MainActivity.kt`,

```
if ( Build.VERSION.SDK_INT < Build.VERSION_CODES.P ) {  
    useNNApiCheckBox.isEnabled = false  
    useNNApiCheckBox.text = "Use NNAPI ( Not available as your Android version is less than 9 ( Android Pie )."  
    useNNApi = false  
}
```

The `GpuDelegate` also performs the following compatibility check,

```
if ( !compatList.isDelegateSupportedOnThisDevice ){  
    useGPUCheckBox.isEnabled = false  
    useGPUCheckBox.text = "Use GPU ( GPU acceleration is not available on this device )."  
    useGpu = false  
}
```


### 👨🏻‍✈️ License

```
MIT License  
  
Copyright (c) 2021 Shubham Panchal  
  
Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  
  
The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.  
  
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.
```