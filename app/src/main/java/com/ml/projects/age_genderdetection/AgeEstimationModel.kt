package com.ml.projects.age_genderdetection

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

// Helper class for age estimation model
class AgeEstimationModel {

    // Input image size for our model
    private val inputImageSize = 200

    // Image processor for model inputs.
    // Resize + Normalize
    private val inputImageProcessor =
            ImageProcessor.Builder()
                    .add(ResizeOp(inputImageSize, inputImageSize, ResizeOp.ResizeMethod.BILINEAR))
                    .add(NormalizeOp(0f, 255f))
                    .build()

    // The model returns a normalized value for the age i.e in range ( 0 , 1 ].
    // To get the age, we multiply the model's output with p.
    private val p = 116

    // Time taken by the model ( in milliseconds ) to perform the inference.
    var inferenceTime : Long = 0

    // Interpreter object to use the TFLite model.
    var interpreter : Interpreter? = null

    // Given an input image, return the estimated age.
    // Note: This is a suspended function, and will run within a CoroutineScope.
    suspend fun predictAge(image: Bitmap) = withContext( Dispatchers.Main ) {
        val start = System.currentTimeMillis()
        // Input image tensor shape -> [ 1 , 200 , 200 , 3 ]
        val tensorInputImage = TensorImage.fromBitmap(image)
        // Output tensor shape -> [ 1 , 1 ]
        val ageOutputArray = Array(1){ FloatArray(1) }
        val processedImageBuffer = inputImageProcessor.process(tensorInputImage).buffer
        interpreter?.run(
                processedImageBuffer,
                ageOutputArray
        )
        inferenceTime = System.currentTimeMillis() - start
        return@withContext ageOutputArray[0][0] * p
    }
}



