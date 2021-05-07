package com.ml.projects.age_genderdetection

import android.graphics.Bitmap
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

// Helper class for gender classification model
class GenderClassificationModel {

    // Input image size for our model
    private val inputImageSize = 128

    // Input image processor for model inputs.
    // Resize + Normalize
    private val inputImageProcessor =
            ImageProcessor.Builder()
                    .add( ResizeOp( inputImageSize , inputImageSize , ResizeOp.ResizeMethod.BILINEAR ) )
                    .add( NormalizeOp( 0f , 255f ) )
                    .build()

    // Time taken by the model ( in milliseconds ) to perform the inference.
    var inferenceTime : Long = 0

    // Interpreter object to use the TFLite model.
    var interpreter : Interpreter? = null

    // Given an input image, return the predicted gender.
    // Note: This is a suspended function, and will run within a CoroutineScope.
    suspend fun predictGender( image : Bitmap ) = withContext( Dispatchers.Default ) {
        val start = System.currentTimeMillis()
        // Input tensor shape -> [ 1 , 128 , 128 , 3 ]
        val tensorInputImage = TensorImage.fromBitmap( image )
        // Output tensor shape -> [ 1 , 2 ]
        val genderOutputArray = Array( 1 ){ FloatArray( 2 ) }
        val processedImageBuffer = inputImageProcessor.process( tensorInputImage ).buffer
        interpreter?.run(
            processedImageBuffer,
            genderOutputArray
        )
        inferenceTime = System.currentTimeMillis() - start
        return@withContext genderOutputArray[ 0 ]
    }


}