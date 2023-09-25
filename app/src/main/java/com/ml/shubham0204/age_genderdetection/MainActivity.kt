package com.ml.shubham0204.age_genderdetection

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.media.ExifInterface
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Camera
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Image
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.core.content.FileProvider
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.ml.shubham0204.age_genderdetection.ui.theme.AppTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.io.IOException
import kotlin.math.floor


class MainActivity : ComponentActivity() {

    // Initialize the MLKit FaceDetector
    private val realTimeOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .build()
    private val firebaseFaceDetector = FaceDetection.getClient(realTimeOpts)

    // CoroutineScope in which we'll run our coroutines.
    private val coroutineScope = CoroutineScope( Dispatchers.Main )

    // For reading the full-sized picture
    private lateinit var currentPhotoPath : String

    // TFLite interpreters for both the models
    lateinit var ageModelInterpreter: Interpreter
    lateinit var genderModelInterpreter: Interpreter
    private lateinit var ageEstimationModel: AgeEstimationModel
    private lateinit var genderClassificationModel: GenderClassificationModel

    private val compatList = CompatibilityList()
    // Model names, as shown in the spinner.
    private val modelNames = arrayOf(
        "Age/Gender Detection Model ( Quantized ) ",
        "Age/Gender Detection Model ( Non-quantized )",
        "Age/Gender Detection Lite Model ( Quantized )",
        "Age/Gender Detection Lite Model ( Non-quantized )",
    )
    // File-paths of the models ( in the assets folder ) corresponding to the models in `modelNames`.
    private val modelFilenames = arrayOf(
        arrayOf("model_age_q.tflite", "model_gender_q.tflite"),
        arrayOf("model_age_nonq.tflite", "model_gender_nonq.tflite"),
        arrayOf("model_lite_age_q.tflite", "model_lite_gender_q.tflite"),
        arrayOf("model_lite_age_nonq.tflite", "model_lite_gender_nonq.tflite"),
    )
    // Default model filename
    private var modelFilename = arrayOf( "model_age_q.tflite", "model_gender_q.tflite" )

    private val shift = 5

    private val showModelInitDialog = mutableStateOf( true )
    private val showImageSelectorDialog = mutableStateOf( true )
    private val showProgressDialog = mutableStateOf( false )
    private val predictedAge = mutableStateOf( "" )
    private val predictedGender = mutableStateOf( "" )
    private var subjectImage: Bitmap? = null
    private var ageInferenceTime = 0L
    private var genderInferenceTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AppTheme {
                Surface( modifier = Modifier
                    .background(Color.White)
                    .fillMaxSize() ) {
                    ActivityUI()
                }
            }
        }
    }

    @Preview
    @Composable
    private fun ActivityUI() {
        Column {
            PredictionsLayout()
            ImageSelectorLayout()
            ModelInitDialog()
            ProgressDialog()
        }
    }

    @Composable
    private fun PredictionsLayout() {
        val showImageSelectLayout by remember{ showImageSelectorDialog }
        AnimatedVisibility( !showImageSelectLayout ) {
            Column(
                modifier = Modifier.verticalScroll(rememberScrollState()) ,
                horizontalAlignment = Alignment.CenterHorizontally ) {
                Row(
                    horizontalArrangement = Arrangement.End ,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    Button(
                        onClick = {
                        showModelInitDialog.value = true
                        showImageSelectorDialog.value = true
                    }) { Text(text = "Reinitialize Model") }
                }
                if( subjectImage != null ) {
                    Image(
                        modifier = Modifier
                            .padding( 16.dp ),
                        bitmap = subjectImage!!.asImageBitmap(),
                        contentDescription = "Picture selected" )
                }
                ResultsCard()
                Row(
                    modifier = Modifier
                        .fillMaxSize()
                        .weight(1f),
                    verticalAlignment = Alignment.CenterVertically ,
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    Button(onClick = { dispatchTakePictureIntent() }) {
                        Image(imageVector = Icons.Default.Camera, contentDescription = "Take Photo" , colorFilter= ColorFilter.tint(Color.White))
                        Text(text = "Take Photo") }
                    Button(onClick = { dispatchSelectPictureIntent() }) {
                        Image(imageVector = Icons.Default.Image, contentDescription = "Select Picture" , colorFilter= ColorFilter.tint(Color.White))
                        Text(text = "Select Picture" ) }
                }
            }
        }
    }

    @Composable
    private fun ResultsCard() {
        val predictedAge by remember{ predictedAge }
        val predictedGender by remember{ predictedGender }
        Card(
            elevation = CardDefaults.cardElevation( defaultElevation = 8.dp ) ,
            colors = CardDefaults.cardColors( containerColor = Color.White ) ,
            modifier = Modifier.padding( 16.dp )
        ) {
            Row(
                horizontalArrangement = Arrangement.SpaceEvenly ,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp)
            ) {
                Column( horizontalAlignment = Alignment.CenterHorizontally ) {
                    Text(text = "Predicted Age" , style=MaterialTheme.typography.labelLarge)
                    Text(text = predictedAge)
                }
                Column( horizontalAlignment = Alignment.CenterHorizontally ) {
                    Text(text = "Predicted Gender" , style=MaterialTheme.typography.labelLarge)
                    Text(text = predictedGender)
                }
            }
            Column( modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp) ) {
                Text(text = "Age Estimation Inference Time: $ageInferenceTime ms" ,
                    style=MaterialTheme.typography.labelSmall )
                Text(text = "Gender Detection Inference Time: $genderInferenceTime ms" ,
                    style=MaterialTheme.typography.labelSmall )
            }
        }
    }

    @Composable
    private fun ImageSelectorLayout() {
        val showImageSelectLayout by remember{ showImageSelectorDialog }
        AnimatedVisibility( showImageSelectLayout ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally ,
                verticalArrangement = Arrangement.SpaceEvenly ,
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
            ) {
                Column(
                    verticalArrangement = Arrangement.Center ,
                    horizontalAlignment = Alignment.CenterHorizontally ,
                    modifier = Modifier
                        .fillMaxSize()
                        .weight(2f)
                ) {
                    Text(
                        modifier = Modifier.padding( 16.dp ) ,
                        text = "Select a Picture or Take a Photo" ,
                        textAlign = TextAlign.Center ,
                        style = MaterialTheme.typography.displayMedium
                    )
                }
                Row(
                    modifier = Modifier
                        .fillMaxSize()
                        .weight(1f),
                    verticalAlignment = Alignment.CenterVertically ,
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    Button(onClick = { dispatchTakePictureIntent() }) {
                        Image(imageVector = Icons.Default.Camera, contentDescription = "Take Photo" , colorFilter= ColorFilter.tint(Color.White) )
                        Text(text = "Take Photo") }
                    Button(onClick = { dispatchSelectPictureIntent() }) {
                        Image(imageVector = Icons.Default.Image, contentDescription = "Select Picture" , colorFilter= ColorFilter.tint(Color.White))
                        Text(text = "Select Picture" ) }
                }
            }
        }
    }

    @Composable
    private fun ProgressDialog() {
        val show by remember{ showProgressDialog }
        if( show ) {
            Dialog(onDismissRequest = { /*TODO*/ }) {
                Surface( modifier = Modifier.background(Color.White)) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally ,
                        modifier = Modifier
                            .background(Color.White)
                            .padding(64.dp)
                    ) {
                        CircularProgressIndicator()
                        Text(text = "Processing", style=MaterialTheme.typography.titleLarge)
                    }
                }
            }
        }
    }

    @Composable
    private fun ModelInitDialog() {
        val showDialog by remember{ showModelInitDialog }
        if( showDialog ) {
            Dialog(
                onDismissRequest = {

                }
            ) {
                ModelInitDialogLayout()
            }
        }
    }

    @Composable
    private fun ModelInitDialogLayout() {
        var useGpu by remember{ mutableStateOf( false ) }
        var useNNApi by remember{ mutableStateOf( false ) }
        Card( modifier = Modifier
            .verticalScroll(rememberScrollState())
            .fillMaxWidth()
            .padding(8.dp) ) {
            Column(
                verticalArrangement = Arrangement.spacedBy( 16.dp ) ,
                modifier = Modifier
                    .padding(8.dp)
            ) {
                Text(
                    text = "Initialize the Model" ,
                    textAlign = TextAlign.Center ,
                    modifier = Modifier
                        .fillMaxWidth() ,
                    style = MaterialTheme.typography.titleLarge
                )
                Text(
                    text = "Choose a TFLite Model" ,
                    modifier = Modifier
                        .fillMaxWidth(),
                    style = MaterialTheme.typography.labelMedium
                )
                ModelsList()
                Text(
                    text = "Other Options" ,
                    modifier = Modifier
                        .fillMaxWidth(),
                    style = MaterialTheme.typography.labelMedium
                )
                CheckBox(text = "Use NNAPI", onCheckChecked = { useNNApi = it })
                CheckBox(text = "Use GPU Delegate", onCheckChecked = { useGpu = it })
                Row(
                    modifier = Modifier.fillMaxWidth() ,
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    Button(onClick = {
                        val options = Interpreter.Options().apply {
                            if ( useGpu ) {
                                addDelegate(GpuDelegate( compatList.bestOptionsForThisDevice ) )
                            }
                            if ( useNNApi ) {
                                addDelegate(NnApiDelegate())
                            }
                        }
                        coroutineScope.launch {
                            initModels(options)
                        }
                        showModelInitDialog.value = false
                    }) { Text(text = "Initialize") }
                    Button(onClick = {
                        showModelInitDialog.value = false
                    }) { Text(text = "Close" ) }
                }
            }
        }
    }

    @Composable
    private fun CheckBox( text: String, onCheckChecked: ((Boolean) -> Unit) ) {
        var isChecked by remember{ mutableStateOf( false ) }
        Row(
            verticalAlignment = Alignment.CenterVertically ,
            modifier = Modifier
                .fillMaxWidth()
                .clickable {
                    isChecked = !isChecked
                }
        ) {
            Checkbox(checked = isChecked, onCheckedChange = {
                onCheckChecked( it )
                isChecked = !isChecked
            })
            Text(
                text = text ,
                style = MaterialTheme.typography.labelLarge
            )
        }
    }

    @Composable
    private fun ModelsList( modifier : Modifier = Modifier ) {
        var selectedModelIndex by remember{ mutableIntStateOf( 0 ) }
        Column(
            modifier = modifier ,
            verticalArrangement = Arrangement.spacedBy( 4.dp )
        ) {
            modelNames.forEachIndexed { index , s ->
                Row(
                    verticalAlignment = Alignment.CenterVertically ,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            selectedModelIndex = index
                            modelFilename = modelFilenames[index]
                        }
                        .background(
                            if (index == selectedModelIndex) Color.Blue else {
                                Color.White
                            },
                            RoundedCornerShape(16.dp)
                        )
                        .padding(16.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = "Selected Model" ,
                        tint = if( index == selectedModelIndex ) Color.White else { Color.Black }
                    )
                    Spacer(modifier = Modifier.width(16.dp))
                    Text(
                        color = if( index == selectedModelIndex ) Color.White else { Color.Black } ,
                        text = s ,
                        style = MaterialTheme.typography.labelLarge
                    )
                }
            }
        }
    }


    // Suspending function to initialize the TFLite interpreters.
    private suspend fun initModels(options: Interpreter.Options) = withContext( Dispatchers.Default ) {
        ageModelInterpreter = Interpreter(FileUtil.loadMappedFile( applicationContext , modelFilename[0]), options )
        genderModelInterpreter = Interpreter(FileUtil.loadMappedFile( applicationContext , modelFilename[1]), options )
        withContext( Dispatchers.Main ){
            ageEstimationModel = AgeEstimationModel().apply {
                interpreter = ageModelInterpreter
            }
            genderClassificationModel = GenderClassificationModel().apply {
                interpreter = genderModelInterpreter
            }
            // Notify the user once the models have been initialized.
            Toast.makeText( applicationContext , "Models initialized." , Toast.LENGTH_LONG ).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        ageModelInterpreter.close()
        genderModelInterpreter.close()
    }

    private fun detectFaces(image: Bitmap) {
        val inputImage = InputImage.fromBitmap(image, 0)
        // Pass the clicked picture to MLKit's FaceDetector.
        firebaseFaceDetector.process(inputImage)
                .addOnSuccessListener { faces ->
                    if ( faces.size != 0 ) {
                        // Set the cropped Bitmap
                        subjectImage = cropToBBox(image, faces[0].boundingBox)
                        // Launch a coroutine
                        coroutineScope.launch {

                            // Predict the age and the gender.
                            val age = ageEstimationModel.predictAge(cropToBBox(image, faces[0].boundingBox))
                            val gender = genderClassificationModel.predictGender(cropToBBox(image, faces[0].boundingBox))

                            ageInferenceTime = ageEstimationModel.inferenceTime
                            genderInferenceTime = genderClassificationModel.inferenceTime

                            // Show the final output to the user.
                            predictedAge.value = floor( age.toDouble() ).toInt().toString()
                            predictedGender.value = if ( gender[ 0 ] > gender[ 1 ] ) { "Male" } else { "Female" }
                            showImageSelectorDialog.value = false
                            showProgressDialog.value = false
                        }
                    }
                    else {
                        // Show a dialog to the user when no faces were detected.
                        showProgressDialog.value = false
                        Toast.makeText( this ,
                            "We could not find any faces in the image you just clicked. " +
                                    "Try clicking another image or improve the lightning or the device rotation." ,
                            Toast.LENGTH_LONG
                        ).show()
                    }


                }
    }


    private fun cropToBBox(image: Bitmap, bbox: Rect) : Bitmap {
        return Bitmap.createBitmap(
            image,
            bbox.left - 0 * shift,
            bbox.top + shift,
            bbox.width() + 0 * shift,
            bbox.height() + 0 * shift
        )
    }


    // Create a temporary file, for storing the full-sized picture taken by the user.
    private fun createImageFile() : File {
        val imagesDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("image", ".jpg", imagesDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    // Dispatch an Intent which opens the gallery application for the user.
    private fun dispatchSelectPictureIntent() {
        selectPictureLauncher.launch( "image/*" )
    }

    // Dispatch an Intent which opens the camera application for the user.
    // The code is from -> https://developer.android.com/training/camera/photobasics#TaskPath
    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent( MediaStore.ACTION_IMAGE_CAPTURE )
        if ( takePictureIntent.resolveActivity( packageManager ) != null ) {
            val photoFile: File? = try {
                createImageFile()
            }
            catch (ex: IOException) {
                null
            }
            photoFile?.also {
                val photoURI = FileProvider.getUriForFile(
                    this,
                    "com.ml.projects.age_genderdetection", it
                )
                takePictureLauncher.launch( photoURI )
            }
        }
    }

    private val takePictureLauncher = registerForActivityResult( ActivityResultContracts.TakePicture() ) {
        if( it ) {
            var bitmap = BitmapFactory.decodeFile( currentPhotoPath )
            val exifInterface = ExifInterface( currentPhotoPath )
            bitmap =
                when (exifInterface.getAttributeInt( ExifInterface.TAG_ORIENTATION , ExifInterface.ORIENTATION_UNDEFINED )) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap( bitmap , 90f )
                    ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap( bitmap , 180f )
                    ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap( bitmap , 270f )
                    else -> bitmap
                }
            showProgressDialog.value = true
            // Pass the clicked picture to `detectFaces`.
            detectFaces( bitmap!! )
        }
    }

    private val selectPictureLauncher = registerForActivityResult( ActivityResultContracts.GetContent() ) {
        if( it != null ) {
            val inputStream = contentResolver.openInputStream( it )
            val bitmap = BitmapFactory.decodeStream( inputStream )
            inputStream?.close()
            showProgressDialog.value = true
            // Pass the clicked picture to `detectFaces`.
            detectFaces( bitmap!! )
        }
    }

    private fun rotateBitmap(original: Bitmap, degrees: Float): Bitmap? {
        val matrix = Matrix()
        matrix.preRotate(degrees)
        return Bitmap.createBitmap(original, 0, 0, original.width, original.height, matrix, true)
    }

}

