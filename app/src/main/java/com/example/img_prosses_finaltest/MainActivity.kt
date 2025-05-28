package com.example.img_prosses_finaltest

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.img_prosses_finaltest.ui.theme.Img_Prosses_FinalTestTheme
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {

    private fun matToBitmap(mat: Mat): Bitmap {
        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)
        return bmp
    }

    private fun ensureOddKernelSize(value: Int): Int {
        return if (value <= 0) 1 else if (value % 2 == 0) value + 1 else value
    }

    private fun addNoise(mat: Mat, type: String): Mat {
        val noisyMat = mat.clone()
        val rows = mat.rows()
        val cols = mat.cols()
        when (type) {
            "Gaussian" -> {
                val noise = Mat(mat.size(), mat.type())
                Core.randn(noise, 0.0, 30.0) // Mean = 0, StdDev = 30
                Core.add(noisyMat, noise, noisyMat)
            }
            "Salt & Pepper" -> {
                val noiseAmount = (rows * cols * 0.05).toInt() // 5% noise
                for (i in 0 until noiseAmount) {
                    val row = (Math.random() * rows).toInt()
                    val col = (Math.random() * cols).toInt()
                    if (row < rows && col < cols) {
                        val value = if (Math.random() > 0.5) 255.0 else 0.0
                        noisyMat.submat(row, row + 1, col, col + 1).setTo(Scalar(value, value, value))
                    }
                }
            }
        }
        return noisyMat
    }

    private fun processImage(
        mat: Mat,
        func: String,
        originalBitmap: Bitmap?,
        sliderValue: Float
    ): Bitmap {
        val dst = Mat()
        return when (func) {
            "Original Image" -> originalBitmap ?: matToBitmap(mat)
            "Lowpass Filtering" -> {
                val kSize = ensureOddKernelSize(sliderValue.toInt())
                Imgproc.blur(mat, dst, Size(kSize.toDouble(), kSize.toDouble()))
                matToBitmap(dst)
            }
            "Median Filter" -> {
                val kSize = ensureOddKernelSize(sliderValue.toInt())
                Imgproc.medianBlur(mat, dst, kSize)
                matToBitmap(dst)
            }
            "Highpass Filtering" -> {
                val kSize = ensureOddKernelSize(sliderValue.toInt())
                val blurred = Mat()
                Imgproc.GaussianBlur(mat, blurred, Size(kSize.toDouble(), kSize.toDouble()), 0.0)
                Core.subtract(mat, blurred, dst)
                Core.add(dst, Scalar(128.0, 128.0, 128.0), dst)
                matToBitmap(dst)
            }
            "Robert Filter" -> {
                val gray = Mat()
                Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
                val robertX = Mat()
                val robertY = Mat()
                val kernelX = Mat(2, 2, CvType.CV_32F)
                kernelX.setTo(Scalar(0.0))
                kernelX.row(0).col(0).setTo(Scalar(1.0))
                kernelX.row(1).col(1).setTo(Scalar(-1.0))
                val kernelY = Mat(2, 2, CvType.CV_32F)
                kernelY.setTo(Scalar(0.0))
                kernelY.row(0).col(1).setTo(Scalar(1.0))
                kernelY.row(1).col(0).setTo(Scalar(-1.0))
                Imgproc.filter2D(gray, robertX, CvType.CV_32F, kernelX)
                Imgproc.filter2D(gray, robertY, CvType.CV_32F, kernelY)
                Core.magnitude(robertX, robertY, dst)
                Core.normalize(dst, dst, 0.0, 255.0, Core.NORM_MINMAX)
                dst.convertTo(dst, CvType.CV_8U)
                Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
                matToBitmap(dst)
            }
            "Laplacian Filter" -> {
                val gray = Mat()
                Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
                Imgproc.Laplacian(gray, dst, CvType.CV_64F)
                Core.convertScaleAbs(dst, dst)
                Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
                matToBitmap(dst)
            }
            "Frequency Lowpass Filter" -> {
                val gray = Mat()
                Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
                val dft = Mat()
                gray.convertTo(gray, CvType.CV_32F)
                Core.dft(gray, dft, Core.DFT_COMPLEX_OUTPUT, 0)
                val rows = dft.rows()
                val cols = dft.cols()
                val centerX = cols / 2
                val centerY = rows / 2
                val radius = sliderValue.toInt()
                val zeroPixel = Scalar(0.0, 0.0)
                for (i in 0 until rows) {
                    for (j in 0 until cols) {
                        val distance = sqrt(((i - centerY) * (i - centerY) + (j - centerX) * (j - centerX)).toDouble())
                        if (distance > radius) {
                            dft.submat(i, i + 1, j, j + 1).setTo(zeroPixel)
                        }
                    }
                }
                Core.idft(dft, dst, Core.DFT_SCALE or Core.DFT_REAL_OUTPUT, 0)
                Core.normalize(dst, dst, 0.0, 255.0, Core.NORM_MINMAX)
                dst.convertTo(dst, CvType.CV_8U)
                Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
                matToBitmap(dst)
            }
            "Blur (Filter)" -> {
                val kSize = ensureOddKernelSize(sliderValue.toInt())
                Imgproc.GaussianBlur(mat, dst, Size(kSize.toDouble(), kSize.toDouble()), 0.0)
                matToBitmap(dst)
            }
            "Restoration (Dilation)" -> {
                val kSize = ensureOddKernelSize(sliderValue.toInt())
                val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kSize.toDouble(), kSize.toDouble()))
                Imgproc.dilate(mat, dst, kernel)
                matToBitmap(dst)
            }
            "Edge Detection (Canny)" -> {
                val threshold = sliderValue.toDouble()
                val gray = Mat()
                Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
                Imgproc.Canny(gray, dst, threshold, threshold * 2)
                Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
                matToBitmap(dst)
            }
            "Object Detection (Contours)" -> {
                val gray = Mat()
                Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
                Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
                Imgproc.Canny(gray, dst, 75.0, 200.0)
                val contours = ArrayList<MatOfPoint>()
                val hierarchy = Mat()
                Imgproc.findContours(dst, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE)
                val contourMat = mat.clone()
                Imgproc.drawContours(contourMat, contours, -1, Scalar(255.0, 0.0, 0.0), 2)
                matToBitmap(contourMat)
            }
            "Gaussian Noise" -> {
                val noisyMat = addNoise(mat, "Gaussian")
                matToBitmap(noisyMat)
            }
            "Salt & Pepper Noise" -> {
                val noisyMat = addNoise(mat, "Salt & Pepper")
                matToBitmap(noisyMat)
            }
            else -> matToBitmap(mat)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "Failed to load OpenCV")
        } else {
            Log.d("OpenCV", "OpenCV loaded successfully. Version: ${Core.VERSION}")
        }

        setContent {
            Img_Prosses_FinalTestTheme {
                var selectedFunction by remember { mutableStateOf("Original Image") }
                var bitmap by remember { mutableStateOf<Bitmap?>(null) }
                var matSrc by remember { mutableStateOf<Mat?>(null) }
                var sliderValue by remember { mutableStateOf(15f) }
                val context = LocalContext.current

                val launcher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
                    uri?.let {
                        val inputStream = context.contentResolver.openInputStream(uri)
                        val bmp = BitmapFactory.decodeStream(inputStream)
                        inputStream?.close()
                        bmp?.let { bitmapFromGallery ->
                            bitmap = bitmapFromGallery
                            val mat = Mat()
                            Utils.bitmapToMat(bitmapFromGallery, mat)
                            matSrc = mat
                        }
                    }
                }

                LaunchedEffect(selectedFunction, matSrc, sliderValue) {
                    matSrc?.let {
                        bitmap = processImage(it, selectedFunction, bitmap, sliderValue)
                    }
                }

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(
                        modifier = Modifier
                            .padding(innerPadding)
                            .padding(16.dp)
                    ) {
                        Text(text = "OpenCV Version: ${Core.VERSION}", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(10.dp))

                        Button(onClick = { launcher.launch("image/*") }) {
                            Text("Select Image from Gallery")
                        }

                        Spacer(modifier = Modifier.height(20.dp))

                        if (bitmap != null) {
                            val options = listOf(
                                "Original Image",
                                "Lowpass Filtering",
                                "Median Filter",
                                "Highpass Filtering",
                                "Robert Filter",
                                "Laplacian Filter",
                                "Frequency Lowpass Filter",
                                "Blur (Filter)",
                                "Restoration (Dilation)",
                                "Edge Detection (Canny)",
                                "Object Detection (Contours)",
                                "Gaussian Noise",
                                "Salt & Pepper Noise"
                            )

                            var expanded by remember { mutableStateOf(false) }
                            Box {
                                Text(
                                    text = "Function: $selectedFunction",
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clickable { expanded = true }
                                        .padding(12.dp),
                                    style = MaterialTheme.typography.bodyLarge
                                )
                                DropdownMenu(
                                    expanded = expanded,
                                    onDismissRequest = { expanded = false },
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    options.forEach { option ->
                                        DropdownMenuItem(
                                            text = { Text(option) },
                                            onClick = {
                                                selectedFunction = option
                                                expanded = false
                                            }
                                        )
                                    }
                                }
                            }

                            Spacer(modifier = Modifier.height(10.dp))

                            if (selectedFunction != "Original Image" &&
                                selectedFunction != "Object Detection (Contours)" &&
                                selectedFunction != "Gaussian Noise" &&
                                selectedFunction != "Salt & Pepper Noise") {
                                Text("Adjust Parameter", style = MaterialTheme.typography.labelLarge)
                                Slider(
                                    value = sliderValue,
                                    onValueChange = { sliderValue = it },
                                    valueRange = when (selectedFunction) {
                                        "Blur (Filter)", "Restoration (Dilation)", "Lowpass Filtering", "Median Filter", "Highpass Filtering" -> 1f..49f
                                        "Edge Detection (Canny)" -> 10f..300f
                                        "Frequency Lowpass Filter" -> 10f..100f
                                        else -> 1f..100f
                                    },
                                    steps = 20
                                )
                                Text("Value: ${sliderValue.toInt()}")
                                Spacer(modifier = Modifier.height(10.dp))
                            }

                            Image(
                                bitmap = bitmap!!.asImageBitmap(),
                                contentDescription = null,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(400.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}