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
import kotlin.math.pow
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
        val channels = mat.channels()
        when (type) {
            "Gaussian" -> {
                val noise = Mat(mat.size(), mat.type())
                Core.randn(noise, 0.0, 30.0)
                Core.add(noisyMat, noise, noisyMat)
            }
            "Salt & Pepper" -> {
                val noiseAmount = (rows * cols * 0.05).toInt()
                for (i in 0 until noiseAmount) {
                    val row = (Math.random() * rows).toInt()
                    val col = (Math.random() * cols).toInt()
                    val value = if (Math.random() > 0.5) 255.0 else 0.0
                    val pixel = DoubleArray(channels) { value }
                    if (row in 0 until rows && col in 0 until cols) {
                        noisyMat.put(row, col, *pixel) // ✅ giải quyết lỗi ở đây
                    }
                }
            }
        }
        return noisyMat
    }



    private fun pointOperation(mat: Mat, type: String, param: Double): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val floatMat = Mat()
        gray.convertTo(floatMat, CvType.CV_32F)
        val dst = Mat()
        when (type) {
            "Negative" -> {
                val inverted = Mat()
                val white = Mat.ones(gray.size(), gray.type())
                white.convertTo(white, gray.type(), 255.0)
                Core.subtract(white, gray, inverted)
                inverted.copyTo(dst)
            }
            "Log" -> {
                Core.add(floatMat, Scalar.all(1.0), floatMat)
                Core.log(floatMat, floatMat)
                Core.normalize(floatMat, floatMat, 0.0, 255.0, Core.NORM_MINMAX)
                floatMat.convertTo(dst, CvType.CV_8U)
            }
            "Gamma" -> {
                Core.pow(floatMat, param, floatMat)
                Core.normalize(floatMat, floatMat, 0.0, 255.0, Core.NORM_MINMAX)
                floatMat.convertTo(dst, CvType.CV_8U)
            }
            "Threshold" -> {
                Imgproc.threshold(gray, dst, param, 255.0, Imgproc.THRESH_BINARY)
            }
            else -> return mat
        }
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
        return dst
    }

    private fun histogramEqualization(mat: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val equalized = Mat()
        Imgproc.equalizeHist(gray, equalized)
        Imgproc.cvtColor(equalized, equalized, Imgproc.COLOR_GRAY2BGR)
        return equalized
    }
    private fun applyCustomMedianFilter(mat: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val dst = gray.clone()
        for (i in 1 until gray.rows() - 1) {
            for (j in 1 until gray.cols() - 1) {
                val pixels = mutableListOf<Double>()
                for (dx in -1..1) {
                    for (dy in -1..1) {
                        pixels.add(gray.get(i + dx, j + dy)[0])
                    }
                }
                pixels.sort()
                dst.put(i, j, pixels[4])
            }
        }
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
        return dst
    }

    private fun applyCustomLaplacianFilter(mat: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)

        val kernel = Mat(3, 3, CvType.CV_32F)
        kernel.put(0, 0, floatArrayOf(
            0f, 1f, 0f,
            1f, -4f, 1f,
            0f, 1f, 0f
        ))

        val dst = Mat()
        Imgproc.filter2D(gray, dst, CvType.CV_8U, kernel)
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
        return dst
    }


    private fun applyRobertCross1(mat: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)

        val kernel = Mat(3, 3, CvType.CV_32F)
        kernel.put(0, 0, floatArrayOf(
            0f, 0f, 0f,
            0f, -1f, 0f,
            0f, 0f, 1f
        ))

        val dst = Mat()
        Imgproc.filter2D(gray, dst, CvType.CV_8U, kernel)
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
        return dst
    }


    private fun applyRobertCross2(mat: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)

        val kernel = Mat(3, 3, CvType.CV_32F)
        kernel.put(0, 0, floatArrayOf(
            0f, 0f, 0f,
            0f, 0f, -1f,
            0f, 1f, 0f
        ))

        val dst = Mat()
        Imgproc.filter2D(gray, dst, CvType.CV_8U, kernel)
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
        return dst
    }


    private fun applyRobertCombined(mat: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val r1 = applyRobertCross1(mat)
        val r2 = applyRobertCross2(mat)
        val r1Gray = Mat()
        val r2Gray = Mat()
        Imgproc.cvtColor(r1, r1Gray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.cvtColor(r2, r2Gray, Imgproc.COLOR_BGR2GRAY)
        val combined = Mat()
        Core.add(gray, r1Gray, combined)
        Core.add(combined, r2Gray, combined)
        Imgproc.cvtColor(combined, combined, Imgproc.COLOR_GRAY2BGR)
        return combined
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
                var selectedFunction by remember { mutableStateOf("Ảnh gốc (Original Image)") }
                var bitmap by remember { mutableStateOf<Bitmap?>(null) }
                var matSrc by remember { mutableStateOf<Mat?>(null) }
                var sliderValue by remember { mutableStateOf(1.0f) }
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
                        val processed = when (selectedFunction) {
                            "Ảnh gốc (Original Image)" -> it
                            "Âm bản (Negative)" -> pointOperation(it, "Negative", 0.0)
                            "Biến đổi Log (Log Transformation)" -> pointOperation(it, "Log", 0.0)
                            "Biến đổi Gamma (Gamma Transformation)" -> pointOperation(it, "Gamma", sliderValue.toDouble())
                            "Nhị phân hóa (Global Thresholding)" -> pointOperation(it, "Threshold", sliderValue.toDouble())
                            "Cân bằng Histogram (Histogram Equalization)" -> histogramEqualization(it)
                            "Thêm nhiễu Gaussian (Gaussian Noise)" -> addNoise(it, "Gaussian")
                            "Thêm nhiễu muối tiêu (Salt & Pepper Noise)" -> addNoise(it, "Salt & Pepper")
                            "Lọc trung vị thủ công (Median Manual)" -> applyCustomMedianFilter(it)
                            "Lọc Laplacian thủ công" -> applyCustomLaplacianFilter(it)
                            "Robert hướng 1" -> applyRobertCross1(it)
                            "Robert hướng 2" -> applyRobertCross2(it)
                            "Robert tổng hợp" -> applyRobertCombined(it)
                            else -> it
                        }
                        bitmap = matToBitmap(processed)
                    }
                }

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(modifier = Modifier.padding(innerPadding).padding(16.dp)) {
                        Text("OpenCV Version: ${Core.VERSION}", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(10.dp))

                        Button(onClick = { launcher.launch("image/*") }) {
                            Text("Chọn ảnh từ thư viện")
                        }

                        Spacer(modifier = Modifier.height(20.dp))

                        if (bitmap != null) {
                            val options = listOf(
                                "Ảnh gốc (Original Image)",
                                "Âm bản (Negative)",
                                "Biến đổi Log (Log Transformation)",
                                "Biến đổi Gamma (Gamma Transformation)",
                                "Nhị phân hóa (Global Thresholding)",
                                "Cân bằng Histogram (Histogram Equalization)",
                                "Thêm nhiễu Gaussian (Gaussian Noise)",
                                "Thêm nhiễu muối tiêu (Salt & Pepper Noise)",
                                "Lọc trung vị thủ công (Median Manual)",
                                "Lọc Laplacian thủ công",
                                "Robert hướng 1",
                                "Robert hướng 2",
                                "Robert tổng hợp"
                            )

                            var expanded by remember { mutableStateOf(false) }
                            Box {
                                Text(
                                    text = "Kỹ thuật xử lý: $selectedFunction",
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

                            if (selectedFunction.contains("Gamma") || selectedFunction.contains("Threshold")) {
                                Text("Điều chỉnh tham số", style = MaterialTheme.typography.labelLarge)
                                Slider(
                                    value = sliderValue,
                                    onValueChange = { sliderValue = it },
                                    valueRange = 0.1f..5.0f,
                                    steps = 20
                                )
                                Text("Giá trị: ${sliderValue}")
                            }

                            Spacer(modifier = Modifier.height(10.dp))

                            Image(
                                bitmap = bitmap!!.asImageBitmap(),
                                contentDescription = null,
                                modifier = Modifier.fillMaxWidth().height(400.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}
