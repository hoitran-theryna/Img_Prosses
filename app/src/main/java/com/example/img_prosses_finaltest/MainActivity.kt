package com.example.img_prosses_finaltest

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.pdf.PdfDocument
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.enableEdgeToEdge
import androidx.annotation.RequiresApi
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.img_prosses_finaltest.ui.theme.Img_Prosses_FinalTestTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.*

class MainActivity : ComponentActivity() {
    private fun matToBitmap(mat: Mat): Bitmap {
        val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(mat, bmp)
        return bmp
    }

    private fun translateImage(src: Mat): Mat {
        val tx = 50.0
        val ty = 30.0
        val transMat = Mat(2, 3, CvType.CV_32F)
        transMat.put(0, 0, 1.0, 0.0, tx, 0.0, 1.0, ty)
        val dst = Mat()
        Imgproc.warpAffine(src, dst, transMat, src.size())
        return dst
    }

    private fun rotateImage(src: Mat, angle: Double): Mat {
        val (h, w) = src.size().let { it.height to it.width }
        val center = Point(w / 2.0, h / 2.0)
        val rotationMatrix = Imgproc.getRotationMatrix2D(center, angle, 1.0)
        val cosTheta = abs(cos(angle * PI / 180.0))
        val sinTheta = abs(sin(angle * PI / 180.0))
        val newWidth = (h * sinTheta + w * cosTheta).toInt()
        val newHeight = (h * cosTheta + w * sinTheta).toInt()
        rotationMatrix.put(0, 2, rotationMatrix.get(0, 2)[0] + (newWidth - w) / 2.0)
        rotationMatrix.put(1, 2, rotationMatrix.get(1, 2)[0] + (newHeight - h) / 2.0)
        val dst = Mat()
        Imgproc.warpAffine(src, dst, rotationMatrix, Size(newWidth.toDouble(), newHeight.toDouble()))
        return dst
    }

    private fun scaleImage(src: Mat, scaleX: Double = 0.5, scaleY: Double = 0.5): Mat {
        val (h, w) = src.size().let { it.height to it.width }
        val newSize = Size(w * scaleX, h * scaleY)
        val dst = Mat()
        Imgproc.resize(src, dst, newSize, 0.0, 0.0, Imgproc.INTER_LINEAR)
        return dst
    }

    private fun flipImage(src: Mat, flipMode: Int = 1): Mat {
        val dst = Mat()
        Core.flip(src, dst, flipMode)
        return dst
    }

    private fun cropImage(src: Mat, startX: Int, startY: Int, width: Int, height: Int): Mat {
        val preview = src.clone()
        Imgproc.rectangle(
            preview,
            Point(startX.toDouble(), startY.toDouble()),
            Point((startX + width).toDouble(), (startY + height).toDouble()),
            Scalar(0.0, 255.0, 0.0),
            2
        )
        val rect = Rect(
            startX.coerceIn(0, src.cols() - 1),
            startY.coerceIn(0, src.rows() - 1),
            width.coerceAtMost(src.cols() - startX),
            height.coerceAtMost(src.rows() - startY)
        )
        return Mat(src, rect)
    }

    private fun splitChannels(src: Mat): List<Bitmap> {
        val channels = ArrayList<Mat>()
        Core.split(src, channels)
        return listOf(channels[2], channels[1], channels[0]).map { ch ->
            val result = Mat()
            val merged = mutableListOf(Mat.zeros(src.size(), CvType.CV_8U), Mat.zeros(src.size(), CvType.CV_8U), Mat.zeros(src.size(), CvType.CV_8U))
            when (channels.indexOf(ch)) {
                0 -> merged[2] = ch // Red
                1 -> merged[1] = ch // Green
                2 -> merged[0] = ch // Blue
            }
            Core.merge(merged, result)
            matToBitmap(result)
        }
    }

    private fun applyGlobalThreshold(mat: Mat, threshold: Double, inverse: Boolean = false): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val result = Mat()
        val thresholdType = if (inverse) Imgproc.THRESH_BINARY_INV else Imgproc.THRESH_BINARY
        Imgproc.threshold(gray, result, threshold, 255.0, thresholdType)
        Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2BGR)
        return result
    }

    private fun applyAdaptiveThreshold(mat: Mat, method: String = "mean", ksize: Int = 11, c: Int = 4): Mat {
        try {
            val validKsize = if (ksize % 2 == 0) ksize + 1 else maxOf(3, ksize)
            val gray = Mat()
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
            val blurred = Mat()
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            val result = Mat()
            val adaptiveMethod = if (method == "gaussian") Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C else Imgproc.ADAPTIVE_THRESH_MEAN_C
            Imgproc.adaptiveThreshold(blurred, result, 255.0, adaptiveMethod, Imgproc.THRESH_BINARY, validKsize, c.toDouble())
            Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2BGR)
            return result
        } catch (e: Exception) {
            Log.e("OpenCV", "Lỗi trong Adaptive Thresholding: ${e.message}")
            return mat.clone()
        }
    }

    private fun applyOtsuThreshold(mat: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        val result = Mat()
        Imgproc.threshold(blurred, result, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
        Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2BGR)
        return result
    }

    private fun applySobelFilter(mat: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val sobelX = Mat()
        val sobelY = Mat()
        Imgproc.Sobel(gray, sobelX, CvType.CV_64F, 1, 0)
        Imgproc.Sobel(gray, sobelY, CvType.CV_64F, 0, 1)
        Core.absdiff(sobelX, Mat.zeros(sobelX.size(), sobelX.type()), sobelX)
        Core.absdiff(sobelY, Mat.zeros(sobelY.size(), sobelY.type()), sobelY)
        sobelX.convertTo(sobelX, CvType.CV_8U)
        sobelY.convertTo(sobelY, CvType.CV_8U)
        val sobelCombined = Mat()
        Core.bitwise_or(sobelX, sobelY, sobelCombined)
        Imgproc.cvtColor(sobelCombined, sobelCombined, Imgproc.COLOR_GRAY2BGR)
        return sobelCombined
    }

    private fun applyLaplacianFilter(mat: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_64F)
        Core.absdiff(lap, Mat.zeros(lap.size(), lap.type()), lap)
        lap.convertTo(lap, CvType.CV_8U)
        Imgproc.cvtColor(lap, lap, Imgproc.COLOR_GRAY2BGR)
        return lap
    }

    private fun applyCannyEdgeDetection(mat: Mat, threshold1: Double = 30.0, threshold2: Double = 150.0): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        val canny = Mat()
        Imgproc.Canny(blurred, canny, threshold1, threshold2)
        Imgproc.cvtColor(canny, canny, Imgproc.COLOR_GRAY2BGR)
        return canny
    }

    private fun drawContours(mat: Mat, threshold1: Double = 30.0, threshold2: Double = 150.0): Mat {
        try {
            val gray = Mat()
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
            val blurred = Mat()
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            val edged = Mat()
            Imgproc.Canny(blurred, edged, threshold1, threshold2)
            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(edged, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            Log.d("Contours", "Số lượng contours: ${contours.size}")
            val result = mat.clone()
            Imgproc.drawContours(result, contours, -1, Scalar(0.0, 255.0, 0.0), 2)
            return result
        } catch (e: Exception) {
            Log.e("OpenCV", "Lỗi trong Draw Contours: ${e.message}")
            return mat.clone()
        }
    }

    private fun numberContours(mat: Mat, threshold1: Double = 30.0, threshold2: Double = 150.0): Mat {
        try {
            val gray = Mat()
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
            val blurred = Mat()
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            val edged = Mat()
            Imgproc.Canny(blurred, edged, threshold1, threshold2)
            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(edged, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val result = mat.clone()
            contours.forEachIndexed { index, contour ->
                val moments = Imgproc.moments(contour)
                if (moments.m00 != 0.0) {
                    val cX = (moments.m10 / moments.m00).toInt()
                    val cY = (moments.m01 / moments.m00).toInt()
                    Imgproc.putText(
                        result,
                        (index + 1).toString(),
                        Point(cX.toDouble(), cY.toDouble()),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        Scalar(255.0, 0.0, 0.0),
                        2
                    )
                }
            }
            return result
        } catch (e: Exception) {
            Log.e("OpenCV", "Lỗi trong Number Contours: ${e.message}")
            return mat.clone()
        }
    }

    private fun contourAreas(mat: Mat, threshold1: Double = 150.0, threshold2: Double = 150.0): Mat {
        try {
            val gray = Mat()
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
            val blurred = Mat()
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            val edged = Mat()
            Imgproc.Canny(blurred, edged, threshold1, threshold2)
            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(edged, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val result = mat.clone()
            contours.forEachIndexed { index, contour ->
                val moments = Imgproc.moments(contour)
                if (moments.m00 != 0.0) {
                    val cX = (moments.m10 / moments.m00).toInt()
                    val cY = (moments.m01 / moments.m00).toInt()
                    val area = Imgproc.contourArea(contour)
                    Imgproc.putText(
                        result,
                        "%.0f".format(area),
                        Point((cX - 20).toDouble(), cY.toDouble()),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        Scalar(0.0, 255.0, 0.0),
                        2
                    )
                }
            }
            return result
        } catch (e: Exception) {
            Log.e("OpenCV", "Lỗi trong Contour Areas: ${e.message}")
            return mat.clone()
        }
    }

    private fun drawBoundingBoxes(mat: Mat, threshold1: Double = 30.0, threshold2: Double = 150.0): Mat {
        try {
            val gray = Mat()
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
            val blurred = Mat()
            Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
            val edged = Mat()
            Imgproc.Canny(blurred, edged, threshold1, threshold2)
            val contours = ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(edged, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val result = mat.clone()
            contours.forEach { contour ->
                val rect = Imgproc.boundingRect(contour)
                Imgproc.rectangle(
                    result,
                    Point(rect.x.toDouble(), rect.y.toDouble()),
                    Point((rect.x + rect.width).toDouble(), (rect.y + rect.height).toDouble()),
                    Scalar(0.0, 255.0, 0.0),
                    2
                )
            }
            return result
        } catch (e: Exception) {
            Log.e("OpenCV", "Lỗi trong Bounding Boxes: ${e.message}")
            return mat.clone()
        }
    }

    private fun arithmeticMeanFilter(mat: Mat, ksize: Int): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val dst = Mat()
        Imgproc.blur(gray, dst, Size(ksize.toDouble(), ksize.toDouble()))
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
        return dst
    }

    private fun geometricMeanFilter(mat: Mat, ksize: Int): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val dst = Mat(gray.size(), CvType.CV_32F)
        val padded = Mat()
        val h = (ksize - 1) / 2
        Core.copyMakeBorder(gray, padded, h, h, h, h, Core.BORDER_REFLECT)
        for (i in h until gray.rows() + h) {
            for (j in h until gray.cols() + h) {
                val region = padded.submat(i - h, i + h + 1, j - h, j + h + 1)
                val logRegion = Mat()
                region.convertTo(logRegion, CvType.CV_32F)
                Core.log(logRegion, logRegion)
                val mean = Core.mean(logRegion).`val`[0]
                dst.put(i - h, j - h, exp(mean))
            }
        }
        dst.convertTo(dst, CvType.CV_8U)
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
        return dst
    }

    private fun harmonicMeanFilter(mat: Mat, ksize: Int): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val dst = Mat(gray.size(), CvType.CV_32F)
        val padded = Mat()
        val h = (ksize - 1) / 2
        Core.copyMakeBorder(gray, padded, h, h, h, h, Core.BORDER_REFLECT)
        for (i in h until gray.rows() + h) {
            for (j in h until gray.cols() + h) {
                val region = padded.submat(i - h, i + h + 1, j - h, j + h + 1)
                val floatRegion = Mat()
                region.convertTo(floatRegion, CvType.CV_32F)
                val reciprocal = Mat()
                Core.divide(1.0, floatRegion, reciprocal)
                val sum = Core.sumElems(reciprocal).`val`[0]
                dst.put(i - h, j - h, (ksize * ksize) / sum)
            }
        }
        dst.convertTo(dst, CvType.CV_8U)
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
        return dst
    }

    private fun contraharmonicMeanFilter(mat: Mat, ksize: Int, q: Double): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val dst = Mat(gray.size(), CvType.CV_32F)
        val padded = Mat()
        val h = (ksize - 1) / 2
        Core.copyMakeBorder(gray, padded, h, h, h, h, Core.BORDER_REFLECT)
        for (i in h until gray.rows() + h) {
            for (j in h until gray.cols() + h) {
                val region = padded.submat(i - h, i + h + 1, j - h, j + h + 1)
                val regionPowQ = Mat()
                val regionPowQPlus1 = Mat()
                region.convertTo(regionPowQ, CvType.CV_32F)
                regionPowQ.copyTo(regionPowQPlus1)
                Core.pow(regionPowQ, q, regionPowQ)
                Core.pow(regionPowQPlus1, q + 1, regionPowQPlus1)
                val numerator = Core.sumElems(regionPowQPlus1).`val`[0]
                val denominator = Core.sumElems(regionPowQ).`val`[0]
                dst.put(i - h, j - h, numerator / denominator)
            }
        }
        dst.convertTo(dst, CvType.CV_8U)
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR)
        return dst
    }

    private fun addGaussianNoise(mat: Mat, stdDev: Double): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val noise = Mat(gray.size(), gray.type())
        Core.randn(noise, 0.0, stdDev)
        val result = Mat()
        Core.add(gray, noise, result)
        Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2BGR)
        return result
    }

    private fun addSaltPepperNoise(mat: Mat, amount: Double): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val noisy = gray.clone()
        val rng = java.util.Random()
        val total = gray.rows() * gray.cols()
        val num = (total * amount).toInt()
        repeat(num / 2) {
            noisy.put(rng.nextInt(gray.rows()), rng.nextInt(gray.cols()), 0.0)
        }
        repeat(num / 2) {
            noisy.put(rng.nextInt(gray.rows()), rng.nextInt(gray.cols()), 255.0)
        }
        Imgproc.cvtColor(noisy, noisy, Imgproc.COLOR_GRAY2BGR)
        return noisy
    }

    private fun addRayleighNoise(mat: Mat, scale: Double): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val noisy = gray.clone()
        val rng = java.util.Random()
        for (i in 0 until gray.rows()) for (j in 0 until gray.cols()) {
            val r = -scale * ln(1 - rng.nextDouble())
            noisy.put(i, j, gray.get(i, j)[0] + r)
        }
        Imgproc.cvtColor(noisy, noisy, Imgproc.COLOR_GRAY2BGR)
        return noisy
    }

    private fun addErlangNoise(mat: Mat, k: Int = 2, scale: Double): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val noisy = gray.clone()
        val rng = java.util.Random()
        for (i in 0 until gray.rows()) for (j in 0 until gray.cols()) {
            var sum = 0.0
            repeat(k) {
                sum += -scale * ln(1 - rng.nextDouble())
            }
            noisy.put(i, j, gray.get(i, j)[0] + sum)
        }
        Imgproc.cvtColor(noisy, noisy, Imgproc.COLOR_GRAY2BGR)
        return noisy
    }

    private fun addExponentialNoise(mat: Mat, scale: Double): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val noisy = gray.clone()
        val rng = java.util.Random()
        for (i in 0 until gray.rows()) for (j in 0 until gray.cols()) {
            val r = -scale * ln(1 - rng.nextDouble())
            noisy.put(i, j, gray.get(i, j)[0] + r)
        }
        Imgproc.cvtColor(noisy, noisy, Imgproc.COLOR_GRAY2BGR)
        return noisy
    }

    private fun addUniformNoise(mat: Mat, a: Double, b: Double): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val noisy = gray.clone()
        val rng = java.util.Random()
        for (i in 0 until gray.rows()) for (j in 0 until gray.cols()) {
            val r = a + rng.nextDouble() * (b - a)
            noisy.put(i, j, gray.get(i, j)[0] + r)
        }
        Imgproc.cvtColor(noisy, noisy, Imgproc.COLOR_GRAY2BGR)
        return noisy
    }

    private fun createDistanceFilter(rows: Int, cols: Int, d0: Double, highpass: Boolean = false, type: String = "ideal", n: Double = 1.0): Mat {
        val filter = Mat(rows, cols, CvType.CV_32F)
        val cx = rows / 2
        val cy = cols / 2
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                val d = sqrt((i - cx).toDouble().pow(2) + (j - cy).toDouble().pow(2))
                val value = when (type) {
                    "ideal" -> if (d <= d0) 1.0 else 0.0
                    "butterworth" -> 1.0 / (1 + (d / d0).pow(2 * n))
                    "gaussian" -> exp(-(d * d) / (2 * d0 * d0))
                    else -> 1.0
                }
                val result = if (highpass) 1.0 - value else value
                filter.put(i, j, result)
            }
        }
        return filter
    }

    private fun applyFrequencyFilter(mat: Mat, filter: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val padded = Mat()
        val optimalRows = Core.getOptimalDFTSize(gray.rows() * 2)
        val optimalCols = Core.getOptimalDFTSize(gray.cols() * 2)
        Core.copyMakeBorder(gray, padded, 0, optimalRows - gray.rows(), 0, optimalCols - gray.cols(), Core.BORDER_CONSTANT, Scalar.all(0.0))
        val planes = mutableListOf(padded.clone().apply { convertTo(this, CvType.CV_32F) }, Mat.zeros(padded.size(), CvType.CV_32F))
        val complexImage = Mat()
        Core.merge(planes, complexImage)
        Core.dft(complexImage, complexImage)
        val filterComplex = mutableListOf(filter, Mat.zeros(filter.size(), CvType.CV_32F))
        val complexFilter = Mat()
        Core.merge(filterComplex, complexFilter)
        Core.mulSpectrums(complexImage, complexFilter, complexImage, 0)
        Core.idft(complexImage, complexImage, Core.DFT_SCALE or Core.DFT_REAL_OUTPUT, 0)
        val result = Mat()
        complexImage.submat(Rect(0, 0, gray.cols(), gray.rows())).convertTo(result, CvType.CV_8U)
        Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2BGR)
        return result
    }

    private fun applyHistogramEqualization(mat: Mat): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val equalized = Mat()
        Imgproc.equalizeHist(gray, equalized)
        Imgproc.cvtColor(equalized, equalized, Imgproc.COLOR_GRAY2BGR)
        return equalized
    }

    private fun applyContrastStretching(mat: Mat, r1: Double, s1: Double, r2: Double, s2: Double): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val result = Mat(gray.size(), gray.type())
        for (i in 0 until gray.rows()) {
            for (j in 0 until gray.cols()) {
                val value = gray.get(i, j)[0]
                val newValue = when {
                    value < r1 -> (s1 / r1) * value
                    value < r2 -> ((s2 - s1) / (r2 - r1)) * (value - r1) + s1
                    else -> ((255 - s2) / (255 - r2)) * (value - r2) + s2
                }
                result.put(i, j, min(255.0, max(0.0, newValue)))
            }
        }
        Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2BGR)
        return result
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    fun savePdfToDownloads(context: Context, bitmap: Bitmap) {
        val pdfDocument = PdfDocument()
        val pageInfo = PdfDocument.PageInfo.Builder(bitmap.width, bitmap.height, 1).create()
        val page = pdfDocument.startPage(pageInfo)
        page.canvas.drawBitmap(bitmap, 0f, 0f, null)
        pdfDocument.finishPage(page)

        val filename = "Image_${System.currentTimeMillis()}.pdf"

        val contentValues = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, filename)
            put(MediaStore.Downloads.MIME_TYPE, "application/pdf")
            put(MediaStore.Downloads.IS_PENDING, 1)
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, contentValues)

        uri?.let {
            resolver.openOutputStream(it)?.use { outStream ->
                pdfDocument.writeTo(outStream)
            }

            contentValues.clear()
            contentValues.put(MediaStore.Downloads.IS_PENDING, 0)
            resolver.update(uri, contentValues, null, null)

            Toast.makeText(context, "Đã lưu PDF vào Download/", Toast.LENGTH_LONG).show()
        }

        pdfDocument.close()
    }

    private fun scanDocument(src: Mat): Mat {
        val img3 = Mat()
        when (src.channels()) {
            4 -> Imgproc.cvtColor(src, img3, Imgproc.COLOR_RGBA2BGR)
            1 -> Imgproc.cvtColor(src, img3, Imgproc.COLOR_GRAY2BGR)
            else -> src.copyTo(img3)
        }

        val limit = 1080.0
        val maxDim = max(img3.rows(), img3.cols()).toDouble()
        val img = if (maxDim > limit) {
            Mat().also { Imgproc.resize(img3, it, Size(), limit / maxDim, limit / maxDim, Imgproc.INTER_LINEAR) }
        } else {
            img3.clone()
        }
        val orig = img.clone()

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.morphologyEx(img, img, Imgproc.MORPH_CLOSE, kernel, Point(-1.0, -1.0), 3)

        val mask = Mat.zeros(img.size(), CvType.CV_8UC1)
        val bgdModel = Mat.zeros(1, 65, CvType.CV_64F)
        val fgdModel = Mat.zeros(1, 65, CvType.CV_64F)
        val rect = Rect(20, 20, img.cols() - 40, img.rows() - 40)
        Imgproc.grabCut(img, mask, rect, bgdModel, fgdModel, 5, Imgproc.GC_INIT_WITH_RECT)

        val mask2 = Mat()
        Core.compare(mask, Scalar(Imgproc.GC_PR_FGD.toDouble()), mask2, Core.CMP_EQ)
        val fg = Mat(img.size(), img.type(), Scalar(0.0, 0.0, 0.0))
        img.copyTo(fg, mask2)

        val gray = Mat().also { Imgproc.cvtColor(fg, it, Imgproc.COLOR_BGR2GRAY) }
        Imgproc.GaussianBlur(gray, gray, Size(11.0, 11.0), 0.0)
        val edges = Mat()
        Imgproc.Canny(gray, edges, 50.0, 200.0)
        Imgproc.dilate(edges, edges, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0)))

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(edges, contours, Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        val top5 = contours.sortedByDescending { Imgproc.contourArea(it) }.take(5)
        var quad: List<Point>? = null
        for (c in top5) {
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(
                MatOfPoint2f(*c.toArray()),
                approx,
                0.02 * Imgproc.arcLength(MatOfPoint2f(*c.toArray()), true),
                true
            )
            if (approx.total() == 4L) {
                quad = approx.toArray().toList()
                break
            }
        }
        if (quad == null) {
            return orig
        }

        val ordered = run {
            val sum = quad.map { it.x + it.y }
            val diff = quad.map { it.y - it.x }
            val tl = quad[sum.indexOf(sum.minOrNull()!!)]
            val br = quad[sum.indexOf(sum.maxOrNull()!!)]
            val tr = quad[diff.indexOf(diff.minOrNull()!!)]
            val bl = quad[diff.indexOf(diff.maxOrNull()!!)]
            listOf(tl, tr, br, bl)
        }

        val (tl, tr, br, bl) = ordered
        val widthA = hypot(br.x - bl.x, br.y - bl.y)
        val widthB = hypot(tr.x - tl.x, tr.y - tl.y)
        val maxWidth = max(widthA, widthB).toInt()
        val heightA = hypot(tr.x - br.x, tr.y - br.y)
        val heightB = hypot(tl.x - bl.x, tl.y - bl.y)
        val maxHeight = max(heightA, heightB).toInt()
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(maxWidth.toDouble(), 0.0),
            Point(maxWidth.toDouble(), maxHeight.toDouble()),
            Point(0.0, maxHeight.toDouble())
        )
        val srcPts = MatOfPoint2f(*ordered.toTypedArray())

        val M = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val scanned = Mat()
        Imgproc.warpPerspective(orig, scanned, M, Size(maxWidth.toDouble(), maxHeight.toDouble()))

        return scanned
    }

    private fun orderPoints(pts: List<Point>): List<Point> {
        val rect = Array(4) { Point() }
        val sum = pts.map { it.x + it.y }
        rect[0] = pts[sum.indexOf(sum.minOrNull()!!)]    // top-left
        rect[2] = pts[sum.indexOf(sum.maxOrNull()!!)]    // bottom-right

        val diff = pts.map { it.y - it.x }
        rect[1] = pts[diff.indexOf(diff.minOrNull()!!)]  // top-right
        rect[3] = pts[diff.indexOf(diff.maxOrNull()!!)]  // bottom-left

        return rect.toList()
    }

    private fun getDestinationPoints(srcPts: List<Point>): List<Point> {
        val (tl, tr, br, bl) = srcPts
        val widthA = hypot(br.x - bl.x, br.y - bl.y)
        val widthB = hypot(tr.x - tl.x, tr.y - tl.y)
        val maxWidth = max(widthA, widthB).toInt()

        val heightA = hypot(tr.x - br.x, tr.y - br.y)
        val heightB = hypot(tl.x - bl.x, tl.y - bl.y)
        val maxHeight = max(heightA, heightB).toInt()

        return orderPoints(
            listOf(
                Point(0.0, 0.0),
                Point(maxWidth.toDouble(), 0.0),
                Point(maxWidth.toDouble(), maxHeight.toDouble()),
                Point(0.0, maxHeight.toDouble())
            )
        )
    }

    private fun saveImageToGallery(bitmap: Bitmap): Boolean {
        val context = this
        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "ProcessedImage_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ImageProcessing")
        }

        try {
            val uri = context.contentResolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            )
            uri?.let {
                context.contentResolver?.openOutputStream(it)?.use { outputStream ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
                }
                return true
            }
        } catch (e: IOException) {
            Log.e("SaveImage", "Error saving image: ${e.message}")
        }
        return false
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    @OptIn(ExperimentalMaterial3Api::class)
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
                var ksize by remember { mutableStateOf(3f) }
                var qValue by remember { mutableStateOf(1.5f) }
                var d0 by remember { mutableStateOf(60.0f) }
                var nValue by remember { mutableStateOf(2.0f) }
                var thresholdValue by remember { mutableStateOf(155.0f) }
                var threshold1 by remember { mutableStateOf(30.0f) }
                var threshold2 by remember { mutableStateOf(150.0f) }
                var adaptiveKsize by remember { mutableStateOf(11f) }
                var adaptiveC by remember { mutableStateOf(4f) }
                var rotationAngle by remember { mutableStateOf(0f) }
                var cropStartX by remember { mutableStateOf(30f) }
                var cropStartY by remember { mutableStateOf(120f) }
                var cropWidth by remember { mutableStateOf(210f) }
                var cropHeight by remember { mutableStateOf(215f) }
                var showSplit by remember { mutableStateOf(false) }
                var adaptiveMethod by remember { mutableStateOf("mean") }
                var snackbarMessage by remember { mutableStateOf<String?>(null) }
                var isLoading by remember { mutableStateOf(false) } // Loading state
                val context = LocalContext.current
                var capturedImageUri by remember { mutableStateOf<Uri?>(null) }

                val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
                    if (success && capturedImageUri != null) {
                        try {
                            val inputStream = context.contentResolver.openInputStream(capturedImageUri!!)
                            val bmp = BitmapFactory.decodeStream(inputStream)
                            inputStream?.close()
                            bmp?.let {
                                bitmap = it
                                val mat = Mat()
                                Utils.bitmapToMat(it, mat)
                                matSrc = mat
                            }
                        } catch (e: Exception) {
                            Log.e("CameraCapture", "Lỗi đọc ảnh: ${e.message}")
                        }
                    }
                }
                val takePicture = {
                    val contentValues = ContentValues().apply {
                        put(MediaStore.Images.Media.DISPLAY_NAME, "Captured_${System.currentTimeMillis()}.jpg")
                        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ImageProcessing")
                    }

                    capturedImageUri = context.contentResolver.insert(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                        contentValues
                    )

                    capturedImageUri?.let { cameraLauncher.launch(it) }
                }
                val snackbarHostState = remember { SnackbarHostState() }
                val coroutineScope = rememberCoroutineScope() // Coroutine scope

                val launcher = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
                    uri?.let {
                        val inputStream = context.contentResolver.openInputStream(uri)
                        val bmp = BitmapFactory.decodeStream(inputStream)
                        inputStream?.close()
                        bmp?.let {
                            bitmap = it
                            val mat = Mat()
                            Utils.bitmapToMat(it, mat)
                            matSrc = mat
                        }
                    }
                }

                val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
                    if (isGranted) {
                        bitmap?.let {
                            if (saveImageToGallery(it)) {
                                snackbarMessage = "Image saved to gallery"
                            } else {
                                snackbarMessage = "Failed to save image"
                            }
                        }
                    } else {
                        snackbarMessage = "Storage permission denied"
                    }
                }

                LaunchedEffect(snackbarMessage) {
                    snackbarMessage?.let {
                        snackbarHostState.showSnackbar(it)
                        snackbarMessage = null
                    }
                }

                LaunchedEffect(
                    selectedFunction,
                    matSrc,
                    d0,
                    nValue,
                    rotationAngle,
                    cropStartX,
                    cropStartY,
                    cropWidth,
                    cropHeight,
                    thresholdValue,
                    threshold1,
                    threshold2,
                    adaptiveKsize,
                    adaptiveC,
                    adaptiveMethod
                ) {
                    matSrc?.let { src ->
                        if (selectedFunction == "Automatic Document Scanner") {
                            isLoading = true
                            coroutineScope.launch(Dispatchers.Default) {
                                try {
                                    val processed = scanDocument(src)
                                    val resultBitmap = matToBitmap(processed)
                                    withContext(Dispatchers.Main) {
                                        bitmap = resultBitmap
                                        isLoading = false
                                    }
                                } catch (e: Exception) {
                                    Log.e("OpenCV", "Error in scanDocument: ${e.message}")
                                    withContext(Dispatchers.Main) {
                                        isLoading = false
                                        snackbarMessage = "Error processing document"
                                    }
                                }
                            }
                        } else {
                            val processed = when (selectedFunction) {
                                "Ảnh gốc (Original Image)" -> src
                                "Simple Thresholding" -> applyGlobalThreshold(src, thresholdValue.toDouble())
                                "Simple Thresholding (Inverse)" -> applyGlobalThreshold(src, thresholdValue.toDouble(), true)
                                "Adaptive Thresholding" -> applyAdaptiveThreshold(src, adaptiveMethod, adaptiveKsize.toInt(), adaptiveC.toInt())
                                "Otsu Thresholding" -> applyOtsuThreshold(src)
                                "Sobel Filter" -> applySobelFilter(src)
                                "Laplacian Filter" -> applyLaplacianFilter(src)
                                "Canny Edge Detection" -> applyCannyEdgeDetection(src, threshold1.toDouble(), threshold2.toDouble())
                                "Draw Contours" -> drawContours(src, threshold1.toDouble(), threshold2.toDouble())
                                "Number Contours" -> numberContours(src, threshold1.toDouble(), threshold2.toDouble())
                                "Contour Areas" -> contourAreas(src, threshold1.toDouble(), threshold2.toDouble())
                                "Bounding Boxes" -> drawBoundingBoxes(src, threshold1.toDouble(), threshold2.toDouble())
                                "Ideal Lowpass Filter (ILPF)" -> applyFrequencyFilter(src, createDistanceFilter(Core.getOptimalDFTSize(src.rows() * 2), Core.getOptimalDFTSize(src.cols() * 2), d0.toDouble(), false, "ideal"))
                                "Butterworth Lowpass Filter (BLPF)" -> applyFrequencyFilter(src, createDistanceFilter(Core.getOptimalDFTSize(src.rows() * 2), Core.getOptimalDFTSize(src.cols() * 2), d0.toDouble(), false, "butterworth", nValue.toDouble()))
                                "Gaussian Lowpass Filter (GLPF)" -> applyFrequencyFilter(src, createDistanceFilter(Core.getOptimalDFTSize(src.rows() * 2), Core.getOptimalDFTSize(src.cols() * 2), d0.toDouble(), false, "gaussian"))
                                "Ideal Highpass Filter (IHPF)" -> applyFrequencyFilter(src, createDistanceFilter(Core.getOptimalDFTSize(src.rows() * 2), Core.getOptimalDFTSize(src.cols() * 2), d0.toDouble(), true, "ideal"))
                                "Butterworth Highpass Filter (BHPF)" -> applyFrequencyFilter(src, createDistanceFilter(Core.getOptimalDFTSize(src.rows() * 2), Core.getOptimalDFTSize(src.cols() * 2), d0.toDouble(), true, "butterworth", nValue.toDouble()))
                                "Gaussian Highpass Filter (GHPF)" -> applyFrequencyFilter(src, createDistanceFilter(Core.getOptimalDFTSize(src.rows() * 2), Core.getOptimalDFTSize(src.cols() * 2), d0.toDouble(), true, "gaussian"))
                                "Gaussian Noise" -> addGaussianNoise(src, d0.toDouble())
                                "Salt & Pepper Noise" -> addSaltPepperNoise(src, d0 / 100.0)
                                "Rayleigh Noise" -> addRayleighNoise(src, d0.toDouble())
                                "Erlang Noise" -> addErlangNoise(src, 2, d0.toDouble())
                                "Exponential Noise" -> addExponentialNoise(src, d0.toDouble())
                                "Uniform Noise" -> addUniformNoise(src, 0.0, d0.toDouble())
                                "Arithmetic Mean Filter" -> arithmeticMeanFilter(src, ksize.toInt())
                                "Geometric Mean Filter" -> geometricMeanFilter(src, ksize.toInt())
                                "Harmonic Mean Filter" -> harmonicMeanFilter(src, ksize.toInt())
                                "Contraharmonic Mean Filter" -> contraharmonicMeanFilter(src, ksize.toInt(), qValue.toDouble())
                                "Histogram Equalization" -> applyHistogramEqualization(src)
                                "Contrast Stretching" -> applyContrastStretching(src, threshold1.toDouble(), thresholdValue.toDouble(), threshold2.toDouble(), adaptiveC * 255.0)
                                "Dịch ảnh (Translation)" -> translateImage(src)
                                "Xoay ảnh (Rotation)" -> rotateImage(src, rotationAngle.toDouble())
                                "Co dãn ảnh (Scaling)" -> scaleImage(src)
                                "Lật ảnh (Flipping)" -> flipImage(src)
                                "Cắt ảnh (Cropping)" -> cropImage(src, cropStartX.toInt(), cropStartY.toInt(), cropWidth.toInt(), cropHeight.toInt())
                                "Tách kênh màu (Split Channels)" -> src
                                else -> src
                            }
                            bitmap = matToBitmap(processed)
                            showSplit = selectedFunction == "Tách kênh màu (Split Channels)"
                        }
                    }
                }

                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    topBar = {
                        TopAppBar(
                            title = {
                                Text(
                                    "Image Processing Studio",
                                    fontSize = 20.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = MaterialTheme.colorScheme.onPrimary
                                )
                            },
                            colors = TopAppBarDefaults.topAppBarColors(
                                containerColor = MaterialTheme.colorScheme.primary,
                                titleContentColor = MaterialTheme.colorScheme.onPrimary
                            )
                        )
                    },
                    floatingActionButton = {
                        Row(
                            modifier = Modifier
                                .padding(end = 16.dp, bottom = 16.dp),
                            horizontalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            FloatingActionButton(
                                onClick = { launcher.launch("image/*") },
                                containerColor = MaterialTheme.colorScheme.secondary
                            ) {
                                Text("Chọn ảnh")
                            }

                            FloatingActionButton(
                                onClick = { takePicture() },
                                containerColor = MaterialTheme.colorScheme.tertiary
                            ) {
                                Text("Chụp")
                            }
                            FloatingActionButton(
                                onClick = {
                                    bitmap?.let { savePdfToDownloads(context, it) }
                                },
                                containerColor = MaterialTheme.colorScheme.primary
                            ) {
                                Text("Xuất PDF")
                            }
                        }
                    },
                    snackbarHost = { SnackbarHost(snackbarHostState) }
                ) { padding ->
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(padding)
                            .padding(horizontal = 16.dp)
                            .verticalScroll(rememberScrollState())
                            .background(MaterialTheme.colorScheme.background),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp),
                            elevation = CardDefaults.cardElevation(4.dp),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Column(
                                modifier = Modifier.padding(16.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                Text(
                                    "OpenCV Version: ${Core.VERSION}",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = MaterialTheme.colorScheme.onSurface
                                )

                                val options = listOf(
                                    "Ảnh gốc (Original Image)",
                                    "Simple Thresholding",
                                    "Simple Thresholding (Inverse)",
                                    "Adaptive Thresholding",
                                    "Otsu Thresholding",
                                    "Sobel Filter",
                                    "Laplacian Filter",
                                    "Canny Edge Detection",
                                    "Draw Contours",
                                    "Number Contours",
                                    "Contour Areas",
                                    "Bounding Boxes",
                                    "Ideal Lowpass Filter (ILPF)",
                                    "Butterworth Lowpass Filter (BLPF)",
                                    "Gaussian Lowpass Filter (GLPF)",
                                    "Ideal Highpass Filter (IHPF)",
                                    "Butterworth Highpass Filter (BHPF)",
                                    "Gaussian Highpass Filter (GHPF)",
                                    "Gaussian Noise",
                                    "Salt & Pepper Noise",
                                    "Rayleigh Noise",
                                    "Erlang Noise",
                                    "Exponential Noise",
                                    "Uniform Noise",
                                    "Arithmetic Mean Filter",
                                    "Geometric Mean Filter",
                                    "Harmonic Mean Filter",
                                    "Contraharmonic Mean Filter",
                                    "Histogram Equalization",
                                    "Contrast Stretching",
                                    "Dịch ảnh (Translation)",
                                    "Xoay ảnh (Rotation)",
                                    "Co dãn ảnh (Scaling)",
                                    "Lật ảnh (Flipping)",
                                    "Cắt ảnh (Cropping)",
                                    "Tách kênh màu (Split Channels)",
                                    "Automatic Document Scanner"
                                )

                                var expanded by remember { mutableStateOf(false) }
                                ExposedDropdownMenuBox(
                                    expanded = expanded,
                                    onExpandedChange = { expanded = !expanded }
                                ) {
                                    OutlinedTextField(
                                        value = selectedFunction,
                                        onValueChange = {},
                                        label = { Text("Kỹ thuật xử lý") },
                                        trailingIcon = {
                                            ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded)
                                        },
                                        readOnly = true,
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .menuAnchor()
                                            .clip(RoundedCornerShape(8.dp))
                                    )
                                    ExposedDropdownMenu(
                                        expanded = expanded,
                                        onDismissRequest = { expanded = false },
                                        modifier = Modifier
                                            .background(MaterialTheme.colorScheme.surface)
                                            .animateContentSize(animationSpec = tween(300))
                                    ) {
                                        options.forEach { option ->
                                            DropdownMenuItem(
                                                text = { Text(option, style = MaterialTheme.typography.bodyMedium) },
                                                onClick = {
                                                    selectedFunction = option
                                                    expanded = false
                                                },
                                                modifier = Modifier
                                                    .fillMaxWidth()
                                                    .padding(horizontal = 8.dp)
                                            )
                                        }
                                    }
                                }

                                if (bitmap != null) {
                                    Button(
                                        onClick = {
                                            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
                                                permissionLauncher.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                                            } else {
                                                if (saveImageToGallery(bitmap!!)) {
                                                    snackbarMessage = "Image saved to gallery"
                                                } else {
                                                    snackbarMessage = "Failed to save image"
                                                }
                                            }
                                        },
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(48.dp),
                                        colors = ButtonDefaults.buttonColors(
                                            containerColor = MaterialTheme.colorScheme.primary
                                        )
                                    ) {
                                        Text("Lưu ảnh", fontWeight = FontWeight.Medium)
                                    }
                                }
                            }
                        }

                        if (bitmap != null) {
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 8.dp),
                                elevation = CardDefaults.cardElevation(4.dp),
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Column(
                                    modifier = Modifier.padding(16.dp),
                                    verticalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    if (showSplit && matSrc != null) {
                                        val images = splitChannels(matSrc!!)
                                        images.forEachIndexed { index, img ->
                                            Column {
                                                Text(
                                                    when (index) {
                                                        0 -> "Red Channel"
                                                        1 -> "Green Channel"
                                                        2 -> "Blue Channel"
                                                        else -> ""
                                                    },
                                                    style = MaterialTheme.typography.titleSmall,
                                                    color = MaterialTheme.colorScheme.primary
                                                )
                                                Spacer(modifier = Modifier.height(4.dp))
                                                Image(
                                                    bitmap = img.asImageBitmap(),
                                                    contentDescription = null,
                                                    modifier = Modifier
                                                        .fillMaxWidth()
                                                        .height(150.dp)
                                                        .clip(RoundedCornerShape(8.dp))
                                                        .background(Color.Black.copy(alpha = 0.1f))
                                                )
                                            }
                                        }
                                    } else {
                                        Box(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .height(400.dp)
                                                .clip(RoundedCornerShape(8.dp))
                                                .background(Color.Black.copy(alpha = 0.1f)),
                                            contentAlignment = Alignment.Center
                                        ) {
                                            if (isLoading) {
                                                CircularProgressIndicator(
                                                    color = MaterialTheme.colorScheme.primary,
                                                    modifier = Modifier.size(48.dp)
                                                )
                                            } else {
                                                bitmap?.let {
                                                    Image(
                                                        bitmap = it.asImageBitmap(),
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

                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 8.dp),
                                elevation = CardDefaults.cardElevation(4.dp),
                                shape = RoundedCornerShape(12.dp)
                            ) {
                                Column(
                                    modifier = Modifier.padding(16.dp),
                                    verticalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    if (selectedFunction == "Xoay ảnh (Rotation)") {
                                        SliderWithLabel(
                                            label = "Góc xoay: ${rotationAngle.toInt()}°",
                                            value = rotationAngle,
                                            onValueChange = { rotationAngle = it },
                                            valueRange = 0f..180f
                                        )
                                    }

                                    if (selectedFunction == "Cắt ảnh (Cropping)") {
                                        SliderWithLabel(
                                            label = "Start X: ${cropStartX.toInt()}",
                                            value = cropStartX,
                                            onValueChange = { cropStartX = it },
                                            valueRange = 0f..(matSrc?.cols()?.toFloat() ?: 100f)
                                        )
                                        SliderWithLabel(
                                            label = "Start Y: ${cropStartY.toInt()}",
                                            value = cropStartY,
                                            onValueChange = { cropStartY = it },
                                            valueRange = 0f..(matSrc?.rows()?.toFloat() ?: 100f)
                                        )
                                        SliderWithLabel(
                                            label = "Width: ${cropWidth.toInt()}",
                                            value = cropWidth,
                                            onValueChange = { cropWidth = it },
                                            valueRange = 1f..(matSrc?.cols()?.toFloat() ?: 100f)
                                        )
                                        SliderWithLabel(
                                            label = "Height: ${cropHeight.toInt()}",
                                            value = cropHeight,
                                            onValueChange = { cropHeight = it },
                                            valueRange = 50f..(matSrc?.rows()?.toFloat() ?: 50f)
                                        )
                                    }

                                    if (selectedFunction == "Adaptive Thresholding") {
                                        Text(
                                            "Phương pháp: $adaptiveMethod",
                                            style = MaterialTheme.typography.titleSmall,
                                            color = MaterialTheme.colorScheme.primary
                                        )
                                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                            Button(
                                                onClick = { adaptiveMethod = "mean" },
                                                modifier = Modifier.weight(1f),
                                                colors = ButtonDefaults.buttonColors(
                                                    containerColor = if (adaptiveMethod == "mean") MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary
                                                )
                                            ) {
                                                Text("Mean")
                                            }
                                            Button(
                                                onClick = { adaptiveMethod = "gaussian" },
                                                modifier = Modifier.weight(1f),
                                                colors = ButtonDefaults.buttonColors(
                                                    containerColor = if (adaptiveMethod == "gaussian") MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondary
                                                )
                                            ) {
                                                Text("Gaussian")
                                            }
                                        }
                                        SliderWithLabel(
                                            label = "Kích thước vùng: ${adaptiveKsize.toInt()}",
                                            value = adaptiveKsize,
                                            onValueChange = { adaptiveKsize = (it.toInt() / 2 * 2 + 1).toFloat() },
                                            valueRange = 3f..21f,
                                            steps = 8
                                        )
                                        SliderWithLabel(
                                            label = "Hằng số C: ${adaptiveC.toInt()}",
                                            value = adaptiveC,
                                            onValueChange = { adaptiveC = it },
                                            valueRange = 0f..10f
                                        )
                                    }

                                    if (selectedFunction == "Simple Thresholding" || selectedFunction == "Simple Thresholding (Inverse)") {
                                        SliderWithLabel(
                                            label = "Ngưỡng T: ${thresholdValue.toInt()}",
                                            value = thresholdValue,
                                            onValueChange = { thresholdValue = it },
                                            valueRange = 0f..255f
                                        )
                                    }

                                    if (selectedFunction == "Canny Edge Detection" || selectedFunction == "Draw Contours" || selectedFunction == "Number Contours" || selectedFunction == "Contour Areas" || selectedFunction == "Bounding Boxes") {
                                        SliderWithLabel(
                                            label = "Threshold 1: ${threshold1.toInt()}",
                                            value = threshold1,
                                            onValueChange = { threshold1 = it },
                                            valueRange = 0f..255f
                                        )
                                        SliderWithLabel(
                                            label = "Threshold 2: ${threshold2.toInt()}",
                                            value = threshold2,
                                            onValueChange = { threshold2 = it },
                                            valueRange = 0f..255f
                                        )
                                    }

                                    if (selectedFunction == "Contrast Stretching") {
                                        SliderWithLabel(
                                            label = "r1: ${threshold1.toInt()}",
                                            value = threshold1,
                                            onValueChange = { threshold1 = it },
                                            valueRange = 0f..255f
                                        )
                                        SliderWithLabel(
                                            label = "s1: ${thresholdValue.toInt()}",
                                            value = thresholdValue,
                                            onValueChange = { thresholdValue = it },
                                            valueRange = 0f..255f
                                        )
                                        SliderWithLabel(
                                            label = "r2: ${threshold2.toInt()}",
                                            value = threshold2,
                                            onValueChange = { threshold2 = it },
                                            valueRange = 0f..255f
                                        )
                                        SliderWithLabel(
                                            label = "s2: ${(adaptiveC * 255).toInt()}",
                                            value = adaptiveC,
                                            onValueChange = { adaptiveC = it },
                                            valueRange = 0f..1f
                                        )
                                    }

                                    if (selectedFunction.contains("Filter") && !selectedFunction.contains("Lowpass") && !selectedFunction.contains("Highpass") && !selectedFunction.contains("Sobel") && !selectedFunction.contains("Laplacian") && !selectedFunction.contains("Canny")) {
                                        SliderWithLabel(
                                            label = "Kích thước bộ lọc: ${ksize.toInt()}x${ksize.toInt()}",
                                            value = ksize,
                                            onValueChange = { ksize = it },
                                            valueRange = 3f..9f,
                                            steps = 3
                                        )
                                    }

                                    if (selectedFunction.contains("Contraharmonic")) {
                                        SliderWithLabel(
                                            label = "Q (bậc lọc): ${qValue}",
                                            value = qValue,
                                            onValueChange = { qValue = it },
                                            valueRange = -2f..3f
                                        )
                                    }

                                    if (selectedFunction.contains("Lowpass") || selectedFunction.contains("Highpass")) {
                                        SliderWithLabel(
                                            label = "D₀ (Cutoff Frequency): ${d0.toInt()}",
                                            value = d0,
                                            onValueChange = { d0 = it },
                                            valueRange = 1f..200f
                                        )
                                        if (selectedFunction.contains("Butterworth")) {
                                            SliderWithLabel(
                                                label = "n (Order): ${nValue}",
                                                value = nValue,
                                                onValueChange = { nValue = it },
                                                valueRange = 1f..10f
                                            )
                                        }
                                    }

                                    if (selectedFunction.contains("Noise")) {
                                        SliderWithLabel(
                                            label = "Tham số: ${d0.toInt()}",
                                            value = d0,
                                            onValueChange = { d0 = it },
                                            valueRange = 1f..100f
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Composable
    fun SliderWithLabel(
        label: String,
        value: Float,
        onValueChange: (Float) -> Unit,
        valueRange: ClosedFloatingPointRange<Float>,
        steps: Int = 0
    ) {
        Column {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Slider(
                value = value,
                onValueChange = onValueChange,
                valueRange = valueRange,
                steps = steps,
                modifier = Modifier.fillMaxWidth(),
                colors = SliderDefaults.colors(
                    thumbColor = MaterialTheme.colorScheme.primary,
                    activeTrackColor = MaterialTheme.colorScheme.primary
                )
            )
        }
    }
}