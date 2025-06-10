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

    // Simple Thresholding (dựa trên trang 2-3 của tài liệu TH#34)
    private fun applyGlobalThreshold(mat: Mat, threshold: Double, inverse: Boolean = false): Mat {
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        val result = Mat()
        val thresholdType = if (inverse) Imgproc.THRESH_BINARY_INV else Imgproc.THRESH_BINARY
        Imgproc.threshold(gray, result, threshold, 255.0, thresholdType)
        Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2BGR)
        return result
    }

    // Adaptive Thresholding (sửa lỗi dựa trên trang 4-5 của tài liệu TH#34)
    private fun applyAdaptiveThreshold(mat: Mat, method: String = "mean", ksize: Int = 11, c: Int = 4): Mat {
        try {
            // Đảm bảo ksize là số lẻ và >= 3
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
            // Trả về ảnh gốc nếu có lỗi
            return mat.clone()
        }
    }

    // Otsu Thresholding (dựa trên trang 5-6 của tài liệu TH#34)
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

    // Sobel Filter (dựa trên trang 7-8 của tài liệu TH#34)
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

    // Laplacian Filter (dựa trên trang 7-8 của tài liệu TH#34)
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

    // Canny Edge Detection (dựa trên trang 8-9 của tài liệu TH#34)
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

    // Draw Contours (dựa trên trang 2-4 của tài liệu TH#05)
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

    // Number Contours (dựa trên trang 5-6 của tài liệu TH#05)
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
                        Scalar(255.0, 0.0, 0.0), // Blue color
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

    // Contour Areas (dựa trên trang 6-8 của tài liệu TH#05)
    private fun contourAreas(mat: Mat, threshold1: Double = 30.0, threshold2: Double = 150.0): Mat {
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
                        Scalar(0.0, 255.0, 0.0), // Green color
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

    // Bounding Boxes (dựa trên trang 9-11 của tài liệu TH#05)
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
                    Scalar(0.0, 255.0, 0.0), // Green color
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
                val region = padded.submat(i - h, i + h + 1, j - h, j + h + 1) // Xóa dấu ) thừa
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        if (!OpenCVLoader.initDebug()) Log.e("OpenCV", "Failed to load OpenCV")
        else Log.d("OpenCV", "OpenCV loaded successfully. Version: ${Core.VERSION}")

        setContent {
            Img_Prosses_FinalTestTheme {
                var selectedFunction by remember { mutableStateOf("Ảnh gốc (Original Image)") }
                var bitmap by remember { mutableStateOf<Bitmap?>(null) }
                var matSrc by remember { mutableStateOf<Mat?>(null) }
                var ksize by remember { mutableStateOf(3f) }
                var qValue by remember { mutableStateOf(1.5f) }
                var d0 by remember { mutableStateOf(60.0f) }
                var nValue by remember { mutableStateOf(2.0f) }
                var thresholdValue by remember { mutableStateOf(155.0f) } // Mặc định T=155 từ TH#34
                var threshold1 by remember { mutableStateOf(30.0f) } // Mặc định cho Canny
                var threshold2 by remember { mutableStateOf(150.0f) } // Mặc định cho Canny
                var adaptiveKsize by remember { mutableStateOf(11f) } // Mặc định ksize=11 từ TH#34
                var adaptiveC by remember { mutableStateOf(4f) } // Mặc định C=4 từ TH#34
                var rotationAngle by remember { mutableStateOf(0f) }
                var cropStartX by remember { mutableStateOf(30f) }
                var cropStartY by remember { mutableStateOf(120f) }
                var cropWidth by remember { mutableStateOf(210f) }
                var cropHeight by remember { mutableStateOf(215f) }
                var showSplit by remember { mutableStateOf(false) }
                var adaptiveMethod by remember { mutableStateOf("mean") }

                val context = LocalContext.current

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

                LaunchedEffect(selectedFunction, matSrc, d0, nValue, rotationAngle, cropStartX, cropStartY, cropWidth, cropHeight, thresholdValue, threshold1, threshold2, adaptiveKsize, adaptiveC, adaptiveMethod) {
                    matSrc?.let {
                        val processed = when (selectedFunction) {
                            "Ảnh gốc (Original Image)" -> it
                            "Simple Thresholding" -> applyGlobalThreshold(it, thresholdValue.toDouble())
                            "Simple Thresholding (Inverse)" -> applyGlobalThreshold(it, thresholdValue.toDouble(), true)
                            "Adaptive Thresholding" -> applyAdaptiveThreshold(it, adaptiveMethod, adaptiveKsize.toInt(), adaptiveC.toInt())
                            "Otsu Thresholding" -> applyOtsuThreshold(it)
                            "Sobel Filter" -> applySobelFilter(it)
                            "Laplacian Filter" -> applyLaplacianFilter(it)
                            "Canny Edge Detection" -> applyCannyEdgeDetection(it, threshold1.toDouble(), threshold2.toDouble())
                            "Draw Contours" -> drawContours(it, threshold1.toDouble(), threshold2.toDouble()) // TH#05
                            "Number Contours" -> numberContours(it, threshold1.toDouble(), threshold2.toDouble()) // TH#05
                            "Contour Areas" -> contourAreas(it, threshold1.toDouble(), threshold2.toDouble()) // TH#05
                            "Bounding Boxes" -> drawBoundingBoxes(it, threshold1.toDouble(), threshold2.toDouble()) // TH#05
                            "Ideal Lowpass Filter (ILPF)" -> applyFrequencyFilter(it, createDistanceFilter(Core.getOptimalDFTSize(it.rows() * 2), Core.getOptimalDFTSize(it.cols() * 2), d0.toDouble(), false, "ideal"))
                            "Butterworth Lowpass Filter (BLPF)" -> applyFrequencyFilter(it, createDistanceFilter(Core.getOptimalDFTSize(it.rows() * 2), Core.getOptimalDFTSize(it.cols() * 2), d0.toDouble(), false, "butterworth", nValue.toDouble()))
                            "Gaussian Lowpass Filter (GLPF)" -> applyFrequencyFilter(it, createDistanceFilter(Core.getOptimalDFTSize(it.rows() * 2), Core.getOptimalDFTSize(it.cols() * 2), d0.toDouble(), false, "gaussian"))
                            "Ideal Highpass Filter (IHPF)" -> applyFrequencyFilter(it, createDistanceFilter(Core.getOptimalDFTSize(it.rows() * 2), Core.getOptimalDFTSize(it.cols() * 2), d0.toDouble(), true, "ideal"))
                            "Butterworth Highpass Filter (BHPF)" -> applyFrequencyFilter(it, createDistanceFilter(Core.getOptimalDFTSize(it.rows() * 2), Core.getOptimalDFTSize(it.cols() * 2), d0.toDouble(), true, "butterworth", nValue.toDouble()))
                            "Gaussian Highpass Filter (GHPF)" -> applyFrequencyFilter(it, createDistanceFilter(Core.getOptimalDFTSize(it.rows() * 2), Core.getOptimalDFTSize(it.cols() * 2), d0.toDouble(), true, "gaussian"))
                            "Gaussian Noise" -> addGaussianNoise(it, d0.toDouble())
                            "Salt & Pepper Noise" -> addSaltPepperNoise(it, d0 / 100.0)
                            "Rayleigh Noise" -> addRayleighNoise(it, d0.toDouble())
                            "Erlang Noise" -> addErlangNoise(it, 2, d0.toDouble())
                            "Exponential Noise" -> addExponentialNoise(it, d0.toDouble())
                            "Uniform Noise" -> addUniformNoise(it, 0.0, d0.toDouble())
                            "Arithmetic Mean Filter" -> arithmeticMeanFilter(it, ksize.toInt())
                            "Geometric Mean Filter" -> geometricMeanFilter(it, ksize.toInt())
                            "Harmonic Mean Filter" -> harmonicMeanFilter(it, ksize.toInt())
                            "Contraharmonic Mean Filter" -> contraharmonicMeanFilter(it, ksize.toInt(), qValue.toDouble())
                            "Histogram Equalization" -> applyHistogramEqualization(it)
                            "Contrast Stretching" -> applyContrastStretching(it, threshold1.toDouble(), thresholdValue.toDouble(), threshold2.toDouble(), adaptiveC * 255.0)
                            "Dịch ảnh (Translation)" -> translateImage(it)
                            "Xoay ảnh (Rotation)" -> rotateImage(it, rotationAngle.toDouble())
                            "Co dãn ảnh (Scaling)" -> scaleImage(it)
                            "Lật ảnh (Flipping)" -> flipImage(it)
                            "Cắt ảnh (Cropping)" -> cropImage(it, cropStartX.toInt(), cropStartY.toInt(), cropWidth.toInt(), cropHeight.toInt())
                            "Tách kênh màu (Split Channels)" -> it
                            else -> it
                        }
                        bitmap = matToBitmap(processed)
                        showSplit = selectedFunction == "Tách kênh màu (Split Channels)"
                    }
                }

                Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
                    Column(modifier = Modifier.padding(padding).padding(16.dp)) {
                        Text("OpenCV Version: ${Core.VERSION}", style = MaterialTheme.typography.titleMedium)
                        Spacer(modifier = Modifier.height(10.dp))

                        Button(onClick = { launcher.launch("image/*") }) {
                            Text("Chọn ảnh từ thư viện")
                        }

                        Spacer(modifier = Modifier.height(20.dp))

                        if (bitmap != null) {
                            val options = listOf(
                                "Ảnh gốc (Original Image)",
                                "Simple Thresholding",
                                "Simple Thresholding (Inverse)",
                                "Adaptive Thresholding",
                                "Otsu Thresholding",
                                "Sobel Filter",
                                "Laplacian Filter",
                                "Canny Edge Detection",
                                "Draw Contours", // TH#05
                                "Number Contours", // TH#05
                                "Contour Areas", // TH#05
                                "Bounding Boxes", // TH#05
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
                                "Tách kênh màu (Split Channels)"
                            )

                            var expanded by remember { mutableStateOf(false) }
                            Box {
                                Text("Kỹ thuật xử lý: $selectedFunction", modifier = Modifier.fillMaxWidth().clickable { expanded = true }.padding(12.dp))
                                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                                    options.forEach { option ->
                                        DropdownMenuItem(text = { Text(option) }, onClick = {
                                            selectedFunction = option
                                            expanded = false
                                        })
                                    }
                                }
                            }

                            if (selectedFunction == "Xoay ảnh (Rotation)") {
                                Text("Góc xoay: ${rotationAngle.toInt()}°")
                                Slider(value = rotationAngle, onValueChange = { rotationAngle = it }, valueRange = 0f..180f)
                            }

                            if (selectedFunction == "Cắt ảnh (Cropping)") {
                                Text("Start X: ${cropStartX.toInt()}")
                                Slider(value = cropStartX, onValueChange = { cropStartX = it }, valueRange = 0f..(matSrc?.cols()?.toFloat() ?: 100f))
                                Text("Start Y: ${cropStartY.toInt()}")
                                Slider(value = cropStartY, onValueChange = { cropStartY = it }, valueRange = 0f..(matSrc?.rows()?.toFloat() ?: 100f))
                                Text("Width: ${cropWidth.toInt()}")
                                Slider(value = cropWidth, onValueChange = { cropWidth = it }, valueRange = 1f..(matSrc?.cols()?.toFloat() ?: 100f))
                                Text("Height: ${cropHeight.toInt()}")
                                Slider(value = cropHeight, onValueChange = { cropHeight = it }, valueRange = 1f..(matSrc?.rows()?.toFloat() ?: 100f))
                            }

                            if (selectedFunction == "Adaptive Thresholding") {
                                Text("Phương pháp: $adaptiveMethod")
                                Row {
                                    Button(onClick = { adaptiveMethod = "mean" }) { Text("Mean") }
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Button(onClick = { adaptiveMethod = "gaussian" }) { Text("Gaussian") }
                                }
                                Text("Kích thước vùng: ${adaptiveKsize.toInt()}")
                                Slider(
                                    value = adaptiveKsize,
                                    onValueChange = { adaptiveKsize = (it.toInt() / 2 * 2 + 1).toFloat() }, // Đảm bảo số lẻ
                                    valueRange = 3f..21f,
                                    steps = 8 // Các giá trị: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
                                )
                                Text("Hằng số C: ${adaptiveC.toInt()}")
                                Slider(value = adaptiveC, onValueChange = { adaptiveC = it }, valueRange = 0f..10f)
                            }

                            if (showSplit && matSrc != null) {
                                val images = splitChannels(matSrc!!)
                                images.forEachIndexed { index, img ->
                                    Text(when (index) {
                                        0 -> "Red Channel"
                                        1 -> "Green Channel"
                                        2 -> "Blue Channel"
                                        else -> ""
                                    })
                                    Image(bitmap = img.asImageBitmap(), contentDescription = null, modifier = Modifier.fillMaxWidth().height(150.dp))
                                    Spacer(modifier = Modifier.height(8.dp))
                                }
                            } else if (bitmap != null) {
                                Image(bitmap = bitmap!!.asImageBitmap(), contentDescription = null, modifier = Modifier.fillMaxWidth().height(400.dp))
                            }

                            if (selectedFunction == "Simple Thresholding" || selectedFunction == "Simple Thresholding (Inverse)") {
                                Text("Ngưỡng T: ${thresholdValue.toInt()}")
                                Slider(value = thresholdValue, onValueChange = { thresholdValue = it }, valueRange = 0f..255f)
                            }

                            if (selectedFunction == "Canny Edge Detection" || selectedFunction == "Draw Contours" || selectedFunction == "Number Contours" || selectedFunction == "Contour Areas" || selectedFunction == "Bounding Boxes") {
                                Text("Threshold 1: ${threshold1.toInt()}")
                                Slider(value = threshold1, onValueChange = { threshold1 = it }, valueRange = 0f..255f)
                                Text("Threshold 2: ${threshold2.toInt()}")
                                Slider(value = threshold2, onValueChange = { threshold2 = it }, valueRange = 0f..255f)
                            }

                            if (selectedFunction == "Contrast Stretching") {
                                Text("r1: ${threshold1.toInt()}")
                                Slider(value = threshold1, onValueChange = { threshold1 = it }, valueRange = 0f..255f)
                                Text("s1: ${thresholdValue.toInt()}")
                                Slider(value = thresholdValue, onValueChange = { thresholdValue = it }, valueRange = 0f..255f)
                                Text("r2: ${threshold2.toInt()}")
                                Slider(value = threshold2, onValueChange = { threshold2 = it }, valueRange = 0f..255f)
                                Text("s2: ${(adaptiveC * 255).toInt()}")
                                Slider(value = adaptiveC, onValueChange = { adaptiveC = it }, valueRange = 0f..1f)
                            }

                            if (selectedFunction.contains("Filter") && !selectedFunction.contains("Lowpass") && !selectedFunction.contains("Highpass") && !selectedFunction.contains("Sobel") && !selectedFunction.contains("Laplacian") && !selectedFunction.contains("Canny")) {
                                Text("Kích thước bộ lọc: ${ksize.toInt()}x${ksize.toInt()}")
                                Slider(value = ksize, onValueChange = { ksize = it }, valueRange = 3f..9f, steps = 3)
                            }

                            if (selectedFunction.contains("Contraharmonic")) {
                                Text("Q (bậc lọc): ${qValue}")
                                Slider(value = qValue, onValueChange = { qValue = it }, valueRange = -2f..3f)
                            }

                            if (selectedFunction.contains("Lowpass") || selectedFunction.contains("Highpass")) {
                                Text("D₀ (Cutoff Frequency): ${d0.toInt()}")
                                Slider(value = d0, onValueChange = { d0 = it }, valueRange = 1f..200f)
                                if (selectedFunction.contains("Butterworth")) {
                                    Text("n (Order): ${nValue}")
                                    Slider(value = nValue, onValueChange = { nValue = it }, valueRange = 1f..10f)
                                }
                            }

                            if (selectedFunction.contains("Noise")) {
                                Text("Tham số: ${d0.toInt()}")
                                Slider(value = d0, onValueChange = { d0 = it }, valueRange = 1f..100f)
                            }
                        }
                    }
                }
            }
        }
    }
}