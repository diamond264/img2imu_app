package com.example.img2imu

import android.content.Context
import android.graphics.*
import android.util.Log
import com.github.psambit9791.jdsp.transform.ShortTimeFourier
import com.github.psambit9791.jdsp.windows.Hanning
import com.github.psambit9791.jdsp.windows._Window
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.apache.commons.math3.complex.Complex
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.roundToInt
import kotlin.math.sin

class SignalProcessor(ctx: Context) {
    private var mContext: Context = ctx
    val TAG = "SignalProcessor"

    fun createSpectrogram(filePath: String): Pair<Bitmap,Long>{
        val inputStream = mContext.assets.open(filePath)
        val reader = BufferedReader(InputStreamReader(inputStream, StandardCharsets.UTF_8))
        val rawData = mutableListOf<FloatArray>()
        var line: String?
        while (reader.readLine().also { line = it } != null) {
            val values = line?.split(",")?.map { it.toFloat() }?.toFloatArray()
            if (values != null) {
                rawData.add(values)
            }
        }
        reader.close()

        val startTime = System.currentTimeMillis()

        val spectrograms = runBlocking {
            withContext(Dispatchers.Default) {
                computeSpectrogram(rawData)
            }
        }
        val elapsedTime = System.currentTimeMillis() - startTime
        val width = spectrograms[0][0].size
        val height = spectrograms[0].size

        val new_spectrograms = mutableListOf<List<FloatArray>>()
        for (list in spectrograms) {
            val values = list.flatMap { it.asList() }
            val min = values.min() ?: 0f
            val max = values.max() ?: 1f
            val range = max - min

            val result = mutableListOf<FloatArray>()
            for (array in list) {
                result.add(array.map { (it - min) / range }.toFloatArray())
            }
            new_spectrograms.add(result)
        }

        val graybitmaps = mutableListOf<Bitmap>()
        for (list in new_spectrograms){
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)
            val paint = Paint().apply {
                colorFilter = ColorMatrixColorFilter(
                    ColorMatrix().apply{setSaturation(0f)}
//                    ColorMatrix(floatArrayOf(
//                        0.33f, 0.33f, 0.33f, 0.0f, 0.0f,
//                        0.33f, 0.33f, 0.33f, 0.0f, 0.0f,
//                        0.33f, 0.33f, 0.33f, 0.0f, 0.0f,
//                        0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
//                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f
//                    ))
                )
            }
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val va = (list[y][x]*255).roundToInt()
                    paint.color = Color.rgb(va,va,va)
                    canvas.drawPoint(x.toFloat(), height-y.toFloat(), paint)
                }
            }
            graybitmaps.add(bitmap)
        }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = Color.red(graybitmaps[0].getPixel(x, y))
                val g = Color.green(graybitmaps[1].getPixel(x, y))
                val b = Color.blue(graybitmaps[2].getPixel(x, y))
                bitmap.setPixel(x, y, Color.rgb(r, g, b))
            }
        }
//        val elapsedTime = System.currentTimeMillis() - startTime
//        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
//        val canvas = Canvas(bitmap)
//        val paint = Paint().apply { color = Color.BLACK }
//        for (y in 0 until height) {
//            for (x in 0 until width) {
//                val r = new_spectrograms[0][y][x]
//                val g = new_spectrograms[1][y][x]
//                val b = new_spectrograms[2][y][x]
//                paint.color = Color.rgb((r * 255).roundToInt(), (g * 255).roundToInt(), (b * 255).roundToInt())
//                canvas.drawPoint(x.toFloat(), height-y.toFloat(), paint)
//            }
//        }
        return Pair(bitmap,elapsedTime)
    }

    fun getPSD(complexArray: Array<Array<Complex>>): Array<DoubleArray> {
        val result = Array(complexArray.size) {
            DoubleArray(
                complexArray[0].size
            )
        }

        // Fill in the output
        for (c in complexArray[0].indices) {
            for (r in complexArray.indices) {
                result[r][c] = complexArray[r][c].real*complexArray[r][c].real+
                        complexArray[r][c].imaginary*complexArray[r][c].imaginary
            }
        }
        return result
    }

    suspend fun computeSpectrogram(rawData: List<FloatArray>): List<List<FloatArray>> {
        return withContext(Dispatchers.Default) {
            val samplingFrequency = 64f
            val nfft = 128
            val noverlap = 116
            val spectrograms = mutableListOf<List<FloatArray>>()
            for (i in 0 until 3) {
//                Log.d(TAG, rawData.size.toString())
//                val spectrogram = stft(rawData.map{it[i]}, nfft, nfft-noverlap)
                val tmpdata = rawData.map{it[i]}
//                Log.d(TAG, tmpdata.size.toString())
                val window: _Window = Hanning(nfft)
                val STFT = ShortTimeFourier(
                    DoubleArray(tmpdata.size) { tmpdata[it].toDouble() },
                    nfft, noverlap, nfft,
                    window, samplingFrequency.toDouble())
                STFT.transform()
//                val spectrogram = getPSD(STFT.getComplex(true))
//                val spectrogram = STFT.getMagnitude(true).toList()
                var spectrogram = STFT.spectrogram(true).toList()

//                var sum = 0f
//                for (w in window.window) {
//                    val k = w.toFloat()
//                    sum += k*k
//                }

                spectrograms.add(spectrogram.map { it.map { log10(it).toFloat() }.toFloatArray() })
//                spectrograms.add(spectrogram.map { it.map {
//                            10*Math.log10((it.toFloat()*2/samplingFrequency/sum).toDouble()).toFloat()
//                            }.toFloatArray() })
//                spectrograms.add(spectrogram.map { it.map { it.toFloat() }.toFloatArray() })
            }
            spectrograms
        }
    }

    // make own stft function
    fun stft(signal: List<Float>, windowSize: Int, hopSize: Int): List<FloatArray> {
        val n = signal.size
        val window = FloatArray(windowSize) { i -> (0.54f - 0.46f * cos(2f * Math.PI.toFloat() * i / windowSize)).toFloat() }
        val numFrames = (n - windowSize) / hopSize + 1
        Log.d(TAG, n.toString())
        val result = MutableList(numFrames) { FloatArray(windowSize) }

        for (frameIndex in 0 until numFrames) {
            val start = frameIndex * hopSize
            val end = start + windowSize
            for (i in start until end) {
                result[frameIndex][i - start] = signal[i] * window[i - start]
            }

            val fft = FloatArray(windowSize) { i ->
                result[frameIndex][i].let { re ->
                    result[frameIndex][i].let { im ->
                        re * cos(2f * Math.PI.toFloat() * i / windowSize).toFloat() + im * sin(2f * Math.PI.toFloat() * i / windowSize).toFloat()
                    }
                }
            }

            result[frameIndex] = fft
        }

        return result
    }
}