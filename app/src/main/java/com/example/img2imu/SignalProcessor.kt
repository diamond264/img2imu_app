package com.example.img2imu

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import com.github.psambit9791.jdsp.transform.ShortTimeFourier
import com.github.psambit9791.jdsp.windows.Hanning
import com.github.psambit9791.jdsp.windows._Window
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.File
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets
import kotlin.math.cos
import kotlin.math.sin

class SignalProcessor(ctx: Context) {
    private var mContext: Context = ctx
    val TAG = "SignalProcessor"

    fun createSpectrogram(filePath: String): Bitmap {
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

        val spectrograms = runBlocking {
            withContext(Dispatchers.Default) {
                computeSpectrogram(rawData)
            }
        }
        val width = spectrograms[0][0].size
        val height = spectrograms[0].size
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint().apply { color = Color.BLACK }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = spectrograms[0][y][x]
                val g = spectrograms[1][y][x]
                val b = spectrograms[2][y][x]
                paint.color = Color.rgb((r * 255).toInt(), (g * 255).toInt(), (b * 255).toInt())
                canvas.drawPoint(x.toFloat(), y.toFloat(), paint)
            }
        }
        return bitmap
    }

    suspend fun computeSpectrogram(rawData: List<FloatArray>): List<List<FloatArray>> {
        return withContext(Dispatchers.Default) {
            val samplingFrequency = 64f
            val nfft = 128
            val noverlap = 116
            val spectrograms = mutableListOf<List<FloatArray>>()
            for (i in 0 until 3) {
                Log.d(TAG, rawData.size.toString())
//                val spectrogram = stft(rawData.map{it[i]}, nfft, nfft-noverlap)
                val tmpdata = rawData.map{it[i]}
                Log.d(TAG, tmpdata.size.toString())
                val window: _Window = Hanning(nfft)
                val STFT = ShortTimeFourier(
                    DoubleArray(tmpdata.size) { tmpdata[it].toDouble() }, nfft, noverlap, nfft, window,samplingFrequency.toDouble())
                STFT.transform()
                val spectrogram = STFT.spectrogram(true).toList()

                spectrograms.add(spectrogram.map { it.map { it.toFloat() }.toFloatArray() })
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