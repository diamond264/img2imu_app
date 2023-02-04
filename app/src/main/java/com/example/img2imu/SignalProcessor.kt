package com.example.img2imu

import android.content.Context
import android.graphics.*
import com.github.psambit9791.jdsp.transform.ShortTimeFourier
import com.github.psambit9791.jdsp.windows.Hanning
import com.github.psambit9791.jdsp.windows._Window
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.apache.commons.math3.util.FastMath
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets
import kotlin.math.roundToInt

class SignalProcessor(ctx: Context) {
    private var mContext: Context = ctx
    val TAG = "SignalProcessor"

    fun createSpectrogram(filePath: String): Pair<Bitmap,Long>{
        val rawData = loadCsv(filePath)

        val startTime = System.currentTimeMillis()
        val spectrograms = runBlocking {
            withContext(Dispatchers.Default) {
                computeSpectrogram(rawData)
            }
        }
        val normalizedSpectrogram = normalizeSpectrogram(spectrograms)

        val width = spectrograms[0][0].size
        val height = spectrograms[0].size

        val graybitmaps = mutableListOf<Bitmap>()
        val paint = Paint().apply {
            colorFilter = ColorMatrixColorFilter(
//                ColorMatrix().apply{setSaturation(0f)}
                ColorMatrix(floatArrayOf(
                    0.33f, 0.33f, 0.33f, 0.0f, 0.0f,
                    0.33f, 0.33f, 0.33f, 0.0f, 0.0f,
                    0.33f, 0.33f, 0.33f, 0.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f, 1.0f
                ))
            )
        }

        for (list in normalizedSpectrogram){
            val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            val canvas = Canvas(bitmap)

            for (y in 0 until height) {
                for (x in 0 until width) {
                    val va = (list[y][x]*255).roundToInt()
                    paint.color = Color.rgb(va,va,va)
                    canvas.drawPoint(x.toFloat(), height-y.toFloat(), paint)
                }
            }
            graybitmaps.add(bitmap)
        }

        val specBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val r = Color.red(graybitmaps[0].getPixel(x, y))
                val g = Color.green(graybitmaps[1].getPixel(x, y))
                val b = Color.blue(graybitmaps[2].getPixel(x, y))
                specBitmap.setPixel(x, y, Color.rgb(r, g, b))
            }
        }
        val elapsedTime = System.currentTimeMillis() - startTime

        return Pair(specBitmap,elapsedTime)
    }

    fun loadCsv(filePath: String): List<FloatArray> {
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

        return rawData
    }

    suspend fun computeSpectrogram(rawData: List<FloatArray>): List<List<FloatArray>> {
        return withContext(Dispatchers.Default) {
            val samplingFrequency = 64f
            val nfft = 128
            val noverlap = 116
            val window: _Window = Hanning(nfft)

//            val spectrograms = mutableListOf<List<FloatArray>>()
            val tasks = (0 until 3).map { i ->
                async {
                    val tmpdata = rawData.map{it[i]}
                    val STFT = ShortTimeFourier(
                        DoubleArray(tmpdata.size) { tmpdata[it].toDouble() },
                        nfft, noverlap, nfft,
                        window, samplingFrequency.toDouble())
                    STFT.transform()
                    var spectrogram = STFT.spectrogram(true).toList()
                    spectrogram.map { it.map { FastMath.log10(it).toFloat() }.toFloatArray() }
//                    spectrograms
                }
            }
            tasks.map { it.await() }
        }
    }

    suspend fun normalizeSpectrogramSuspend(spectrograms: List<List<FloatArray>>): List<List<FloatArray>> {
        return withContext(Dispatchers.Default) {
            val tasks = spectrograms.map { list ->
                async {
                    val values = list.flatMap { it.asList() }
                    val min = values.min() ?: 0f
                    val max = values.max() ?: 1f
                    val range = max - min

                    val result = mutableListOf<FloatArray>()
                    list.forEach { array ->
                        result.add(array.map { (it - min) / range }.toFloatArray())
                    }
                    result
                }
            }
            tasks.map { it.await() }
        }
    }

    fun normalizeSpectrogram(spectrograms: List<List<FloatArray>>): MutableList<List<FloatArray>> {
        val normalizedSpectrogram = mutableListOf<List<FloatArray>>()
        for (list in spectrograms) {
            val values = list.flatMap { it.asList() }
            val min = values.min()
            val max = values.max()
            val range = max - min

            val result = mutableListOf<FloatArray>()
            for (array in list) {
                val newArray = FloatArray(array.size) { i -> (array[i] - min) / range }
                result.add(newArray)
            }
            normalizedSpectrogram.add(result)
        }

        return normalizedSpectrogram
    }
}