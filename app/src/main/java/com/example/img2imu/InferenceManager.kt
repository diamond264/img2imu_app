package com.example.img2imu

import android.content.Context
import android.graphics.*
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.util.Log
import androidx.annotation.RequiresApi
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.*
import java.nio.charset.StandardCharsets

class InferenceManager(ctx: Context) {
    private var mContext: Context = ctx
    private val TAG = "MobileInference"
    private val storageDir: String = Environment.getExternalStorageDirectory().absolutePath

    @RequiresApi(Build.VERSION_CODES.O)
    fun performPytorchInference(modelPath: String, inputImage: Bitmap): Pair<Float,Long> {
        val modelFile = assetFilePath(mContext,modelPath)
        val module: Module = Module.load(modelFile)

        val startTime = System.currentTimeMillis()
        val inputBitmap = inputImage
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            inputBitmap.copy(Bitmap.Config.ARGB_8888, true),
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        val outTensor = module.forward(IValue.from(inputTensor)).toTensor()
        val scores: FloatArray = outTensor.getDataAsFloatArray()

        var maxScore = -Float.MAX_VALUE
        var maxScoreIdx: Float = (-1).toFloat()
        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxScoreIdx = i.toFloat()
            }
        }
        val elapsedTime = System.currentTimeMillis() - startTime

        return Pair(maxScoreIdx,elapsedTime)
    }

    fun assetFilePath(context: Context, asset: String): String {
        val file = File(context.filesDir, asset)

        try {
            val inpStream: InputStream = context.assets.open(asset)
            try {
                val outStream = FileOutputStream(file, false)
                val buffer = ByteArray(4 * 1024)
                var read: Int

                while (true) {
                    read = inpStream.read(buffer)
                    if (read == -1) {
                        break
                    }
                    outStream.write(buffer, 0, read)
                }
                outStream.flush()
            } catch (ex: Exception) {
            }
            return file.absolutePath
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return ""
    }
}