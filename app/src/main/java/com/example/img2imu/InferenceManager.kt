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
    fun performPytorchInference(modelPath: String, inputImage: Bitmap): Pair<FloatArray?,Long> {
//        val modelFile = "$storageDir/$modelPath"
//        val inputPath = "$storageDir/inputimg"
        val modelFile = assetFilePath(mContext,modelPath)
        Log.d(TAG, "Perform Pytorch Inference")
//        Log.d(TAG, "module path: "+modelFile)
        val module: Module = Module.load(modelFile)
//        Log.d(TAG, "module: "+module)

        val outputTensorList = mutableListOf<Tensor>()
        val outputList = mutableListOf<Float>()

        val startTime = System.currentTimeMillis()

        val inputBitmap = inputImage
        val height: Int = inputBitmap.getHeight()
        val width: Int = inputBitmap.getWidth()
        Log.d(TAG, width.toString() + " , " + height.toString())
        val resizedInput= Bitmap.createScaledBitmap(inputBitmap, 128, 96, false)
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedInput.copy(Bitmap.Config.ARGB_8888, true),
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

//        Log.d(TAG, inputTensor.getDataAsFloatArray().copyOfRange(0,10).joinToString())
        Log.d(TAG, inputTensor.shape().joinToString())
        val outTensor = module.forward(IValue.from(inputTensor)).toTensor()

        // getting tensor content as java array of floats
        val scores: FloatArray = outTensor.getDataAsFloatArray()

        Log.d(TAG, scores.joinToString())

        // searching for the index with maximum score
        var maxScore = -Float.MAX_VALUE
        var maxScoreIdx: Float = (-1).toFloat()
        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxScoreIdx = i.toFloat()
            }
        }
        outputList.add(maxScoreIdx)
        outputTensorList.add(outTensor)

        val elapsedTime = System.currentTimeMillis() - startTime

//        File(inputPath).walk().forEach {
//            if (it.isFile) {
//                val inputBitmap = BitmapFactory.decodeFile(it.path)
//                val height: Int = inputBitmap.getHeight()
//                val width: Int = inputBitmap.getWidth()
//                Log.d(TAG, it.toString())
//                Log.d(TAG, width.toString() + " , " + height.toString())
//                val resizedInput= Bitmap.createScaledBitmap(inputBitmap, 128, 96, false) //128,96
//                val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
//                    resizedInput.copy(Bitmap.Config.ARGB_8888, true),
//                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
//                    TensorImageUtils.TORCHVISION_NORM_STD_RGB
//                )
//
////                                Log.d(TAG, inputTensor.getDataAsFloatArray().copyOfRange(500,600).joinToString())
//                Log.d(TAG, inputTensor.shape().joinToString())
//                val outTensor = module.forward(IValue.from(inputTensor)).toTensor()
//
//                // getting tensor content as java array of floats
//                val scores: FloatArray = outTensor.getDataAsFloatArray()
//
//                Log.d(TAG, scores.joinToString())
//
//                // searching for the index with maximum score
//                var maxScore = -Float.MAX_VALUE
//                var maxScoreIdx: Float = (-1).toFloat()
//                for (i in scores.indices) {
//                    if (scores[i] > maxScore) {
//                        maxScore = scores[i]
//                        maxScoreIdx = i.toFloat()
//                    }
//                }
//                outputList.add(maxScoreIdx)
//                outputTensorList.add(outTensor)
//            }
//        }
        Log.e("Result", outputList.toString())

        val out = outputList.toFloatArray()
//                        Log.e("ResultOUT", outTensor.dataAsFloatArray.get(0).toString())
//                        Log.e("ResultOUT", maxScoreIdx.toString())
        return Pair(out,elapsedTime)
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