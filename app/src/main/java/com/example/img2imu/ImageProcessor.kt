package com.example.img2imu

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import java.io.InputStream

import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import org.pytorch.torchvision.TensorImageUtils.bitmapToFloat32Tensor

class ImageProcessor(ctx: Context) {
    private var mContext: Context = ctx

    fun open(path: String): Bitmap? {
        val stream = mContext.assets.open(path)
        return BitmapFactory.decodeStream(stream)
    }

    fun loadTensor(path: String): Tensor {
        var bitmap = open(path)
//        var rotateMatrix = Matrix()
//        rotateMatrix.postRotate(90F)
//        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), rotateMatrix, false);
        bitmap = Bitmap.createScaledBitmap(bitmap!!, 128, 96, true)
//        var rotateMatrix2 = Matrix()
//        rotateMatrix2.postRotate(-90F)
//        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), rotateMatrix2, false);
        return bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
    }
}