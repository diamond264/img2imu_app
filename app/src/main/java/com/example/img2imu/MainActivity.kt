package com.example.img2imu

import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import java.util.*

class MainActivity : AppCompatActivity() {
    private val mTAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val signalPath = "jogging.csv"
        val signalProcessor = SignalProcessor(applicationContext)
        var spectrogramBitmap = signalProcessor.createSpectrogram(signalPath)
        spectrogramBitmap = Bitmap.createScaledBitmap(spectrogramBitmap, 64, 48, false)

        val spectrogramImage = findViewById<ImageView>(R.id.image1)
        spectrogramImage.setImageBitmap(spectrogramBitmap)

        val imagePath = "jogging_combined.png"
        val imageProcessor = ImageProcessor(applicationContext)
        var gtSpectrogram = imageProcessor.open(imagePath)
        gtSpectrogram = Bitmap.createScaledBitmap(gtSpectrogram!!, 64, 48, false)

        val spectrogramImage2 = findViewById<ImageView>(R.id.image2)
        spectrogramImage2.setImageBitmap(gtSpectrogram)
    }

    fun testImageProcessor() {
        val imageProcessor = ImageProcessor(applicationContext)

        val imgPath = "test2.png"
        val imgTensor = imageProcessor.loadTensor(imgPath)
        Log.d(mTAG, ""+imgTensor.dtype())
        Log.d(mTAG, ""+imgTensor.shape().contentToString())
        val farr = imgTensor.dataAsFloatArray

        for(i: Int in 200..300 step(1)) {
            Log.d(mTAG, ""+i+": "+farr.get(i))
        }
    }
}