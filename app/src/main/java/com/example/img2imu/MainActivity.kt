package com.example.img2imu

import android.graphics.Bitmap
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.RequiresApi
import java.util.*

class MainActivity : AppCompatActivity() {
    private val mTAG = "MainActivity"

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val signalPath = "jogging.csv"
        val signalProcessor = SignalProcessor(applicationContext)
        val (spectrogramBitmap,SPtime) = signalProcessor.createSpectrogram(signalPath)
        var tmpspectrogramBitmap = Bitmap.createScaledBitmap(spectrogramBitmap, 128, 96, false)

        val spectrogramImage = findViewById<ImageView>(R.id.image1)
        spectrogramImage.setImageBitmap(tmpspectrogramBitmap)

        val imagePath = "jogging_combined.png"
        val imageProcessor = ImageProcessor(applicationContext)
        var gtSpectrogram = imageProcessor.open(imagePath)
        gtSpectrogram = Bitmap.createScaledBitmap(gtSpectrogram!!, 128, 96, false)

        val spectrogramImage2 = findViewById<ImageView>(R.id.image2)
        spectrogramImage2.setImageBitmap(gtSpectrogram)

        val modelPath = "test_model.pt"
        val inferenceManager = InferenceManager(applicationContext)
        val (outputFloatArray,IFtime) = inferenceManager.performPytorchInference(modelPath,spectrogramBitmap)
        val outtext = findViewById<TextView>(R.id.outtext)
        outtext.setText(outputFloatArray.contentToString()+"   SPtime: "+SPtime+"   IFtime: "+IFtime)
    }

}