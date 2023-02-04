package com.example.img2imu

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.RequiresApi
import java.util.*

class MainActivity : AppCompatActivity() {
    @SuppressLint("SetTextI18n")
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val signalPath = "jogging.csv"
        val modelPath = "test_model.pt"

        val inputDescription = findViewById<TextView>(R.id.inputDescription)
        inputDescription.text = "input: $signalPath\nmodel: $modelPath"

        val signalProcessor = SignalProcessor(applicationContext)
        var (spectrogramBitmap, SPtime) = signalProcessor.createSpectrogram(signalPath)
        spectrogramBitmap = Bitmap.createScaledBitmap(spectrogramBitmap,
            128, 96, false)

        val spectrogramImage = findViewById<ImageView>(R.id.image1)
        spectrogramImage.setImageBitmap(spectrogramBitmap)

        val inferenceManager = InferenceManager(applicationContext)
        val (outIdx, IFtime) = inferenceManager.performPytorchInference(modelPath,
            spectrogramBitmap)

        if (outIdx == 1.0f) {
            val inferenceResult = findViewById<TextView>(R.id.inferenceResult)
            inferenceResult.text = "Result: Jogging"
        } else {
            val inferenceResult = findViewById<TextView>(R.id.inferenceResult)
            inferenceResult.text = "Result: $outIdx"
        }

        val outtext = findViewById<TextView>(R.id.outtext)
        outtext.text = "Transformation time: $SPtime ms\nInference time: $IFtime ms"
    }

}