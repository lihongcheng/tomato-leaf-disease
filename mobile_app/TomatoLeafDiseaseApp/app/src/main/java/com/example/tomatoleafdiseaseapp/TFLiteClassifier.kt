package com.example.tomatoleafdiseaseapp

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TFLiteClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null

    private lateinit var labels: List<String>

    fun initialize() {
        val model = loadModelFile("MobileNetV3_small.tflite")
        model.order(ByteOrder.nativeOrder())
        interpreter = Interpreter(model)
        labels = loadLabels()
    }

    private fun loadLabels(): List<String> {
        return context.assets.open("labels.txt").bufferedReader().readLines()
    }

    fun classify(bitmap: Bitmap): List<Pair<String, Float>> {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(224 * 224)
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.width, 0, 0, resizedBitmap.width, resizedBitmap.height)

        var pixel = 0
        for (i in 0 until 224) {
            for (j in 0 until 224) {
                val `val` = intValues[pixel++]
                byteBuffer.putFloat(((`val` shr 16) and 0xFF) / 255.0f)
                byteBuffer.putFloat(((`val` shr 8) and 0xFF) / 255.0f)
                byteBuffer.putFloat((`val` and 0xFF) / 255.0f)
            }
        }

        val output = Array(1) { FloatArray(11) } // Fix: Changed from 10 to 11 classes
        val startTime = SystemClock.elapsedRealtime()
        interpreter?.run(byteBuffer, output)
        val endTime = SystemClock.elapsedRealtime()
        val inferenceTime = endTime - startTime
        Log.d("Performance", "Inference Time: $inferenceTime ms")

        val results = mutableListOf<Pair<String, Float>>()
        val softmaxOutput = softmax(output[0])
        softmaxOutput.forEachIndexed { index, confidence ->
            results.add(Pair(labels[index], confidence))
        }

        results.sortByDescending { it.second }
        return results.take(3)
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0.0f
        val exps = logits.map { kotlin.math.exp(it - maxLogit) }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }.toFloatArray()
    }

    private fun loadModelFile(modelName: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun close() {
        interpreter?.close()
    }
}