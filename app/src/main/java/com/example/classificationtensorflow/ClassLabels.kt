package com.example.classificationtensorflow

import android.content.Context

class ClassLabels(private val context: Context) {

    fun getLabelAndPercentage(floatArray: FloatArray): Pair<String, Float> {
        val (labelIndex, labelPercentage) = getIndexOfMaxValue(floatArray)
        var counter = 0
        var label: String? = null

        context.assets.open("labels.txt").bufferedReader().use { reader ->
            while (counter <= labelIndex) {
                label = reader.readLine()
                counter++
            }
        }
        return Pair(label ?: "", labelPercentage)
    }

    private fun getIndexOfMaxValue(arr: FloatArray): Pair<Int, Float> {
        val labelPercentage = arr.maxOrNull() ?: Float.MIN_VALUE
        val labelIndex = arr.indexOfFirst { it == labelPercentage }
        return Pair(labelIndex, labelPercentage)
    }
}
