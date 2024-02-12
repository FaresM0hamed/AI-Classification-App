package com.example.classificationtensorflow

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import com.bumptech.glide.Glide
import com.example.classificationtensorflow.databinding.ActivityMainBinding
import com.example.classificationtensorflow.ml.MobilenetV110224Quant
import com.permissionx.guolindev.PermissionX
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.InputStream
import java.util.*

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var bitmap: Bitmap

    private val galleryLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            handleGalleryResult(result.resultCode, result.data)
        }
    private val cameraLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            handleCameraResult(result.resultCode, result.data)
        }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        disableDarkMode ()

        binding.galleryConstraintLayout.setOnClickListener {
            openGallery()
        }

        binding.cameraConstraintLayout.setOnClickListener {
            requestCameraPermission()
        }

    }

    private fun requestCameraPermission() {
        PermissionX.init(this).permissions(Manifest.permission.CAMERA)
            .onForwardToSettings { scope, deniedList ->
                scope.showForwardToSettingsDialog(
                    deniedList, getString(R.string.requestPermissionBody), "Ok", "Cancel"
                )
            }.request { allGranted, _, _ ->
                if (allGranted) {
                    openCamera()
                }
            }
    }

    private fun openGallery() {
        val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        galleryLauncher.launch(galleryIntent)
    }

    private fun openCamera() {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraLauncher.launch(cameraIntent)
    }


    private fun handleGalleryResult(resultCode: Int, data: Intent?) {
        if (resultCode == Activity.RESULT_OK && data != null && data.data != null) {
            val selectedImageUri = data.data!!
            val inputStream: InputStream? =
                contentResolver.openInputStream(selectedImageUri)
            val selectedBitmap = BitmapFactory.decodeStream(inputStream)
            bitmap = selectedBitmap
            runPotatoModel(bitmap)
            binding.emptyStateGroup.visibility = View.GONE
            binding.informationGroup.visibility = View.VISIBLE
            Glide.with(this).load(bitmap).into(binding.selectedImage)
        }
    }

    private fun handleCameraResult(resultCode: Int, data: Intent?) {
        if (resultCode == Activity.RESULT_OK && data != null) {
            val thumbnail = data.getParcelableExtra<Bitmap>("data")
            bitmap = thumbnail!!
            runPotatoModel(bitmap)
            binding.emptyStateGroup.visibility = View.GONE
            binding.informationGroup.visibility = View.VISIBLE
            Glide.with(this).load(bitmap).into(binding.selectedImage)

        }
    }

    private fun runPotatoModel(selectedBitmap: Bitmap) {
        val model = MobilenetV110224Quant.newInstance(this)
        bitmap = Bitmap.createScaledBitmap(selectedBitmap, 224, 224, true)
        // Creates inputs for model.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
        val tensorImage = TensorImage(DataType.UINT8)
        tensorImage.load(bitmap)
        inputFeature0.loadBuffer(tensorImage.buffer)
        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        Log .d ("outputFeature0", outputFeature0.floatArray.contentToString())
        // Get the label that is associated with the highest probability
        val (predictedLabel, maxProbability) = ClassLabels(this).getLabelAndPercentage(outputFeature0.floatArray)
        val formattedPercentage = String.format(Locale.ENGLISH, "%.2f%%", maxProbability * 100)
        // Show the result
        binding.labelName.text = predictedLabel
//        binding.percentage.text = formattedPercentage
        // Releases model resources if no longer used.
        model.close()
    }

    private fun disableDarkMode() {
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO)
    }


}

