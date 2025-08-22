package com.example.tomatoleafdiseaseapp

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.view.View
import android.widget.LinearLayout
import android.graphics.Color

class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var buttonSelectImage: Button
    private lateinit var buttonCamera: Button
    private lateinit var textViewResult1: TextView
    private lateinit var textViewResult2: TextView
    private lateinit var textViewResult3: TextView
    private lateinit var buttonSuggestion1: Button
    private lateinit var buttonSuggestion2: Button
    private lateinit var buttonSuggestion3: Button
    private lateinit var buttonHistory: Button
    private lateinit var placeholderText: TextView
    private lateinit var progressBar1: ProgressBar
    private lateinit var progressBar2: ProgressBar
    private lateinit var progressBar3: ProgressBar
    private lateinit var progressText1: TextView
    private lateinit var progressText2: TextView
    private lateinit var progressText3: TextView
    private lateinit var resultsTitle: TextView
    private lateinit var resultsContainer: LinearLayout

    private lateinit var classifier: TFLiteClassifier

    private val diseaseNameMap = mapOf(
        "Bacterial_spot" to "疮痂病",
        "Early_blight" to "早疫病",
        "Healthy" to "健康",
        "Late_blight" to "晚疫病",
        "Leaf_Mold" to "叶霉病",
        "Powdery_Mildew" to "白粉病",
        "Septoria_leaf_spot" to "斑枯病",
        "Spider_mites_Two_spotted_spider_mite" to "红蜘蛛/二斑叶螨",
        "Target_Spot" to "靶斑病",
        "Yellow_Leaf_Curl_Virus" to "黄化曲叶病毒",
        "mosaic_virus" to "花叶病毒"
    )

    private lateinit var pickImageLauncher: ActivityResultLauncher<Array<String>>
    private lateinit var takePictureLauncher: ActivityResultLauncher<Uri>
    private lateinit var requestPermissionLauncher: ActivityResultLauncher<String>


    private var latestImageUri: Uri? = null
    private var latestTimestamp: Long = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // 初始化历史存储
        HistoryRepository.initialize(applicationContext)

        imageView = findViewById(R.id.image_view)
        buttonSelectImage = findViewById(R.id.button_select_image)
        buttonCamera = findViewById(R.id.button_camera)
        textViewResult1 = findViewById(R.id.text_result1)
        textViewResult2 = findViewById(R.id.text_result2)
        textViewResult3 = findViewById(R.id.text_result3)
        buttonSuggestion1 = findViewById(R.id.button_suggestion1)
        buttonSuggestion2 = findViewById(R.id.button_suggestion2)
        buttonSuggestion3 = findViewById(R.id.button_suggestion3)
        buttonHistory = findViewById(R.id.button_history)
        placeholderText = findViewById(R.id.placeholder_text)
        progressBar1 = findViewById(R.id.progress_bar1)
        progressBar2 = findViewById(R.id.progress_bar2)
        progressBar3 = findViewById(R.id.progress_bar3)
        progressText1 = findViewById(R.id.progress_text1)
        progressText2 = findViewById(R.id.progress_text2)
        progressText3 = findViewById(R.id.progress_text3)
        resultsTitle = findViewById(R.id.results_title)
        resultsContainer = findViewById<LinearLayout>(R.id.results_container)

        // 初始隐藏结果区域
        resultsTitle.visibility = View.GONE
        resultsContainer.visibility = View.GONE

        classifier = TFLiteClassifier(this)
        classifier.initialize()

        // Register Activity Result APIs
        requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted: Boolean ->
            if (isGranted) {
                // Permission granted, launch camera
                val photoUri = createImageUri()
                latestImageUri = photoUri
                if (photoUri != null) {
                    takePictureLauncher.launch(photoUri)
                }
            } else {
                // Permission denied. Handle the case where the user denies the permission.
                // You might want to show a message to the user.
            }
        }

        pickImageLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
            if (uri != null) {
                val takeFlags: Int = Intent.FLAG_GRANT_READ_URI_PERMISSION
                contentResolver.takePersistableUriPermission(uri, takeFlags)
                latestImageUri = uri
                latestTimestamp = System.currentTimeMillis()
                val bitmap = loadBitmapFromUri(uri)
                showImageAndClassify(bitmap)
            }
        }

        takePictureLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { success: Boolean ->
            if (success) {
                latestImageUri?.let { uri ->
                    latestTimestamp = System.currentTimeMillis()
                    val bitmap = loadBitmapFromUri(uri)
                    showImageAndClassify(bitmap)
                }
            }
        }

        buttonSelectImage.setOnClickListener {
            pickImageLauncher.launch(arrayOf("image/*"))
        }

        buttonCamera.setOnClickListener {
            when {
                checkSelfPermission(android.Manifest.permission.CAMERA) == android.content.pm.PackageManager.PERMISSION_GRANTED -> {
                    val photoUri = createImageUri()
                    latestImageUri = photoUri
                    if (photoUri != null) {
                        takePictureLauncher.launch(photoUri)
                    }
                }
                else -> {
                    requestPermissionLauncher.launch(android.Manifest.permission.CAMERA)
                }
            }
        }

        buttonHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }
    }

    // Remove deprecated onActivityResult and request codes

    private fun showImageAndClassify(bitmap: Bitmap) {
        imageView.setImageBitmap(bitmap)
        placeholderText.text = ""
        // 显示结果区域
        resultsTitle.visibility = View.VISIBLE
        resultsContainer.visibility = View.VISIBLE
        val results = classifier.classify(bitmap)
        displayResults(results)
        if (results.isNotEmpty()) {
            // 保存第一名到历史记录
            val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
            val timeLabel = sdf.format(Date(latestTimestamp))
            val diseaseName = results[0].first // 使用完整的英文名
            val confidence = results[0].second
            latestImageUri?.let {
                HistoryRepository.addHistoryItem(diseaseName, confidence, it.toString())
            }
        }
    }

    private fun displayResults(results: List<Pair<String, Float>>) {
        // Reset
        textViewResult1.text = getString(R.string.default_result_1)
        textViewResult2.text = getString(R.string.default_result_2)
        textViewResult3.text = getString(R.string.default_result_3)
        progressBar1.progress = 0
        progressBar2.progress = 0
        progressBar3.progress = 0
        progressText1.text = ""
        progressText2.text = ""
        progressText3.text = ""
        textViewResult1.setTextColor(Color.BLACK)

        if (results.isNotEmpty()) {
            val result1 = results[0]
            val diseaseName1 = result1.first.replace("Tomato___", "")
            val chineseName1 = diseaseNameMap[diseaseName1] ?: diseaseName1
            textViewResult1.text = "1. $chineseName1"
            textViewResult1.setTextColor(Color.RED)
            progressBar1.progress = (result1.second * 100).toInt()
            progressText1.text = "置信度: ${String.format("%.2f", result1.second * 100)}%"
            buttonSuggestion1.setOnClickListener {
                openSuggestion(result1.first, result1.second)
            }
        }
        if (results.size >= 2) {
            val result2 = results[1]
            val diseaseName2 = result2.first.replace("Tomato___", "")
            val chineseName2 = diseaseNameMap[diseaseName2] ?: diseaseName2
            textViewResult2.text = "2. $chineseName2"
            progressBar2.progress = (result2.second * 100).toInt()
            progressText2.text = "置信度: ${String.format("%.2f", result2.second * 100)}%"
            buttonSuggestion2.setOnClickListener {
                openSuggestion(result2.first, result2.second)
            }
        }
        if (results.size >= 3) {
            val result3 = results[2]
            val diseaseName3 = result3.first.replace("Tomato___", "")
            val chineseName3 = diseaseNameMap[diseaseName3] ?: diseaseName3
            textViewResult3.text = "3. $chineseName3"
            progressBar3.progress = (result3.second * 100).toInt()
            progressText3.text = "置信度: ${String.format("%.2f", result3.second * 100)}%"
            buttonSuggestion3.setOnClickListener {
                openSuggestion(result3.first, result3.second)
            }
        }
    }

    private fun openSuggestion(diseaseName: String, confidence: Float) {
        val intent = Intent(this, SuggestionActivity::class.java)
        intent.putExtra("disease_name", diseaseName)
        intent.putExtra("confidence", confidence)
        intent.putExtra("timestamp", latestTimestamp)
        latestImageUri?.let { intent.putExtra("image_uri", it.toString()) }
        startActivity(intent)
    }

    private fun saveBitmapToCache(bitmap: Bitmap): Uri? {
        return try {
            val cacheDir = File(cacheDir, "images").apply { if (!exists()) mkdirs() }
            val file = File(cacheDir, "last_image.jpg")
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            }
            Uri.fromFile(file)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun createImageUri(): Uri? {
        return try {
            val cacheDir = File(cacheDir, "images").apply { if (!exists()) mkdirs() }
            val file = File(cacheDir, "camera_${System.currentTimeMillis()}.jpg")
            FileProvider.getUriForFile(this, "${applicationContext.packageName}.fileprovider", file)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun loadBitmapFromUri(uri: Uri): Bitmap {
        val bitmap = if (Build.VERSION.SDK_INT >= 28) {
            val source = ImageDecoder.createSource(this.contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                decoder.isMutableRequired = true
            }
        } else {
            this.contentResolver.openInputStream(uri).use { input ->
                BitmapFactory.decodeStream(input)
            }
        }
        return bitmap.copy(Bitmap.Config.ARGB_8888, true)
    }

    companion object {
        // removed deprecated request codes
    }
}