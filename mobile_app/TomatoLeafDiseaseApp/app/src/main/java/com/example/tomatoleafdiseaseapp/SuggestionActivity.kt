package com.example.tomatoleafdiseaseapp

import android.net.Uri
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.content.Intent

class SuggestionActivity : AppCompatActivity() {

    private val diseaseNameMap = mapOf(
        "Tomato___Bacterial_spot" to "疮痂病",
        "Tomato___Early_blight" to "早疫病",
        "Tomato___Late_blight" to "晚疫病",
        "Tomato___Leaf_Mold" to "叶霉病",
        "Tomato___Septoria_leaf_spot" to "壳针孢叶斑病",
        "Tomato___Spider_mites_Two_spotted_spider_mite" to "红蜘蛛/二斑叶螨",
        "Tomato___Target_Spot" to "靶斑病",
        "Tomato___Yellow_Leaf_Curl_Virus" to "黄化曲叶病毒病",
        "Tomato___mosaic_virus" to "花叶病毒病",
        "Tomato___Healthy" to "健康",
        "Tomato___Powdery_Mildew" to "白粉病"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_suggestion)

        var diseaseName = intent.getStringExtra("disease_name")
        val confidence = intent.getFloatExtra("confidence", -1f)
        val timestamp = intent.getLongExtra("timestamp", 0L)
        val imageUriString = intent.getStringExtra("image_uri")
        val fromHistory = intent.getBooleanExtra("from_history", false)

        if (fromHistory && !diseaseName.isNullOrEmpty() && !diseaseName.startsWith("Tomato___")) {
            diseaseName = "Tomato___$diseaseName"
        }

        val buttonBack: TextView = findViewById(R.id.button_back)
        val textDiseaseName: TextView = findViewById(R.id.text_disease_name)
        val textConfidence: TextView = findViewById(R.id.text_confidence)
        val textTimestamp: TextView = findViewById(R.id.text_timestamp)
        val imageThumb: ImageView = findViewById(R.id.image_thumbnail)
        val textDiseaseDescription: TextView = findViewById(R.id.text_disease_description)
        val textAgriculturalSuggestion: TextView = findViewById(R.id.text_agricultural_suggestion)
        val textChemicalSuggestion: TextView = findViewById(R.id.text_chemical_suggestion)

        buttonBack.setOnClickListener { finish() }

        val chineseDiseaseName = diseaseNameMap[diseaseName] ?: diseaseName
        textDiseaseName.text = chineseDiseaseName

        if (confidence >= 0f) {
            textConfidence.text = getString(R.string.confidence_value, confidence * 100f)
        } else {
            textConfidence.text = getString(R.string.confidence_placeholder)
        }

        if (timestamp > 0L) {
            val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.getDefault())
            textTimestamp.text = getString(R.string.timestamp_value, sdf.format(java.util.Date(timestamp)))
        } else {
            textTimestamp.text = getString(R.string.timestamp_placeholder)
        }

        if (!imageUriString.isNullOrEmpty()) {
            val uri = android.net.Uri.parse(imageUriString)
            imageThumb.setImageURI(uri)
            imageThumb.setOnClickListener {
                val intent = Intent(this, ViewImageActivity::class.java)
                intent.putExtra("image_uri", imageUriString)
                startActivity(intent)
            }
        }

        when (diseaseName) {
            "Tomato___Bacterial_spot" -> {
                textDiseaseDescription.text = getString(R.string.disease_bacterial_spot_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_bacterial_spot)
                textChemicalSuggestion.text = getString(R.string.chem_bacterial_spot)
            }
            "Tomato___Early_blight" -> {
                textDiseaseDescription.text = getString(R.string.disease_early_blight_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_early_blight)
                textChemicalSuggestion.text = getString(R.string.chem_early_blight)
            }
            "Tomato___Late_blight" -> {
                textDiseaseDescription.text = getString(R.string.disease_late_blight_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_late_blight)
                textChemicalSuggestion.text = getString(R.string.chem_late_blight)
            }
            "Tomato___Leaf_Mold" -> {
                textDiseaseDescription.text = getString(R.string.disease_leaf_mold_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_leaf_mold)
                textChemicalSuggestion.text = getString(R.string.chem_leaf_mold)
            }
            "Tomato___Septoria_leaf_spot" -> {
                textDiseaseDescription.text = getString(R.string.disease_septoria_leaf_spot_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_septoria_leaf_spot)
                textChemicalSuggestion.text = getString(R.string.chem_septoria_leaf_spot)
            }
            "Tomato___Spider_mites_Two_spotted_spider_mite" -> {
                textDiseaseDescription.text = getString(R.string.disease_spider_mites_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_spider_mites)
                textChemicalSuggestion.text = getString(R.string.chem_spider_mites)
            }
            "Tomato___Target_Spot" -> {
                textDiseaseDescription.text = getString(R.string.disease_target_spot_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_target_spot)
                textChemicalSuggestion.text = getString(R.string.chem_target_spot)
            }
            "Tomato___Yellow_Leaf_Curl_Virus" -> {
                textDiseaseDescription.text = getString(R.string.disease_yellow_leaf_curl_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_yellow_leaf_curl)
                textChemicalSuggestion.text = getString(R.string.chem_yellow_leaf_curl)
            }
            "Tomato___mosaic_virus" -> {
                textDiseaseDescription.text = getString(R.string.disease_mosaic_virus_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_mosaic_virus)
                textChemicalSuggestion.text = getString(R.string.chem_mosaic_virus)
            }
            "Tomato___Healthy" -> {
                textDiseaseDescription.text = getString(R.string.disease_healthy_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_healthy)
                textChemicalSuggestion.text = getString(R.string.chem_healthy)
            }
            "Tomato___Powdery_Mildew" -> {
                textDiseaseDescription.text = getString(R.string.disease_powdery_mildew_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_powdery_mildew)
                textChemicalSuggestion.text = getString(R.string.chem_powdery_mildew)
            }
            else -> {
                android.util.Log.e("SuggestionActivity", "Unknown disease name: $diseaseName")
                textDiseaseDescription.text = getString(R.string.disease_unknown_desc)
                textAgriculturalSuggestion.text = getString(R.string.agri_unknown)
                textChemicalSuggestion.text = getString(R.string.chem_unknown)
            }
        }
    }
}