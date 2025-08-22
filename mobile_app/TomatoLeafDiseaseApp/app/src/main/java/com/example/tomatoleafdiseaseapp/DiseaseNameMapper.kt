package com.example.tomatoleafdiseaseapp

object DiseaseNameMapper {
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

    fun getChineseName(englishName: String): String {
        return diseaseNameMap[englishName] ?: englishName
    }
}