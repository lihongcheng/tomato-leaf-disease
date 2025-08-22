package com.example.tomatoleafdiseaseapp

import android.content.Context
import android.content.SharedPreferences
import com.google.gson.Gson
import com.google.gson.JsonSyntaxException
import com.google.gson.reflect.TypeToken
import java.util.UUID
import android.util.Log
import java.text.SimpleDateFormat
import java.util.*

data class HistoryItem(
    val id: String,
    val timestamp: Long,
    val diseaseName: String,
    val confidence: Float,
    val imageUri: String
)

object HistoryRepository {
    private const val PREF_NAME = "history_preferences"
    private const val HISTORY_KEY = "history_list"
    
    private lateinit var sharedPreferences: SharedPreferences
    private val gson = Gson()
    private val _history = mutableListOf<HistoryItem>()
    val history: List<HistoryItem> get() = _history

    fun initialize(context: Context) {
        sharedPreferences = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
        loadHistoryFromPrefs()
    }

    private fun loadHistoryFromPrefs() {
        val historyJson = sharedPreferences.getString(HISTORY_KEY, null)
        if (!historyJson.isNullOrEmpty()) {
            try {
                val type = object : TypeToken<List<HistoryItem>>() {}.type
                val savedHistory = gson.fromJson<List<HistoryItem>>(historyJson, type)
                _history.clear()
                _history.addAll(savedHistory)
            } catch (e: JsonSyntaxException) {
                // Handle error from old data format
                Log.e("HistoryRepository", "Error parsing history JSON, likely due to old format", e)
                // Clear corrupted history
                _history.clear()
                saveHistoryToPrefs() // Overwrite old data
            }
        }
    }

    private fun saveHistoryToPrefs() {
        val historyJson = gson.toJson(_history)
        sharedPreferences.edit().putString(HISTORY_KEY, historyJson).apply()
    }

    fun addHistoryItem(diseaseName: String, confidence: Float, imageUri: String) {
        val timestamp = System.currentTimeMillis()
        val id = UUID.randomUUID().toString()
        _history.add(0, HistoryItem(id, timestamp, diseaseName, confidence, imageUri)) // Add to beginning for newest first
        saveHistoryToPrefs()
    }

    fun removeHistoryItem(id: String) {
        _history.removeIf { it.id == id }
        saveHistoryToPrefs()
    }

    fun clearHistory() {
        _history.clear()
        saveHistoryToPrefs()
    }
}