package com.example.tomatoleafdiseaseapp

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class HistoryActivity : AppCompatActivity() {

    private lateinit var historyRecyclerView: RecyclerView
    private lateinit var historyAdapter: HistoryAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)

        val buttonBack: TextView = findViewById(R.id.button_back)
        buttonBack.setOnClickListener { finish() }

        historyRecyclerView = findViewById(R.id.history_recycler_view)
        historyRecyclerView.layoutManager = LinearLayoutManager(this)

        historyAdapter = HistoryAdapter(this, HistoryRepository.history) {
            HistoryRepository.removeHistoryItem(it)
            historyAdapter.updateData(HistoryRepository.history)
        }
        historyRecyclerView.adapter = historyAdapter
    }

    override fun onResume() {
        super.onResume()
        historyAdapter.updateData(HistoryRepository.history)
    }
}