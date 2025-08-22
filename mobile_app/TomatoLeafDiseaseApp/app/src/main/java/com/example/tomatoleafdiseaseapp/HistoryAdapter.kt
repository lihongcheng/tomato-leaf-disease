package com.example.tomatoleafdiseaseapp

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class HistoryAdapter(
    private val context: Context,
    private var historyList: List<HistoryItem>,
    private val onDelete: (String) -> Unit
) : RecyclerView.Adapter<HistoryAdapter.HistoryViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): HistoryViewHolder {
        val view = LayoutInflater.from(context).inflate(R.layout.list_item_history, parent, false)
        return HistoryViewHolder(view)
    }

    override fun onBindViewHolder(holder: HistoryViewHolder, position: Int) {
        val historyItem = historyList[position]
        holder.bind(historyItem)
    }

    override fun getItemCount(): Int = historyList.size

    fun updateData(newHistoryList: List<HistoryItem>) {
        historyList = newHistoryList
        notifyDataSetChanged()
    }

    inner class HistoryViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val imageView: ImageView = itemView.findViewById(R.id.history_image)
        private val diseaseName: TextView = itemView.findViewById(R.id.history_disease_name)
        private val confidence: TextView = itemView.findViewById(R.id.history_confidence)
        private val timestamp: TextView = itemView.findViewById(R.id.history_timestamp)
        private val viewSuggestionButton: Button = itemView.findViewById(R.id.btn_view_suggestion)
        private val deleteButton: Button = itemView.findViewById(R.id.btn_delete)

        fun bind(item: HistoryItem) {
            diseaseName.text = DiseaseNameMapper.getChineseName(item.diseaseName)
            confidence.text = String.format(Locale.getDefault(), "置信度: %.2f%%", item.confidence * 100)
            val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
            timestamp.text = sdf.format(Date(item.timestamp))
            val recognitionTime = "识别时间: "
            timestamp.text = recognitionTime + timestamp.text
            
            try {
                val imageUri = Uri.parse(item.imageUri)
                imageView.setImageURI(imageUri)
            } catch (e: SecurityException) {
                Log.e("HistoryAdapter", "Permission denial for URI: ${item.imageUri}", e)
                imageView.setImageResource(R.drawable.placeholder_image) // Fallback to a placeholder
            }

            viewSuggestionButton.setOnClickListener {
                val intent = Intent(context, SuggestionActivity::class.java).apply {
                    putExtra("disease_name", item.diseaseName)
                    putExtra("confidence", item.confidence)
                    putExtra("timestamp", item.timestamp)
                    putExtra("image_uri", item.imageUri)
                    putExtra("from_history", true)
                }
                context.startActivity(intent)
            }

            deleteButton.setOnClickListener {
                onDelete(item.id)
            }
        }
    }
}