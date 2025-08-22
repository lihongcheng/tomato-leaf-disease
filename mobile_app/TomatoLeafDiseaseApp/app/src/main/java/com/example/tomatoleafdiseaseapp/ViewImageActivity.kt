package com.example.tomatoleafdiseaseapp

import android.net.Uri
import android.os.Bundle
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity

class ViewImageActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_view_image)

        val imageView: ImageView = findViewById(R.id.full_screen_image)
        val imageUriString = intent.getStringExtra("image_uri")

        if (imageUriString != null) {
            val imageUri = Uri.parse(imageUriString)
            imageView.setImageURI(imageUri)
        }

        imageView.setOnClickListener {
            finish()
        }
    }
}