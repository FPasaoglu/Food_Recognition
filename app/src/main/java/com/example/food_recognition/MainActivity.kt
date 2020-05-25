package com.example.food_recognition

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import com.google.firebase.ml.common.FirebaseMLException
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel

class MainActivity : AppCompatActivity() {

    //Custom model adı
    private val HOSTED_MODEL_NAME = "foods"
    private val TAG = "MainActivity"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initFirebase()
    }

    private fun initFirebase(){

        try {
            //Modeli indir
            val conditions = FirebaseModelDownloadConditions.Builder()
                .build()

            val remoteModel = FirebaseCustomRemoteModel.Builder(HOSTED_MODEL_NAME).build()
            Toast.makeText(this, "Model indiriliyor.", Toast.LENGTH_SHORT).show()
            //Model başarıyla indirildiyse , RecognitionActivity git
            FirebaseModelManager.getInstance().download(remoteModel, conditions)
                .addOnSuccessListener {
                    Toast.makeText(this, "Başarılı", Toast.LENGTH_SHORT).show()
                    val intent = Intent(this, RecognitionActivity::class.java)
                    startActivity(intent)
                    finish()
                }
                .addOnFailureListener { error ->
                    Toast.makeText(this, "İnternet Bağlantınızı Kontrol Edin!", Toast.LENGTH_SHORT)
                        .show()
                    Log.e(TAG, error.message)
                }
        }
        catch (e : Exception){
            Log.e(TAG, "Error while setting up the model", e)
        }
    }


}
