package com.example.food_recognition

import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.firebase.ml.common.FirebaseMLException
import com.google.firebase.ml.custom.*
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata
import com.otaliastudios.cameraview.Frame
import kotlinx.android.synthetic.main.activity_recognition.*
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import kotlin.Comparator
import kotlin.experimental.and

class RecognitionActivity : AppCompatActivity() {

    private val TAG = "Realtime"
    //Model Formatı "Quantization veya Floating olabilir.
    //Bundan dolayı bir if bloğu işleminde geçireceğiz
    //Kullanacağımız model formatı Quantization olduğundan - isQuant değişkeni True
    private val isQuant: Boolean = false
    //Quant ve Float olması durumunda işleme sokulacak default değerler
    private val FLOAT_VALUE = 4
    private val QUANT_VALUE = 1
    //Dataset verilerini içeren labels.txt
    private val LABEL_PATH = "labels.txt"
    //Kameradan gelen verinin modele verilmesi için boyut düzenlemesi
    private val DIM_BATCH_SIZE = 1
    private val DIM_PIXEL_SIZE = 3
    private val DIM_IMG_SIZE_X = 100
    private val DIM_IMG_SIZE_Y = 100
    private val intValues = IntArray(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y)
    //Tahmin edilen ilk 3 sonuç göster
    private val RESULTS_TO_SHOW = 3
    //labels ile tanınan nesnelerin karşılaştırılması ?
    private var sortedLabels =
        PriorityQueue<AbstractMap.SimpleEntry<String, Float>>(RESULTS_TO_SHOW,
            Comparator<AbstractMap.SimpleEntry<String, Float>> { o1, o2 -> o1.value.compareTo(o2.value) })
    private lateinit var labelList: List<String>
    //Model input ve output belirlenmesi
    private var interpreter: FirebaseModelInterpreter? = null
    private var dataOptions: FirebaseModelInputOutputOptions? = null

    //Create fonksiyonumuz
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_recognition)
        initFirebase()
        cameraView.setLifecycleOwner(this)
        //Kameradan gelen bilgileri frame olarak classifyFrame fonksiyonuna input olarak gönder
        // Gelen return bilgisini textView'de göster
        cameraView.addFrameProcessor {
            classifyFrame(it)?.addOnSuccessListener { result ->
                txtDetectedObject.text = result[2]
                txtDetectedObject2.text = result[1]
                txtDetectedObject3.text = result[0]
            }
        }
    }

    //Firebase ve kamera işlemleri
    private fun initFirebase() {
        labelList = loadLabelList()
        try {
            val remoteModel = FirebaseCustomRemoteModel.Builder("Fruits-360").build()
            val modelOptions = FirebaseModelInterpreterOptions.Builder(remoteModel).build()
            interpreter = FirebaseModelInterpreter.getInstance(modelOptions)
            val inputDims =
                intArrayOf(DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE)
            val outputDims = intArrayOf(1, labelList.size)
            //Quant - Float32 kontrol
            // !! Quant ise .BYTE , Float ise FLOAT32 belirlenir
            val dataType = if (isQuant) {
                FirebaseModelDataType.BYTE
            } else {
                FirebaseModelDataType.FLOAT32
            }
            dataOptions = FirebaseModelInputOutputOptions.Builder()
                .setInputFormat(0, dataType, inputDims)
                .setOutputFormat(0, dataType, outputDims)
                .build()
        } catch (e: FirebaseMLException) {
            Toast.makeText(this, "Error while setting up the model", Toast.LENGTH_LONG)
        }
    }

    //Frame ile görüntü al , BitmapToBuffer , ardından işlenen frame runInterptreter fonksiyonuna input olarak git
    private fun classifyFrame(frame: Frame): Task<List<String>>? {
        //Frame to Bitmap
        val firebaseVisionImage = getVisionImageFromFrame(frame)
        val bitmap = firebaseVisionImage.bitmap
        if (interpreter == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.")
            val uninitialized = ArrayList<String>()
            uninitialized.add("Uninitialized Classifier.")
            Tasks.forResult<List<String>>(uninitialized)
        }
        //Bitmap to ByteBuffer
        //ByteBuffer use for : This is a good description of its uses and shortcomings.
        // You essentially use it whenever you need to do fast low-level I/O.
        // If you were going to implement a TCP/IP protocol
        // or if you were writing a database (DBMS) this class would come in handy.
        val imgData = convertBitmapToByteBuffer(bitmap)
        val inputs = FirebaseModelInputs.Builder().add(imgData).build()
        return runInterpreter(inputs)
    }

    //Frame recognize işlemi
    private fun runInterpreter(inputs: FirebaseModelInputs): Task<List<String>> {
        val result = dataOptions?.let {
            interpreter!!.run(inputs, it).addOnFailureListener { e ->
                Log.e(TAG, "Failed to get labels array: ${e.message}")
                e.printStackTrace()
            }
                .continueWith { task ->
                    if (isQuant) {
                        val labelProbArray = task.result!!.getOutput<Array<ByteArray>>(0)
                        getTopLabels(labelProbArray)
                    } else {
                        val labelProbArray = task.result!!.getOutput<Array<FloatArray>>(0)
                        getTopLabels(labelProbArray)
                    }
                }
        }
        return result!!
    }

    private fun getTopLabels(labelProbArray: Array<FloatArray>): List<String> {

        for (i in labelList.indices) {
            sortedLabels.add(
                AbstractMap.SimpleEntry(labelList[i], labelProbArray[0][i])
            )
            if (sortedLabels.size > RESULTS_TO_SHOW) {
                sortedLabels.poll()
            }
        }
        val result = ArrayList<String>()
        val size = sortedLabels.size
        for (i in 0 until size) {
            val label = sortedLabels.poll()
            if (label != null) {
                result.add(label.key + ":" + label.value)
            }
        }
        Log.d(TAG, "labels: $result")

        return result
    }

    private fun getTopLabels(labelProbArray: Array<ByteArray>): List<String> {
        for (i in labelList.indices) {
            sortedLabels.add(
                AbstractMap.SimpleEntry(
                    labelList[i],
                    (labelProbArray[0][i] and 0xff.toByte()) / 255.0f
                )
            )
            if (sortedLabels.size > RESULTS_TO_SHOW) {
                sortedLabels.poll()
            }
        }
        val result = ArrayList<String>()
        val size = sortedLabels.size
        for (i in 0 until size) {
            val label = sortedLabels.poll()
            if (label != null) {
                result.add(label.key + ":" + label.value)
            }
        }
        Log.d(TAG, "labels: $result")
        return result
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val value = if (isQuant) {
            QUANT_VALUE
        } else {
            FLOAT_VALUE
        }
        val imgData = ByteBuffer.allocateDirect(
            value * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE
        )
        imgData.order(ByteOrder.nativeOrder())
        val scaledBitmap = Bitmap.createScaledBitmap(
            bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, true
        )
        imgData.rewind()
        scaledBitmap.getPixels(
            intValues, 0, scaledBitmap.width, 0, 0, scaledBitmap.width,
            scaledBitmap.height
        )
        var pixel = 0
        for (i in 0 until DIM_IMG_SIZE_X) {
            for (j in 0 until DIM_IMG_SIZE_Y) {
                val intValue = intValues[pixel++]
                if (isQuant) {
                    imgData.put((intValue shr 16 and 0xFF).toByte())
                    imgData.put((intValue shr 8 and 0xFF).toByte())
                    imgData.put((intValue and 0xFF).toByte())
                } else {
                    imgData.putFloat((intValue shr 16 and 0xFF) / 255.0f)
                    imgData.putFloat((intValue shr 8 and 0xFF) / 255.0f)
                    imgData.putFloat((intValue and 0xFF) / 255.0f)
                }
            }
        }
        return imgData
    }

    private fun getVisionImageFromFrame(frame: Frame): FirebaseVisionImage {
        val data = frame.data
        val imageMetaData = FirebaseVisionImageMetadata.Builder()
            .setFormat(FirebaseVisionImageMetadata.IMAGE_FORMAT_NV21)
            .setRotation(FirebaseVisionImageMetadata.ROTATION_90)
            .setWidth(frame.size.width)
            .setHeight(frame.size.height)
            .build()
        return FirebaseVisionImage.fromByteArray(data, imageMetaData)
    }


    private fun loadLabelList(): List<String> {
        val labelList = ArrayList<String>()
        try {
            BufferedReader(InputStreamReader(this.assets.open(LABEL_PATH))).use { reader ->
                var line = reader.readLine()
                while (line != null) {
                    labelList.add(line)
                    line = reader.readLine()
                }
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read label list.", e)
        }
        return labelList
    }
}
