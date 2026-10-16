package com.educkit.footscanner.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.util.Size
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.io.File
import kotlin.math.max

data class LensInfo(val focalMm: Float?, val sensorLongMm: Float?)

/** CameraX helper for the Pixel 3a: rear camera, zoom fixed at 1x, HD video. */
class ScanRecorder(private val context: Context) {
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var camera: Camera? = null

    var lens: LensInfo = LensInfo(null, null)
        private set

    fun hasPermissions(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    fun bind(previewView: PreviewView, lifecycleOwner: LifecycleOwner) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            provider.unbindAll()

            val preview = Preview.Builder()
                .setTargetResolution(Size(1920, 1080))
                .build()
            preview.setSurfaceProvider(previewView.surfaceProvider)

            val recorder = Recorder.Builder()
                .setQualitySelector(QualitySelector.from(Quality.HD))
                .build()
            videoCapture = VideoCapture.withOutput(recorder)

            camera = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, videoCapture)
            camera?.cameraControl?.setZoomRatio(1.0f)
            camera?.cameraControl?.setLinearZoom(0f)
            lens = readLens()
        }, ContextCompat.getMainExecutor(context))
    }

    @OptIn(ExperimentalCamera2Interop::class)
    private fun readLens(): LensInfo {
        val info = camera?.cameraInfo ?: return LensInfo(null, null)
        return runCatching {
            val c2 = Camera2CameraInfo.from(info)
            val focal = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
            val size = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
            LensInfo(focal, size?.let { max(it.width, it.height) })
        }.getOrDefault(LensInfo(null, null))
    }

    fun start(output: File, onEvent: (VideoRecordEvent) -> Unit) {
        val capture = videoCapture ?: return
        recording = capture.output
            .prepareRecording(context, FileOutputOptions.Builder(output).build())
            .start(ContextCompat.getMainExecutor(context), onEvent)
    }

    fun stop() {
        recording?.stop()
        recording = null
    }
}
