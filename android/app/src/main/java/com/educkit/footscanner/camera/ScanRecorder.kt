package com.educkit.footscanner.camera

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.util.Size
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
import java.util.concurrent.Executors

/**
 * CameraX helper tuned for Pixel 3a Milestone 1:
 * rear camera only, zoom forced to 1x, prefer HD video.
 */
class ScanRecorder(private val context: Context) {
    private val executor = Executors.newSingleThreadExecutor()
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var cameraControl: androidx.camera.core.CameraControl? = null

    fun hasPermissions(): Boolean {
        val cam = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
        val mic = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
        return cam == PackageManager.PERMISSION_GRANTED && mic == PackageManager.PERMISSION_GRANTED
    }

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

            val selector = CameraSelector.DEFAULT_BACK_CAMERA
            val camera = provider.bindToLifecycle(
                lifecycleOwner,
                selector,
                preview,
                videoCapture,
            )
            cameraControl = camera.cameraControl
            cameraControl?.setZoomRatio(1.0f)
            cameraControl?.setLinearZoom(0f)
        }, ContextCompat.getMainExecutor(context))
    }

    fun start(output: File, withAudio: Boolean = false, onEvent: (VideoRecordEvent) -> Unit) {
        val capture = videoCapture ?: return
        val pending = capture.output
            .prepareRecording(context, FileOutputOptions.Builder(output).build())
        recording = if (withAudio) {
            pending.withAudioEnabled().start(ContextCompat.getMainExecutor(context), onEvent)
        } else {
            pending.start(ContextCompat.getMainExecutor(context), onEvent)
        }
    }

    fun stop() {
        recording?.stop()
        recording = null
    }
}
