package com.educkit.footscanner.ui

import android.Manifest
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import coil.compose.AsyncImage
import com.educkit.footscanner.camera.ScanRecorder
import com.educkit.footscanner.network.ApiClient
import com.educkit.footscanner.network.CreateScanRequest
import com.educkit.footscanner.network.FrameInfo
import com.educkit.footscanner.network.ProcessRequest
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File

@Composable
fun FootScannerApp() {
    val nav = rememberNavController()
    var foot by remember { mutableStateOf("left") }
    var scanId by remember { mutableStateOf<String?>(null) }
    var videoPath by remember { mutableStateOf<String?>(null) }
    var frames by remember { mutableStateOf<List<FrameInfo>>(emptyList()) }
    var apiBase by remember { mutableStateOf(ApiClient.currentBaseUrl()) }

    NavHost(navController = nav, startDestination = "splash") {
        composable("splash") {
            SplashScreen(onFinished = {
                nav.navigate("home") {
                    popUpTo("splash") { inclusive = true }
                }
            })
        }
        composable("home") {
            HomeScreen(
                apiBase = apiBase,
                onApiBaseChange = {
                    apiBase = it
                    ApiClient.setBaseUrl(it)
                },
                onStart = { nav.navigate("choose") },
            )
        }
        composable("choose") {
            ChooseFootScreen(
                onChoose = {
                    foot = it
                    nav.navigate("prep")
                },
                onBack = { nav.popBackStack() },
            )
        }
        composable("prep") {
            PrepScreen(
                foot = foot,
                onContinue = { nav.navigate("scan") },
                onBack = { nav.popBackStack() },
            )
        }
        composable("scan") {
            GuidedScanScreen(
                onFinished = { path ->
                    videoPath = path
                    nav.navigate("upload")
                },
                onCancel = { nav.popBackStack() },
            )
        }
        composable("upload") {
            UploadScreen(
                foot = foot,
                videoPath = videoPath,
                onDone = { id, selected ->
                    scanId = id
                    frames = selected
                    nav.navigate("frames")
                },
                onBack = { nav.popBackStack() },
            )
        }
        composable("frames") {
            FramesScreen(
                frames = frames,
                apiBase = apiBase,
                onHome = {
                    nav.navigate("home") {
                        popUpTo("home") { inclusive = true }
                    }
                },
            )
        }
    }
}

@Composable
private fun HomeScreen(
    apiBase: String,
    onApiBaseChange: (String) -> Unit,
    onStart: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(24.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        androidx.compose.foundation.Image(
            painter = androidx.compose.ui.res.painterResource(id = com.educkit.footscanner.R.drawable.nikonyx_logo),
            contentDescription = "Nikonyx",
            modifier = Modifier.height(96.dp),
        )
        Text("Nikonyx", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
        Text("Foot Scan test module (standalone until validation)", color = MaterialTheme.colorScheme.secondary)
        Text("Device target: Google Pixel 3a. Keep the foot still; move the phone around it.")
        OutlinedTextField(
            value = apiBase,
            onValueChange = onApiBaseChange,
            label = { Text("PC backend URL") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
        )
        Text("Example: http://192.168.1.42:8000 — phone and PC on same Wi‑Fi.")
        Button(onClick = onStart, modifier = Modifier.fillMaxWidth()) {
            Text("New foot scan")
        }
    }
}

@Composable
private fun ChooseFootScreen(onChoose: (String) -> Unit, onBack: () -> Unit) {
    Column(
        Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Which foot?", style = MaterialTheme.typography.headlineMedium)
        Button(onClick = { onChoose("left") }, modifier = Modifier.fillMaxWidth()) { Text("Left") }
        Button(onClick = { onChoose("right") }, modifier = Modifier.fillMaxWidth()) { Text("Right") }
        OutlinedButton(onClick = onBack) { Text("Back") }
    }
}

@Composable
private fun PrepScreen(foot: String, onContinue: () -> Unit, onBack: () -> Unit) {
    Column(
        Modifier
            .fillMaxSize()
            .padding(24.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("Prepare ${foot.uppercase()} foot", style = MaterialTheme.typography.headlineMedium)
        listOf(
            "Barefoot — remove socks/shoes; trousers clear of ankle",
            "Weight-bearing: stand naturally, foot flat, toes not curled",
            "Place printable calibration mat (or A4) under/around the foot",
            "Bright diffuse light; avoid strong shadows / reflections",
            "Keep the FOOT still. Move the PHONE around for 30 seconds",
            "Cover top, inner, outer, and heel views (about 60–80 images will be kept)",
        ).forEach { Text("• $it") }
        Spacer(Modifier.height(12.dp))
        Button(onClick = onContinue, modifier = Modifier.fillMaxWidth()) { Text("Start guided scan") }
        OutlinedButton(onClick = onBack) { Text("Back") }
    }
}

@Composable
private fun GuidedScanScreen(onFinished: (String) -> Unit, onCancel: () -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val recorder = remember { ScanRecorder(context) }
    var recording by remember { mutableStateOf(false) }
    var seconds by remember { mutableFloatStateOf(0f) }
    var permissionGranted by remember { mutableStateOf(recorder.hasPermissions()) }
    val previewView = remember { PreviewView(context) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { result ->
        permissionGranted = result.values.all { it }
        if (permissionGranted) recorder.bind(previewView, lifecycleOwner)
    }

    LaunchedEffect(Unit) {
        if (!permissionGranted) {
            permissionLauncher.launch(
                arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO),
            )
        } else {
            recorder.bind(previewView, lifecycleOwner)
        }
    }

    LaunchedEffect(recording) {
        if (!recording) return@LaunchedEffect
        seconds = 0f
        while (recording) {
            delay(100)
            seconds += 0.1f
            if (seconds >= 30f) {
                recorder.stop()
                recording = false
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose { recorder.stop() }
    }

    Column(Modifier.fillMaxSize()) {
        Box(
            Modifier
                .weight(1f)
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.surface),
        ) {
            AndroidView({ previewView }, Modifier.fillMaxSize())
            Column(
                Modifier
                    .align(Alignment.BottomStart)
                    .padding(16.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.background.copy(alpha = 0.75f))
                    .padding(12.dp),
            ) {
                Text(if (recording) "Recording ${"%.1f".format(seconds)}s / 30s" else "Ready")
                Text(
                    when {
                        seconds < 5f -> "Start above the toes / top of foot"
                        seconds < 10f -> "Move slowly to the INNER side"
                        seconds < 16f -> "Continue toward the HEEL"
                        seconds < 22f -> "Move to the OUTER side"
                        seconds < 27f -> "Add another top / front pass"
                        else -> "Finish with a slow overview"
                    },
                )
                Text("Keep foot still. No digital zoom.")
            }
        }
        Row(
            Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            OutlinedButton(onClick = onCancel, modifier = Modifier.weight(1f)) { Text("Cancel") }
            Button(
                onClick = {
                    if (!recording) {
                        val out = File(context.cacheDir, "scan_${System.currentTimeMillis()}.mp4")
                        recorder.start(out) { event ->
                            if (event is androidx.camera.video.VideoRecordEvent.Finalize) {
                                if (!event.hasError()) onFinished(out.absolutePath)
                            }
                        }
                        recording = true
                    } else {
                        recorder.stop()
                        recording = false
                    }
                },
                modifier = Modifier.weight(1f),
                enabled = permissionGranted,
            ) {
                Text(if (recording) "Stop" else "Record")
            }
        }
    }
}

@Composable
private fun UploadScreen(
    foot: String,
    videoPath: String?,
    onDone: (String, List<FrameInfo>) -> Unit,
    onBack: () -> Unit,
) {
    val scope = rememberCoroutineScope()
    var message by remember { mutableStateOf("Ready to upload") }
    var progress by remember { mutableFloatStateOf(0f) }
    var busy by remember { mutableStateOf(false) }

    Column(
        Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        NikonyxLoadingHeader(
            title = "Upload & process",
            subtitle = message,
            progress = progress,
        )
        Button(
            enabled = !busy && videoPath != null,
            onClick = {
                scope.launch {
                    busy = true
                    try {
                        message = "Creating scan…"
                        progress = 0.1f
                        val created = ApiClient.api.createScan(
                            CreateScanRequest(
                                foot = foot,
                                device = mapOf(
                                    "manufacturer" to Build.MANUFACTURER,
                                    "model" to Build.MODEL,
                                    "camera_id" to "rear_main",
                                    "resolution" to "1080p",
                                ),
                            ),
                        )
                        message = "Uploading video…"
                        progress = 0.35f
                        val file = File(videoPath!!)
                        ApiClient.api.uploadVideo(created.scanId, ApiClient.videoPart(file))
                        message = "Selecting frames…"
                        progress = 0.55f
                        ApiClient.api.process(created.scanId, ProcessRequest(70))
                        while (true) {
                            delay(700)
                            val st = ApiClient.api.status(created.scanId)
                            progress = (st.progressPercent / 100f).coerceIn(0f, 1f)
                            message = "${st.stage}: ${st.message}"
                            when (st.status) {
                                "frames_ready" -> {
                                    val fr = ApiClient.api.frames(created.scanId)
                                    onDone(created.scanId, fr.frames)
                                    return@launch
                                }
                                "failed" -> {
                                    message = "Failed: ${st.errorCode ?: st.message}"
                                    busy = false
                                    return@launch
                                }
                            }
                        }
                    } catch (e: Exception) {
                        message = "Error: ${e.message}"
                        busy = false
                    }
                }
            },
            modifier = Modifier.fillMaxWidth(),
        ) { Text("Upload to PC") }
        OutlinedButton(onClick = onBack, enabled = !busy) { Text("Back") }
    }
}

@Composable
private fun FramesScreen(
    frames: List<FrameInfo>,
    apiBase: String,
    onHome: () -> Unit,
) {
    Column(
        Modifier
            .fillMaxSize()
            .padding(16.dp),
    ) {
        Text("Selected frames (${frames.size})", style = MaterialTheme.typography.headlineSmall)
        Text("Milestone 1 complete: phone → PC → frame selection.")
        Spacer(Modifier.height(8.dp))
        LazyVerticalGrid(
            columns = GridCells.Adaptive(110.dp),
            contentPadding = PaddingValues(4.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.weight(1f),
        ) {
            items(frames) { frame ->
                Column {
                    AsyncImage(
                        model = apiBase.trimEnd('/') + frame.thumbnailUrl,
                        contentDescription = frame.filename,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(90.dp)
                            .clip(RoundedCornerShape(8.dp)),
                    )
                    Text("sharp ${"%.0f".format(frame.sharpness)}", style = MaterialTheme.typography.labelSmall)
                }
            }
        }
        Button(onClick = onHome, modifier = Modifier.fillMaxWidth()) { Text("Done") }
    }
}
