package com.educkit.footscanner.ui

import android.Manifest
import android.graphics.Bitmap
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import coil.compose.AsyncImage
import com.educkit.footscanner.R
import com.educkit.footscanner.camera.LensInfo
import com.educkit.footscanner.camera.ScanRecorder
import com.educkit.footscanner.network.ApiClient
import com.educkit.footscanner.network.BestFrame
import com.educkit.footscanner.network.CreateScanRequest
import com.educkit.footscanner.network.FootMesh
import com.educkit.footscanner.network.GuideResponse
import com.educkit.footscanner.network.MeasurementVisual
import com.educkit.footscanner.network.ProcessRequest
import com.educkit.footscanner.network.StatusResponse
import com.educkit.footscanner.network.VideoScanResult
import com.educkit.footscanner.ui.nx.NxCard
import com.educkit.footscanner.ui.nx.NxPrimaryButton
import com.educkit.footscanner.ui.nx.NxSecondaryButton
import com.educkit.footscanner.ui.nx.NxTopBar
import com.educkit.footscanner.ui.nx.SegmentedSwitch
import com.educkit.footscanner.ui.nx.StatusDot
import com.educkit.footscanner.ui.nx.StatusPill
import com.educkit.footscanner.ui.theme.Nx
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.math.max

private const val SCAN_SECONDS = 30f
private const val GUIDE_INTERVAL_MS = 700L

private data class ScanPhase(val id: String, val label: String, val instruction: String, val start: Float, val end: Float, val view: String)

private val PHASES = listOf(
    ScanPhase("top", "Top view", "Hold the phone above the foot — A4 sheet and whole foot in view", 0f, 5f, "top"),
    ScanPhase("inner", "Inner side", "Move slowly to the INNER side, phone at ankle height", 5f, 10f, "medial"),
    ScanPhase("heel", "Heel", "Continue around to the HEEL", 10f, 16f, "rear"),
    ScanPhase("outer", "Outer side", "Move to the OUTER side", 16f, 22f, "lateral"),
    ScanPhase("front", "Front / top", "Finish over the toes and the top of the foot", 22f, 30f, "top"),
)

private fun phaseAt(t: Float): ScanPhase = PHASES.firstOrNull { t >= it.start && t < it.end } ?: PHASES.last()

private fun sideName(foot: String) = foot.replaceFirstChar { it.uppercase() }

@Composable
fun FootScannerApp() {
    val nav = rememberNavController()
    var foot by remember { mutableStateOf("right") }
    var videoPath by remember { mutableStateOf<String?>(null) }
    var lens by remember { mutableStateOf(LensInfo(null, null)) }
    var scanId by remember { mutableStateOf<String?>(null) }
    var result by remember { mutableStateOf<VideoScanResult?>(null) }

    fun restart() = nav.navigate("home") { popUpTo("home") { inclusive = true } }

    NavHost(navController = nav, startDestination = "splash") {
        composable("splash") {
            SplashScreen(onFinished = { nav.navigate("home") { popUpTo("splash") { inclusive = true } } })
        }
        composable("home") { HomeScreen(onStart = { nav.navigate("choose") }) }
        composable("choose") {
            ChooseFootScreen(onChoose = { foot = it; nav.navigate("prep") }, onBack = { nav.popBackStack() })
        }
        composable("prep") {
            PrepScreen(foot, onContinue = { nav.navigate("scan") }, onBack = { nav.popBackStack() })
        }
        composable("scan") {
            GuidedScanScreen(
                foot = foot,
                onFinished = { path, l ->
                    videoPath = path
                    lens = l
                    nav.navigate("frames")
                },
                onCancel = { nav.popBackStack() },
            )
        }
        composable("frames") {
            BestFramesScreen(
                foot = foot,
                videoPath = videoPath,
                lens = lens,
                onNext = { id -> scanId = id; nav.navigate("processing") },
                onRescan = { nav.navigate("scan") { popUpTo("prep") } },
            )
        }
        composable("processing") {
            ProcessingScreen(
                scanId = scanId,
                onDone = { r -> result = r; nav.navigate("result") { popUpTo("home") } },
                onBack = { nav.popBackStack() },
            )
        }
        composable("result") {
            FootResultScreen(result = result, onScanAgain = { restart() })
        }
    }
}

@Composable
private fun Page(content: @Composable () -> Unit) {
    Box(
        Modifier
            .fillMaxSize()
            .background(Nx.Background),
    ) { content() }
}

// ---------------------------------------------------------------- Page 1: New foot scan

@Composable
private fun HomeScreen(onStart: () -> Unit) {
    var showServer by remember { mutableStateOf(false) }
    var server by remember { mutableStateOf(ApiClient.currentBaseUrl()) }
    Page {
        Column(
            Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Spacer(Modifier.height(12.dp))
            Image(painterResource(R.drawable.nikonyx_logo), "NIKONYX", Modifier.size(96.dp))
            Text("Measure Your Foot", color = Nx.Navy, fontSize = 28.sp, fontWeight = FontWeight.Bold)
            Text(
                "A 30-second guided video around your foot on an A4 sheet gives a calibrated 3D foot with every measurement in cm.",
                color = Nx.Muted,
                textAlign = TextAlign.Center,
            )
            NxCard {
                Text("You need", color = Nx.Navy, fontWeight = FontWeight.Bold)
                listOf(
                    "A white A4 sheet (21.0 × 29.7 cm) on a darker, hard floor",
                    "Bare foot, standing naturally with full weight",
                    "Good, even light and 30 seconds",
                ).forEach { Text("• $it", color = Nx.Muted, fontSize = 14.sp) }
            }
            NxPrimaryButton("New foot scan", onClick = onStart)
            Text("Server", color = Nx.Blue, fontSize = 13.sp, modifier = Modifier.clickable { showServer = !showServer })
            if (showServer) {
                OutlinedTextField(
                    value = server,
                    onValueChange = { server = it; runCatching { ApiClient.setBaseUrl(it) } },
                    label = { Text("Scan server URL") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}

// ---------------------------------------------------------------- Page 2: Which foot?

@Composable
private fun ChooseFootScreen(onChoose: (String) -> Unit, onBack: () -> Unit) {
    Page {
        Column(Modifier.fillMaxSize()) {
            NxTopBar("Which foot?", "Each foot is scanned and measured on its own", onBack)
            Row(
                Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                listOf("left", "right").forEach { side ->
                    Column(
                        Modifier
                            .weight(1f)
                            .aspectRatio(0.8f)
                            .clip(RoundedCornerShape(24.dp))
                            .background(Nx.Card)
                            .border(1.5.dp, Nx.Border, RoundedCornerShape(24.dp))
                            .clickable { onChoose(side) }
                            .padding(16.dp),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        Box(
                            Modifier
                                .size(72.dp)
                                .clip(CircleShape)
                                .background(Nx.PaleBlue),
                            contentAlignment = Alignment.Center,
                        ) { Text(if (side == "left") "L" else "R", color = Nx.Blue, fontSize = 30.sp, fontWeight = FontWeight.Bold) }
                        Spacer(Modifier.height(14.dp))
                        Text("${sideName(side)} foot", color = Nx.Navy, fontSize = 20.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- Page 3: Prepare

@Composable
private fun PrepScreen(foot: String, onContinue: () -> Unit, onBack: () -> Unit) {
    Page {
        Column(Modifier.fillMaxSize()) {
            NxTopBar("Prepare your ${foot} foot", "Read once, then start the 30 s guided scan", onBack)
            Column(
                Modifier
                    .weight(1f)
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                NxCard {
                    Text("Set up", color = Nx.Navy, fontWeight = FontWeight.Bold)
                    listOf(
                        "Put the white A4 sheet on a darker floor — not on a white table or white tiles, so its 4 corners stand out.",
                        "Stand barefoot with your ${foot} heel on the paper and the whole foot on the sheet.",
                        "Full weight on the foot, toes relaxed. Trousers above the ankle.",
                        "Bright, even light. Avoid strong shadows across the paper.",
                    ).forEach { Text("• $it", color = Nx.Muted, fontSize = 14.sp) }
                }
                NxCard {
                    Text("During the 30 s scan", color = Nx.Navy, fontWeight = FontWeight.Bold)
                    Text("Keep the FOOT still and move the PHONE slowly around it:", color = Nx.Muted, fontSize = 14.sp)
                    PHASES.forEach { p ->
                        Row(horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text("${p.start.toInt()}–${p.end.toInt()} s", color = Nx.Blue, fontWeight = FontWeight.SemiBold, fontSize = 13.sp, modifier = Modifier.size(width = 56.dp, height = 20.dp))
                            Text("${p.label}: ${p.instruction}", color = Nx.Navy, fontSize = 13.sp)
                        }
                    }
                    Text("Follow the colours: green = good, amber = adjust, red = fix now.", color = Nx.Muted, fontSize = 13.sp)
                }
            }
            Column(
                Modifier
                    .navigationBarsPadding()
                    .padding(16.dp),
            ) { NxPrimaryButton("Start guided scan (30 s)", onClick = onContinue) }
        }
    }
}

// ---------------------------------------------------------------- Page 4a: Guided 30 s scan

@Composable
private fun GuidedScanScreen(foot: String, onFinished: (String, LensInfo) -> Unit, onCancel: () -> Unit) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()
    val recorder = remember { ScanRecorder(context) }
    val previewView = remember { PreviewView(context) }
    var permission by remember { mutableStateOf(recorder.hasPermissions()) }
    var recording by remember { mutableStateOf(false) }
    var seconds by remember { mutableFloatStateOf(0f) }
    var guide by remember { mutableStateOf<GuideResponse?>(null) }
    var offline by remember { mutableStateOf(false) }
    val session = remember { "v${System.currentTimeMillis()}" }

    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        permission = granted
        if (granted) recorder.bind(previewView, owner)
    }

    LaunchedEffect(Unit) {
        if (permission) recorder.bind(previewView, owner) else launcher.launch(Manifest.permission.CAMERA)
    }

    LaunchedEffect(permission) {
        if (!permission) return@LaunchedEffect
        while (true) {
            delay(GUIDE_INTERVAL_MS)
            val bmp = previewView.bitmap ?: continue
            val phase = phaseAt(seconds)
            try {
                val bytes = withContext(Dispatchers.Default) { encodeFrame(bmp) }
                val r = withContext(Dispatchers.IO) {
                    ApiClient.api.guideFrame(
                        ApiClient.jpegPart("file", bytes),
                        ApiClient.text("${foot}_${phase.view}")!!,
                        ApiClient.text(session)!!,
                        ApiClient.text(recorder.lens.focalMm),
                        ApiClient.text(recorder.lens.sensorLongMm),
                    )
                }
                guide = r
                offline = false
            } catch (e: Exception) {
                offline = true
            }
        }
    }

    LaunchedEffect(recording) {
        if (!recording) return@LaunchedEffect
        seconds = 0f
        while (recording) {
            delay(100)
            seconds += 0.1f
            if (seconds >= SCAN_SECONDS) {
                recorder.stop()
                recording = false
            }
        }
    }

    DisposableEffect(Unit) { onDispose { recorder.stop() } }

    val phase = phaseAt(seconds)
    val state = if (offline) "AMBER" else guide?.guidanceState ?: "AMBER"
    val color = Nx.state(state)
    val message = when {
        offline -> "Live guidance offline — check the PC server"
        guide == null -> "Point the camera at your foot on the A4 sheet"
        else -> guide!!.message
    }

    Column(
        Modifier
            .fillMaxSize()
            .background(Nx.Navy),
    ) {
        Box(Modifier.background(Nx.Background)) {
            NxTopBar("${sideName(foot)} foot · guided scan", if (recording) phase.instruction else "Set up the A4 and foot, then start", onCancel.takeIf { !recording })
        }
        PhaseBar(seconds, recording)
        Box(
            Modifier
                .weight(1f)
                .fillMaxWidth(),
        ) {
            if (permission) {
                AndroidView({ previewView }, Modifier.fillMaxSize())
                Canvas(Modifier.fillMaxSize()) { drawGuide(guide, color, phase.view == "top") }
            } else {
                Text("Camera permission is required", color = Color.White, modifier = Modifier.align(Alignment.Center))
            }
            if (recording) {
                Box(
                    Modifier
                        .align(Alignment.TopEnd)
                        .padding(12.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(Nx.Card.copy(alpha = 0.92f))
                        .padding(horizontal = 14.dp, vertical = 8.dp),
                ) {
                    Text("● ${"%.1f".format(seconds)} / 30 s", color = Nx.Red, fontWeight = FontWeight.Bold)
                }
            }
        }
        Column(
            Modifier
                .fillMaxWidth()
                .background(Nx.Background)
                .padding(horizontal = 16.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Column(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(20.dp))
                    .background(Nx.Card)
                    .border(1.5.dp, color, RoundedCornerShape(20.dp))
                    .padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    StatusDot(color, 14)
                    Text(message, color = Nx.Navy, fontWeight = FontWeight.SemiBold, fontSize = 15.sp, modifier = Modifier.weight(1f))
                }
                Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
                    Text("Now: ${phase.label}", color = Nx.Blue, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                    guide?.cameraTiltDeg?.let { t ->
                        Text("Tilt ${"%.0f".format(t)}°", color = if (t <= 10) Nx.Green else if (t <= 15) Nx.Amber else Nx.Red, fontSize = 13.sp)
                    }
                    Text("Quality ${guide?.qualityScore ?: 0}%", color = Nx.Muted, fontSize = 13.sp)
                }
            }
            NxPrimaryButton(
                text = if (recording) "Stop" else "Start 30 s scan",
                enabled = permission,
                onClick = {
                    if (!recording) {
                        // A new scan replaces any earlier one whose upload never finished; each is ~35 MB.
                        context.cacheDir.listFiles { f -> f.name.startsWith("scan_") && f.name.endsWith(".mp4") }?.forEach { it.delete() }
                        val out = File(context.cacheDir, "scan_${System.currentTimeMillis()}.mp4")
                        recorder.start(out) { event ->
                            if (event is VideoRecordEvent.Finalize && !event.hasError()) {
                                scope.launch { onFinished(out.absolutePath, recorder.lens) }
                            }
                        }
                        recording = true
                    } else {
                        recorder.stop()
                        recording = false
                    }
                },
            )
            Spacer(Modifier.navigationBarsPadding())
        }
    }
}

private fun encodeFrame(src: Bitmap): ByteArray {
    val scale = 640f / max(src.width, src.height)
    val bmp = if (scale < 1f) Bitmap.createScaledBitmap(src, (src.width * scale).toInt(), (src.height * scale).toInt(), true) else src
    val out = ByteArrayOutputStream()
    bmp.compress(Bitmap.CompressFormat.JPEG, 80, out)
    return out.toByteArray()
}

@Composable
private fun PhaseBar(seconds: Float, recording: Boolean) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(Nx.Background)
            .padding(horizontal = 16.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        PHASES.forEach { p ->
            val fill = when {
                !recording -> 0f
                seconds >= p.end -> 1f
                seconds <= p.start -> 0f
                else -> (seconds - p.start) / (p.end - p.start)
            }
            Column(Modifier.weight(p.end - p.start)) {
                LinearProgressIndicator(
                    progress = { fill },
                    color = Nx.MeasureBlue,
                    trackColor = Nx.Border,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(5.dp)
                        .clip(RoundedCornerShape(3.dp)),
                )
                Text(p.label, fontSize = 10.sp, color = if (recording && phaseAt(seconds) == p) Nx.Navy else Nx.Muted, maxLines = 1)
            }
        }
    }
}

private fun DrawScope.drawGuide(guide: GuideResponse?, color: Color, isTop: Boolean) {
    val w = size.width
    val h = size.height
    fun map(p: List<Double>) = Offset(p[0].toFloat() * w, p[1].toFloat() * h)
    fun poly(points: List<List<Double>>): Path = Path().apply {
        points.forEachIndexed { i, p -> val o = map(p); if (i == 0) moveTo(o.x, o.y) else lineTo(o.x, o.y) }
        close()
    }
    val dash = PathEffect.dashPathEffect(floatArrayOf(18f, 12f))
    val paper = guide?.overlays?.paper
    if (paper != null && paper.size == 4) {
        drawPath(poly(paper), color, style = Stroke(width = 5f))
    } else if (isTop) {
        val tw = w * 0.6f
        val th = tw * 297f / 210f
        drawRect(
            Color.White.copy(alpha = 0.85f),
            topLeft = Offset((w - tw) / 2f, (h - th) / 2f),
            size = androidx.compose.ui.geometry.Size(tw, th),
            style = Stroke(width = 4f, pathEffect = dash),
        )
    }
    guide?.overlays?.heelLine?.takeIf { it.size == 2 }?.let { drawLine(Nx.MeasureBlue, map(it[0]), map(it[1]), strokeWidth = 5f, pathEffect = dash) }
    guide?.overlays?.footContour?.takeIf { it.size > 3 }?.let { pts ->
        val path = poly(pts)
        drawPath(path, color.copy(alpha = 0.18f))
        drawPath(path, color, style = Stroke(width = 4f))
    }
}

// ---------------------------------------------------------------- Page 4b: Best 5 frames

@Composable
private fun BestFramesScreen(foot: String, videoPath: String?, lens: LensInfo, onNext: (String) -> Unit, onRescan: () -> Unit) {
    var message by remember { mutableStateOf("Uploading your 30 s scan…") }
    var progress by remember { mutableFloatStateOf(0.05f) }
    var scanId by remember { mutableStateOf<String?>(null) }
    var best by remember { mutableStateOf<List<BestFrame>>(emptyList()) }
    var counts by remember { mutableStateOf<Pair<Int, Int>?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var attempt by remember { mutableIntStateOf(0) }

    LaunchedEffect(attempt) {
        if (videoPath == null) return@LaunchedEffect
        error = null
        try {
            val created = ApiClient.api.createScan(
                CreateScanRequest(
                    foot = foot,
                    device = mapOf(
                        "manufacturer" to Build.MANUFACTURER,
                        "model" to Build.MODEL,
                        "camera_id" to "rear_main",
                        "resolution" to "720p",
                        "focal_length_mm" to lens.focalMm,
                        "sensor_long_mm" to lens.sensorLongMm,
                    ),
                ),
            )
            progress = 0.15f
            ApiClient.api.uploadVideo(created.scanId, ApiClient.videoPart(File(videoPath)))
            message = "Extracting frames…"
            progress = 0.3f
            ApiClient.api.process(created.scanId, ProcessRequest(70))
            while (true) {
                delay(800)
                val st = ApiClient.api.status(created.scanId)
                progress = 0.3f + 0.7f * st.progressPercent / 100f
                message = st.message
                if (st.status == "frames_ready") break
                if (st.status == "failed") throw IllegalStateException(st.message)
            }
            val fr = ApiClient.api.frames(created.scanId)
            counts = fr.selectedFrameCount to fr.originalFrameCount
            best = fr.bestFrames
            scanId = created.scanId
            File(videoPath).delete()
        } catch (e: Exception) {
            error = "Could not prepare frames: ${e.message}"
        }
    }

    Page {
        Column(Modifier.fillMaxSize()) {
            NxTopBar("Best 5 frames", "${sideName(foot)} foot · chosen from different positions")
            Column(
                Modifier
                    .weight(1f)
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                if (scanId == null && error == null) {
                    NxCard {
                        Text(message, color = Nx.Navy, fontWeight = FontWeight.SemiBold)
                        LinearProgressIndicator(progress = { progress }, color = Nx.Blue, trackColor = Nx.Border, modifier = Modifier.fillMaxWidth())
                        Text("Keeping the sharpest 60–80 images from your 30 s video.", color = Nx.Muted, fontSize = 13.sp)
                    }
                }
                error?.let {
                    NxCard(borderColor = Nx.Red) {
                        Text(it, color = Nx.Red)
                        NxSecondaryButton("Try again", onClick = { attempt++ })
                    }
                }
                counts?.let { (sel, total) ->
                    Text("$sel sharp images kept from $total captured frames", color = Nx.Muted, fontSize = 13.sp)
                }
                if (best.isNotEmpty() && best.none { it.a4Detected }) {
                    NxCard(borderColor = Nx.Amber) {
                        Text("A4 sheet not detected in these frames", color = Nx.Amber, fontWeight = FontWeight.Bold)
                        Text("Without the A4 corners the system cannot give real centimetres. Put the sheet on a darker floor and keep all 4 corners in view during the top view.", color = Nx.Muted, fontSize = 13.sp)
                    }
                }
                best.chunked(2).forEach { row ->
                    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        row.forEach { BestFrameTile(it, Modifier.weight(1f)) }
                        if (row.size == 1) Spacer(Modifier.weight(1f))
                    }
                }
            }
            Column(
                Modifier
                    .navigationBarsPadding()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                NxPrimaryButton("Next", onClick = { scanId?.let(onNext) }, enabled = scanId != null)
                NxSecondaryButton("Scan again", onClick = onRescan)
            }
        }
    }
}

@Composable
private fun BestFrameTile(frame: BestFrame, modifier: Modifier) {
    Column(
        modifier
            .clip(RoundedCornerShape(20.dp))
            .background(Nx.Card)
            .border(1.dp, Nx.Border, RoundedCornerShape(20.dp))
            .padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        AsyncImage(
            model = ApiClient.absolute(frame.imageUrl),
            contentDescription = frame.label,
            contentScale = ContentScale.Crop,
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(0.75f)
                .clip(RoundedCornerShape(14.dp))
                .background(Nx.PaleBlue),
        )
        Text("${frame.rank}. ${frame.label}", color = Nx.Navy, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
            StatusPill("${frame.qualityScore}%", if (frame.qualityScore >= 75) Nx.Green else if (frame.qualityScore >= 50) Nx.Amber else Nx.Red)
            if (frame.a4Detected) Text("A4 ✓", color = Nx.Green, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

// ---------------------------------------------------------------- Page 5: Processing

@Composable
private fun ProcessingScreen(scanId: String?, onDone: (VideoScanResult) -> Unit, onBack: () -> Unit) {
    var status by remember { mutableStateOf<StatusResponse?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var attempt by remember { mutableIntStateOf(0) }

    LaunchedEffect(attempt) {
        val id = scanId ?: return@LaunchedEffect
        error = null
        try {
            ApiClient.api.measure(id)
            while (true) {
                delay(700)
                val st = ApiClient.api.status(id)
                status = st
                when (st.status) {
                    "complete" -> {
                        onDone(ApiClient.api.result(id))
                        return@LaunchedEffect
                    }
                    "failed" -> {
                        error = st.message
                        return@LaunchedEffect
                    }
                }
            }
        } catch (e: Exception) {
            error = "Server error: ${e.message}"
        }
    }

    val pct = status?.progressPercent ?: 0
    Page {
        Column(
            Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .verticalScroll(rememberScrollState())
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Image(painterResource(R.drawable.nikonyx_logo), "NIKONYX", Modifier.size(72.dp))
            Text("Measuring your foot", color = Nx.Navy, fontSize = 22.sp, fontWeight = FontWeight.Bold)
            Box(contentAlignment = Alignment.Center, modifier = Modifier.size(140.dp)) {
                CircularProgressIndicator(
                    progress = { pct / 100f },
                    color = Nx.Blue,
                    trackColor = Nx.Border,
                    strokeWidth = 10.dp,
                    modifier = Modifier.fillMaxSize(),
                )
                Text("$pct%", color = Nx.Navy, fontSize = 32.sp, fontWeight = FontWeight.Bold)
            }
            status?.message?.takeIf { it.isNotBlank() }?.let { Text(it, color = Nx.Muted, fontSize = 14.sp, textAlign = TextAlign.Center) }
            NxCard {
                (status?.stages ?: emptyList()).forEach { s ->
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                        when (s.state) {
                            "running" -> CircularProgressIndicator(Modifier.size(16.dp), color = Nx.Blue, strokeWidth = 2.dp)
                            "done" -> StatusDot(Nx.Green, 14)
                            "failed" -> StatusDot(Nx.Red, 14)
                            else -> StatusDot(Nx.Border, 14)
                        }
                        Text(s.label, color = if (s.state == "pending") Nx.Muted else Nx.Navy, fontWeight = if (s.state == "running") FontWeight.Bold else FontWeight.Normal)
                    }
                }
                if (status?.stages.isNullOrEmpty()) Text("Starting…", color = Nx.Muted)
            }
            error?.let {
                Text(it, color = Nx.Red, textAlign = TextAlign.Center)
                NxPrimaryButton("Try again", onClick = { attempt++ })
                NxSecondaryButton("Back", onClick = onBack)
            }
        }
    }
}

// ---------------------------------------------------------------- Page 6: 3D foot result

private val TABS = listOf("overview" to "Overview", "top" to "Top", "side" to "Side", "girth" to "Girth", "3d" to "3D")
private val OVERVIEW_KEYS = listOf("foot_length", "max_foot_width", "ball_width", "heel_width", "instep_height")

private fun tabItems(tab: String, all: List<MeasurementVisual>): List<MeasurementVisual> = when (tab) {
    "overview" -> OVERVIEW_KEYS.mapNotNull { k -> all.firstOrNull { it.key == k } }
    "top" -> all.filter { it.view == "top" }
    "side" -> all.filter { it.view == "side" || it.view == "plantar" }
    "girth" -> all.filter { it.view == "girth" }
    else -> all
}

private fun tabPreset(tab: String, side: String): CameraPreset = when (tab) {
    "top" -> presetFor("top", side)
    "side" -> presetFor("side", side)
    else -> presetFor("oblique", side)
}

private fun statusLabel(status: String): String? = when (status) {
    "final" -> "Final"
    "estimated" -> "Estimated"
    else -> null
}

private fun cmInt(v: Double?): String? = v?.let { kotlin.math.round(it).toInt().toString() }

private fun sizeText(m: MeasurementVisual): String =
    m.displayValue.takeIf { it.isNotBlank() && it != "—" && !it.contains("scan", ignoreCase = true) }
        ?: m.valueMm?.let { if (m.key == "foot_volume") "${kotlin.math.round(it / 1000.0).toInt()} cm³" else "${kotlin.math.round(it / 10.0).toInt()} cm" }
        ?: "—"

@Composable
private fun FootResultScreen(result: VideoScanResult?, onScanAgain: () -> Unit) {
    val r = result ?: return
    val side = r.foot
    val foot = r.footResult
    var tab by remember { mutableStateOf("overview") }
    var selected by remember { mutableStateOf<String?>("foot_length") }
    var mesh by remember { mutableStateOf<FootMesh?>(null) }
    var meshError by remember { mutableStateOf<String?>(null) }
    val viewer = remember { FootViewerState(presetFor("top", side)) }
    val scope = rememberCoroutineScope()
    val items = tabItems(tab, foot?.measurements ?: emptyList())

    LaunchedEffect(r.scanId) {
        if (foot?.meshAvailable == true) {
            mesh = runCatching { ApiClient.api.mesh(r.scanId) }.onFailure { meshError = "Could not load 3D foot: ${it.message}" }.getOrNull()
        }
    }

    Page {
        Column(Modifier.fillMaxSize()) {
            NxTopBar("Your ${side} foot · 3D", "Drag to rotate · pinch to zoom · two fingers to pan")
            Column(
                Modifier
                    .weight(1f)
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .height(380.dp)
                        .clip(RoundedCornerShape(24.dp))
                        .background(Nx.Card)
                        .border(1.dp, Nx.Border, RoundedCornerShape(24.dp)),
                ) {
                    val m = mesh
                    if (m != null) {
                        FootViewer(
                            mesh = m,
                            state = viewer,
                            measurements = foot?.measurements ?: emptyList(),
                            landmarks = foot?.visualization?.landmarks ?: emptyList(),
                            selectedKey = selected,
                            showLandmarks = selected != null,
                            modifier = Modifier.fillMaxSize(),
                        )
                        Text(
                            "Reset",
                            color = Nx.Blue,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .padding(10.dp)
                                .clip(RoundedCornerShape(12.dp))
                                .background(Nx.PaleBlue)
                                .clickable {
                                    selected = "foot_length"
                                    scope.launch { viewer.animateTo(presetFor("top", side)) }
                                }
                                .padding(horizontal = 12.dp, vertical = 6.dp),
                        )
                    } else {
                        Text(
                            meshError ?: if (foot?.meshAvailable == true) "Loading 3D foot…" else "The 3D foot needs the phone to move all the way around the foot during the 30 s scan.",
                            color = Nx.Muted,
                            textAlign = TextAlign.Center,
                            modifier = Modifier
                                .align(Alignment.Center)
                                .padding(24.dp),
                        )
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    SummaryTile("Foot length", r.summary?.lengthCm, r.summary?.lengthStatus, Modifier.weight(1f))
                    SummaryTile("Foot width", r.summary?.widthCm, r.summary?.widthStatus, Modifier.weight(1f))
                }
                NxCard {
                    Text("Measurements", color = Nx.Navy, fontWeight = FontWeight.Bold)
                    (foot?.measurements ?: emptyList()).forEach { m ->
                        Text("• ${m.measurement} — ${sizeText(m)}", color = Nx.Navy, fontSize = 13.sp)
                    }
                }
                foot?.let { fr ->
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        StatusDot(if (fr.meshMethod == "focus_sfm") Nx.Green else Nx.Blue, 10)
                        Text(
                            when {
                                fr.meshMethod == "focus_sfm" && fr.meshScale == "a4_foot_length" -> "Colour 3D from your video · scaled with the A4 sheet"
                                fr.meshMethod == "focus_sfm" -> "Colour 3D from your video · not to scale"
                                fr.meshAvailable -> "Calibrated 3D shape from the top and side views"
                                else -> "No 3D foot for this scan"
                            },
                            color = Nx.Muted,
                            fontSize = 12.sp,
                        )
                    }
                    if (fr.meshMessages.isNotEmpty()) {
                        NxCard(borderColor = Nx.Amber) {
                            fr.meshMessages.forEach { Text(it, color = Nx.Navy, fontSize = 13.sp) }
                        }
                    }
                }
                SegmentedSwitch(TABS, tab) {
                    tab = it
                    selected = null
                    scope.launch { viewer.animateTo(tabPreset(it, side)) }
                }
                items.forEach { m ->
                    MeasurementRow(m, m.key == selected) {
                        selected = if (selected == m.key) null else m.key
                        if (selected != null) {
                            val preset = presetFor(if (m.view == "girth" || m.view == "3d") "oblique" else m.view, side)
                            scope.launch { viewer.animateTo(preset) }
                        }
                    }
                }
                r.footwear?.let { f ->
                    NxCard(borderColor = Nx.MeasureBlue) {
                        Text("Shoe size", color = Nx.Muted, fontSize = 13.sp)
                        Text("EU ${f.euSize?.let { if (it % 1.0 == 0.0) "%.0f".format(it) else "%.1f".format(it) } ?: "—"}", color = Nx.Navy, fontSize = 28.sp, fontWeight = FontWeight.Bold)
                        Text("Mondopoint ${f.mondopointMm ?: "—"} mm · width ${f.widthCategory ?: "—"} · instep ${f.instepCategory ?: "—"}", color = Nx.Muted, fontSize = 13.sp)
                        if (f.provisional) StatusPill("Provisional", Nx.Amber)
                    }
                }
                r.framesUsed?.let {
                    Text("Measured from ${it.top} calibrated top frames and ${it.side} side frames.", color = Nx.Muted, fontSize = 12.sp)
                }
                Spacer(Modifier.height(8.dp))
            }
            Column(
                Modifier
                    .navigationBarsPadding()
                    .padding(16.dp),
            ) { NxPrimaryButton("Scan again", onClick = onScanAgain) }
        }
    }
}

@Composable
private fun SummaryTile(title: String, valueCm: Double?, status: String?, modifier: Modifier) {
    Column(
        modifier
            .clip(RoundedCornerShape(20.dp))
            .background(Nx.Card)
            .border(1.dp, Nx.Border, RoundedCornerShape(20.dp))
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(title, color = Nx.Muted, fontSize = 13.sp)
        Text(cmInt(valueCm)?.let { "$it cm" } ?: "—", color = if (valueCm != null) Nx.Navy else Nx.Muted, fontSize = 24.sp, fontWeight = FontWeight.Bold)
        statusLabel(status ?: "")?.let { StatusPill(it, Nx.state(status ?: "")) }
    }
}

@Composable
private fun MeasurementRow(m: MeasurementVisual, selected: Boolean, onClick: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(if (selected) Nx.PaleBlue else Nx.Card)
            .border(if (selected) 1.5.dp else 1.dp, if (selected) Nx.MeasureBlue else Nx.Border, RoundedCornerShape(18.dp))
            .clickable(onClick = onClick)
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Box(
                Modifier
                    .size(28.dp)
                    .clip(CircleShape)
                    .background(Nx.Navy),
                contentAlignment = Alignment.Center,
            ) { Text(m.number.toString(), color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold) }
            Text(m.measurement, color = Nx.Navy, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
            Text(
                sizeText(m),
                color = if (m.valueMm != null) Nx.MeasureBlue else Nx.Muted,
                fontWeight = FontWeight.Bold,
                fontSize = 17.sp,
            )
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            statusLabel(m.status)?.let { StatusPill(it, Nx.state(m.status)) }
            if (m.valueMm != null) Text("Confidence ${"%.0f".format(m.confidence * 100)}%", color = Nx.Muted, fontSize = 12.sp)
        }
        if (selected && m.startLandmark != null && m.valueMm != null && m.startLandmark != m.endLandmark) {
            Text(
                "From ${m.startLandmark.replace('_', ' ')} to ${m.endLandmark?.replace('_', ' ')} = ${sizeText(m)}",
                color = Nx.Navy,
                fontSize = 12.sp,
            )
        }
    }
}