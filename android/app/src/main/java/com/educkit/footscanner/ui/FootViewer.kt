package com.educkit.footscanner.ui

import android.graphics.Paint
import android.graphics.RectF
import android.os.Build
import androidx.compose.animation.core.animate
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import com.educkit.footscanner.network.FootMesh
import com.educkit.footscanner.network.LandmarkVisual
import com.educkit.footscanner.network.MeasurementVisual
import com.educkit.footscanner.ui.theme.Nx
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

data class CameraPreset(val yaw: Float, val pitch: Float)

fun presetFor(view: String, side: String): CameraPreset = when (view) {
    "top" -> CameraPreset(0f, 90f)
    "side" -> CameraPreset(if (side == "right") 90f else -90f, 0f)
    "lateral" -> CameraPreset(if (side == "right") -90f else 90f, 0f)
    else -> CameraPreset(if (side == "right") 35f else -35f, 32f)
}

class FootViewerState(initial: CameraPreset) {
    var yaw by mutableFloatStateOf(initial.yaw)
    var pitch by mutableFloatStateOf(initial.pitch)
    var zoom by mutableFloatStateOf(1f)
    var panX by mutableFloatStateOf(0f)
    var panY by mutableFloatStateOf(0f)

    suspend fun animateTo(p: CameraPreset, ms: Int = 420) = coroutineScope {
        launch { animate(yaw, p.yaw, animationSpec = tween(ms)) { v, _ -> yaw = v } }
        launch { animate(pitch, p.pitch, animationSpec = tween(ms)) { v, _ -> pitch = v } }
        launch { animate(zoom, 1f, animationSpec = tween(ms)) { v, _ -> zoom = v } }
        launch { animate(panX, 0f, animationSpec = tween(ms)) { v, _ -> panX = v } }
        launch { animate(panY, 0f, animationSpec = tween(ms)) { v, _ -> panY = v } }
    }
}

private val DEFAULT_RGB = floatArrayOf(201f, 220f, 245f)

/** [faceRgb] holds one base colour per face (r, g, b), averaged from vertex colours when the mesh has them. */
private class PreparedMesh(val verts: FloatArray, val faces: IntArray, val faceRgb: FloatArray, val center: FloatArray, val extent: Float)

private fun prepare(mesh: FootMesh): PreparedMesh? {
    if (mesh.vertices.isEmpty() || mesh.faces.isEmpty()) return null
    val n = mesh.vertices.size
    val v = FloatArray(n * 3)
    val lo = floatArrayOf(Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE)
    val hi = floatArrayOf(-Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE)
    mesh.vertices.forEachIndexed { i, p ->
        for (k in 0..2) {
            val c = p[k].toFloat()
            v[i * 3 + k] = c
            lo[k] = min(lo[k], c)
            hi[k] = max(hi[k], c)
        }
    }
    val f = IntArray(mesh.faces.size * 3)
    mesh.faces.forEachIndexed { i, t -> for (k in 0..2) f[i * 3 + k] = t[k] }
    val vc = mesh.colors?.takeIf { it.size == n }
    val rgb = FloatArray(mesh.faces.size * 3)
    for (i in mesh.faces.indices) {
        for (ch in 0..2) {
            rgb[i * 3 + ch] = if (vc == null) DEFAULT_RGB[ch] else (vc[f[i * 3]][ch] + vc[f[i * 3 + 1]][ch] + vc[f[i * 3 + 2]][ch]) / 3f
        }
    }
    val center = FloatArray(3) { (lo[it] + hi[it]) / 2f }
    val extent = max(hi[0] - lo[0], max(hi[1] - lo[1], hi[2] - lo[2]))
    return PreparedMesh(v, f, rgb, center, extent)
}

private class Projector(state: FootViewerState, private val center: FloatArray, private val scale: Float, private val ox: Float, private val oy: Float) {
    private val cy = cos(Math.toRadians(state.yaw.toDouble())).toFloat()
    private val sy = sin(Math.toRadians(state.yaw.toDouble())).toFloat()
    private val cp = cos(Math.toRadians(state.pitch.toDouble())).toFloat()
    private val sp = sin(Math.toRadians(state.pitch.toDouble())).toFloat()

    /** Returns screen x, screen y, depth (larger = further from the viewer). */
    fun project(x: Float, y: Float, z: Float, out: FloatArray, o: Int = 0) {
        val px = x - center[0]
        val py = y - center[1]
        val pz = z - center[2]
        val qx = px * cy - py * sy
        val qy = px * sy + py * cy
        val up = pz * cp + qy * sp
        val depth = qy * cp - pz * sp
        out[o] = ox + qx * scale
        out[o + 1] = oy - up * scale
        out[o + 2] = depth
    }

    fun offset(p: List<Double>): Offset {
        val tmp = FloatArray(3)
        project(p[0].toFloat(), p[1].toFloat(), p[2].toFloat(), tmp)
        return Offset(tmp[0], tmp[1])
    }
}

@Composable
fun FootViewer(
    mesh: FootMesh?,
    state: FootViewerState,
    measurements: List<MeasurementVisual>,
    landmarks: List<LandmarkVisual>,
    selectedKey: String?,
    showLandmarks: Boolean,
    modifier: Modifier = Modifier,
) {
    val prepared = remember(mesh) { mesh?.let { prepare(it) } }
    val labelPaint = remember {
        Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = 34f; isFakeBoldText = true }
    }
    val bgPaint = remember { Paint(Paint.ANTI_ALIAS_FLAG) }
    val meshPaint = remember { Paint(Paint.ANTI_ALIAS_FLAG) }

    Canvas(
        modifier.pointerInput(Unit) {
            awaitEachGesture {
                awaitFirstDown(requireUnconsumed = false)
                do {
                    val event = awaitPointerEvent()
                    val pressed = event.changes.count { it.pressed }
                    if (pressed >= 2) {
                        state.zoom = (state.zoom * event.calculateZoom()).coerceIn(0.5f, 4f)
                        val pan = event.calculatePan()
                        state.panX += pan.x
                        state.panY += pan.y
                    } else if (pressed == 1) {
                        val d = event.calculatePan()
                        state.yaw += d.x * 0.45f
                        state.pitch = (state.pitch + d.y * 0.45f).coerceIn(-20f, 90f)
                    }
                    event.changes.forEach { if (it.positionChanged()) it.consume() }
                } while (event.changes.any { it.pressed })
            }
        },
    ) {
        val pm = prepared ?: return@Canvas
        val scale = 0.78f * min(size.width, size.height) / pm.extent * state.zoom
        val proj = Projector(state, pm.center, scale, size.width / 2f + state.panX, size.height / 2f + state.panY)
        drawMesh(pm, proj, meshPaint)
        drawMeasurements(proj, measurements, selectedKey, labelPaint, bgPaint)
        if (showLandmarks) drawLandmarks(proj, landmarks, measurements.firstOrNull { it.key == selectedKey }, labelPaint)
        drawScaleBar(scale, labelPaint)
    }
}

private val LIGHT = run {
    val l = floatArrayOf(-0.35f, 0.55f, -1f)
    val n = sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2])
    floatArrayOf(l[0] / n, l[1] / n, l[2] / n)
}

private fun DrawScope.drawMesh(pm: PreparedMesh, proj: Projector, paint: Paint) {
    val nv = pm.verts.size / 3
    val sp = FloatArray(nv * 3)
    for (i in 0 until nv) proj.project(pm.verts[i * 3], pm.verts[i * 3 + 1], pm.verts[i * 3 + 2], sp, i * 3)
    val nf = pm.faces.size / 3
    val depth = FloatArray(nf)
    for (f in 0 until nf) {
        depth[f] = (sp[pm.faces[f * 3] * 3 + 2] + sp[pm.faces[f * 3 + 1] * 3 + 2] + sp[pm.faces[f * 3 + 2] * 3 + 2]) / 3f
    }
    val order = (0 until nf).sortedByDescending { depth[it] }
    val pos = FloatArray(nf * 6)
    val colors = IntArray(nf * 3)
    order.forEachIndexed { k, f ->
        val br = pm.faceRgb[f * 3]
        val bg = pm.faceRgb[f * 3 + 1]
        val bb = pm.faceRgb[f * 3 + 2]
        val a = pm.faces[f * 3]
        val b = pm.faces[f * 3 + 1]
        val c = pm.faces[f * 3 + 2]
        // View-space normal from screen x, screen up (−y) and depth.
        val ux = sp[b * 3] - sp[a * 3]; val uy = -(sp[b * 3 + 1] - sp[a * 3 + 1]); val uz = sp[b * 3 + 2] - sp[a * 3 + 2]
        val vx = sp[c * 3] - sp[a * 3]; val vy = -(sp[c * 3 + 1] - sp[a * 3 + 1]); val vz = sp[c * 3 + 2] - sp[a * 3 + 2]
        var nx = uy * vz - uz * vy
        var ny = uz * vx - ux * vz
        var nz = ux * vy - uy * vx
        val len = sqrt(nx * nx + ny * ny + nz * nz).coerceAtLeast(1e-6f)
        nx /= len; ny /= len; nz /= len
        val shade = 0.42f + 0.58f * abs(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2])
        val col = android.graphics.Color.rgb(
            (br * shade).toInt().coerceIn(0, 255),
            (bg * shade).toInt().coerceIn(0, 255),
            (bb * shade).toInt().coerceIn(0, 255),
        )
        for ((j, vi) in intArrayOf(a, b, c).withIndex()) {
            pos[k * 6 + j * 2] = sp[vi * 3]
            pos[k * 6 + j * 2 + 1] = sp[vi * 3 + 1]
            colors[k * 3 + j] = col
        }
    }
    drawIntoCanvas { canvas ->
        val nc = canvas.nativeCanvas
        if (Build.VERSION.SDK_INT >= 29) {
            nc.drawVertices(android.graphics.Canvas.VertexMode.TRIANGLES, pos.size, pos, 0, null, 0, colors, 0, null, 0, 0, paint)
        } else {
            val path = android.graphics.Path()
            for (k in 0 until nf) {
                path.reset()
                path.moveTo(pos[k * 6], pos[k * 6 + 1])
                path.lineTo(pos[k * 6 + 2], pos[k * 6 + 3])
                path.lineTo(pos[k * 6 + 4], pos[k * 6 + 5])
                path.close()
                paint.color = colors[k * 3]
                nc.drawPath(path, paint)
            }
        }
    }
}

private fun DrawScope.drawMeasurements(proj: Projector, items: List<MeasurementVisual>, selected: String?, labelPaint: Paint, bgPaint: Paint) {
    items.forEach { m ->
        if (m.valueMm == null) return@forEach
        val isSel = m.key == selected
        if (selected != null && !isSel) return@forEach
        val color = Nx.hex(m.color)
        val width = if (isSel) 7f else 3.5f
        val ring = m.ring
        var labelAt: Offset? = null
        if (!ring.isNullOrEmpty()) {
            val path = Path()
            ring.forEachIndexed { i, p -> val o = proj.offset(p); if (i == 0) path.moveTo(o.x, o.y) else path.lineTo(o.x, o.y) }
            path.close()
            drawPath(path, color, style = Stroke(width = width))
            labelAt = proj.offset(ring[ring.size / 4])
        } else if (m.startXyz != null && m.endXyz != null) {
            val a = proj.offset(m.startXyz)
            val b = proj.offset(m.endXyz)
            drawLine(color, a, b, strokeWidth = width)
            drawCircle(color, radius = width + 3f, center = a)
            drawCircle(color, radius = width + 3f, center = b)
            labelAt = Offset((a.x + b.x) / 2f, (a.y + b.y) / 2f)
        }
        if (isSel && labelAt != null) drawLabel(m.displayValue, labelAt, color, labelPaint, bgPaint)
    }
}

private fun DrawScope.drawLabel(text: String, at: Offset, color: Color, textPaint: Paint, bgPaint: Paint) {
    drawIntoCanvas { c ->
        val nc = c.nativeCanvas
        val w = textPaint.measureText(text)
        val rect = RectF(at.x - w / 2 - 18f, at.y - 64f, at.x + w / 2 + 18f, at.y - 14f)
        bgPaint.color = android.graphics.Color.WHITE
        nc.drawRoundRect(rect, 24f, 24f, bgPaint)
        bgPaint.color = color.toArgb()
        bgPaint.style = Paint.Style.STROKE
        bgPaint.strokeWidth = 3f
        nc.drawRoundRect(rect, 24f, 24f, bgPaint)
        bgPaint.style = Paint.Style.FILL
        textPaint.color = Nx.Navy.toArgb()
        nc.drawText(text, at.x - w / 2, at.y - 27f, textPaint)
    }
}

private fun DrawScope.drawLandmarks(proj: Projector, landmarks: List<LandmarkVisual>, selected: MeasurementVisual?, paint: Paint) {
    val focus = setOfNotNull(selected?.startLandmark, selected?.endLandmark)
    landmarks.forEach { lm ->
        if (focus.isNotEmpty() && lm.id !in focus) return@forEach
        val o = proj.offset(lm.xyz)
        drawCircle(Color.White, radius = 15f, center = o)
        drawCircle(Nx.Navy, radius = 12f, center = o)
        drawIntoCanvas { c ->
            paint.color = android.graphics.Color.WHITE
            paint.textSize = 18f
            val t = lm.index.toString()
            c.nativeCanvas.drawText(t, o.x - paint.measureText(t) / 2, o.y + 6f, paint)
            paint.textSize = 34f
        }
    }
}

private fun DrawScope.drawScaleBar(scale: Float, paint: Paint) {
    val len = 50f * scale
    val x0 = 32f
    val y = size.height - 36f
    drawLine(Nx.Navy, Offset(x0, y), Offset(x0 + len, y), strokeWidth = 5f)
    drawLine(Nx.Navy, Offset(x0, y - 10f), Offset(x0, y + 10f), strokeWidth = 4f)
    drawLine(Nx.Navy, Offset(x0 + len, y - 10f), Offset(x0 + len, y + 10f), strokeWidth = 4f)
    drawIntoCanvas { c ->
        paint.color = Nx.Navy.toArgb()
        paint.textSize = 26f
        c.nativeCanvas.drawText("5 cm", x0 + len + 12f, y + 9f, paint)
        paint.textSize = 34f
    }
}
