package com.educkit.footscanner.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

object Nx {
    val Navy = Color(0xFF061B4E)
    val Blue = Color(0xFF075AC8)
    val MeasureBlue = Color(0xFF147DFF)
    val Background = Color(0xFFF6FAFF)
    val Card = Color(0xFFFFFFFF)
    val Border = Color(0xFFD7E6F7)
    val Green = Color(0xFF24C875)
    val Red = Color(0xFFEF4444)
    val Amber = Color(0xFFF5A623)
    val Muted = Color(0xFF5B6B85)
    val PaleRed = Color(0xFFFDECEC)
    val PaleBlue = Color(0xFFEAF3FF)

    fun state(state: String?): Color = when (state) {
        "GREEN", "final", "done" -> Green
        "AMBER", "estimated" -> Amber
        "RED", "additional_scan_required", "failed" -> Red
        else -> Muted
    }

    fun hex(value: String?, fallback: Color = MeasureBlue): Color =
        runCatching { Color(android.graphics.Color.parseColor(value)) }.getOrDefault(fallback)
}

private val Scheme = lightColorScheme(
    primary = Nx.Blue,
    onPrimary = Color.White,
    secondary = Nx.Muted,
    background = Nx.Background,
    surface = Nx.Card,
    onBackground = Nx.Navy,
    onSurface = Nx.Navy,
    outline = Nx.Border,
    error = Nx.Red,
)

@Composable
fun FootScannerTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Scheme, content = content)
}
