package com.educkit.footscanner.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Scheme = darkColorScheme(
    primary = Color(0xFF7CDECB),
    onPrimary = Color(0xFF00382F),
    secondary = Color(0xFFB5C9C3),
    background = Color(0xFF0F1412),
    surface = Color(0xFF171D1B),
    onBackground = Color(0xFFE2E8E5),
    onSurface = Color(0xFFE2E8E5),
)

@Composable
fun FootScannerTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Scheme, content = content)
}
