package com.educkit.footscanner

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.educkit.footscanner.ui.FootScannerApp
import com.educkit.footscanner.ui.theme.FootScannerTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            FootScannerTheme {
                FootScannerApp()
            }
        }
    }
}
