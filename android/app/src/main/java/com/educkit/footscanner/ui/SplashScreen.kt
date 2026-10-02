package com.educkit.footscanner.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.educkit.footscanner.R
import kotlinx.coroutines.delay

@Composable
fun SplashScreen(onFinished: () -> Unit) {
    LaunchedEffect(Unit) {
        delay(1800)
        onFinished()
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
            modifier = Modifier.padding(24.dp),
        ) {
            Image(
                painter = painterResource(id = R.drawable.nikonyx_logo),
                contentDescription = "Nikonyx",
                modifier = Modifier.size(160.dp),
            )
            Text(
                text = "Nikonyx",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "Foot Scan · Test Module",
                color = MaterialTheme.colorScheme.secondary,
            )
            CircularProgressIndicator(
                modifier = Modifier
                    .padding(top = 12.dp)
                    .size(28.dp),
                strokeWidth = 3.dp,
            )
        }
    }
}

@Composable
fun NikonyxLoadingHeader(
    title: String,
    subtitle: String,
    progress: Float,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Image(
            painter = painterResource(id = R.drawable.nikonyx_logo),
            contentDescription = "Nikonyx",
            modifier = Modifier.size(88.dp),
        )
        Text("Nikonyx", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
        Text(title)
        Text(subtitle, color = MaterialTheme.colorScheme.secondary)
        LinearProgressIndicator(
            progress = { progress.coerceIn(0f, 1f) },
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 8.dp),
        )
    }
}
