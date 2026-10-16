package com.educkit.footscanner.ui.nx

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.educkit.footscanner.ui.theme.Nx

private val ButtonShape = RoundedCornerShape(20.dp)

@Composable
fun NxPrimaryButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true) {
    val brush = if (enabled) Brush.horizontalGradient(listOf(Nx.Navy, Nx.Blue))
    else Brush.horizontalGradient(listOf(Nx.Border, Nx.Border))
    Box(
        modifier
            .fillMaxWidth()
            .height(56.dp)
            .clip(ButtonShape)
            .background(brush)
            .clickable(enabled = enabled, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, color = if (enabled) Color.White else Nx.Muted, fontWeight = FontWeight.SemiBold, fontSize = 16.sp)
    }
}

@Composable
fun NxSecondaryButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true) {
    Box(
        modifier
            .fillMaxWidth()
            .height(54.dp)
            .clip(ButtonShape)
            .background(Nx.Card)
            .border(1.5.dp, if (enabled) Nx.Navy else Nx.Border, ButtonShape)
            .clickable(enabled = enabled, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, color = if (enabled) Nx.Navy else Nx.Muted, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
    }
}

@Composable
fun NxRetakeButton(text: String = "Retake", onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier
            .height(40.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(Nx.PaleRed)
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(text, color = Nx.Red, fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
    }
}

@Composable
fun NxCard(modifier: Modifier = Modifier, borderColor: Color = Nx.Border, content: @Composable ColumnScope.() -> Unit) {
    Column(
        modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(Nx.Card)
            .border(1.dp, borderColor, RoundedCornerShape(20.dp))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
        content = content,
    )
}

@Composable
fun NxTopBar(title: String, subtitle: String? = null, onBack: (() -> Unit)? = null) {
    Row(
        Modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        if (onBack != null) {
            Box(
                Modifier
                    .size(40.dp)
                    .clip(CircleShape)
                    .background(Nx.Card)
                    .border(1.dp, Nx.Border, CircleShape)
                    .clickable(onClick = onBack),
                contentAlignment = Alignment.Center,
            ) { Text("‹", color = Nx.Navy, fontSize = 24.sp, fontWeight = FontWeight.Bold) }
        }
        Column(Modifier.weight(1f)) {
            Text(title, color = Nx.Navy, fontWeight = FontWeight.Bold, fontSize = 20.sp)
            if (subtitle != null) Text(subtitle, color = Nx.Muted, fontSize = 13.sp)
        }
    }
}

@Composable
fun StatusDot(color: Color, size: Int = 10) {
    Box(
        Modifier
            .size(size.dp)
            .clip(CircleShape)
            .background(color),
    )
}

@Composable
fun StatusPill(text: String, color: Color) {
    Row(
        Modifier
            .clip(RoundedCornerShape(12.dp))
            .background(color.copy(alpha = 0.12f))
            .padding(horizontal = 10.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        StatusDot(color, 8)
        Text(text, color = color, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
fun SegmentedSwitch(options: List<Pair<String, String>>, selected: String, onSelect: (String) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Nx.PaleBlue)
            .padding(4.dp),
    ) {
        options.forEach { (id, label) ->
            val on = id == selected
            Box(
                Modifier
                    .weight(1f)
                    .height(40.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(if (on) Nx.Card else Color.Transparent)
                    .clickable { onSelect(id) },
                contentAlignment = Alignment.Center,
            ) {
                Text(label, color = if (on) Nx.Navy else Nx.Muted, fontWeight = if (on) FontWeight.Bold else FontWeight.Medium, fontSize = 14.sp)
            }
        }
    }
}
