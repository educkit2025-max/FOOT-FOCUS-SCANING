package com.educkit.footscanner.network

import com.squareup.moshi.Json

data class GuideCheck(val check: String, val state: String, val message: String)

data class GuideOverlays(
    val paper: List<List<Double>>? = null,
    @Json(name = "heel_line") val heelLine: List<List<Double>>? = null,
    @Json(name = "foot_contour") val footContour: List<List<Double>>? = null,
)

data class GuideResponse(
    @Json(name = "guidance_state") val guidanceState: String = "RED",
    val message: String = "",
    val checks: List<GuideCheck> = emptyList(),
    val overlays: GuideOverlays = GuideOverlays(),
    @Json(name = "camera_tilt_deg") val cameraTiltDeg: Double? = null,
    @Json(name = "quality_score") val qualityScore: Int = 0,
)

data class StageItem(val id: String, val label: String, val state: String)

data class RetakeItem(val view: String, val reasons: List<String>)

data class MeasurementVisual(
    val key: String,
    val number: Int,
    val measurement: String,
    val view: String,
    @Json(name = "value_mm") val valueMm: Double? = null,
    @Json(name = "display_value") val displayValue: String = "",
    @Json(name = "start_landmark") val startLandmark: String? = null,
    @Json(name = "end_landmark") val endLandmark: String? = null,
    @Json(name = "start_xyz") val startXyz: List<Double>? = null,
    @Json(name = "end_xyz") val endXyz: List<Double>? = null,
    val ring: List<List<Double>>? = null,
    val confidence: Double = 0.0,
    val status: String = "",
    val color: String = "#147DFF",
    val reason: String? = null,
)

data class LandmarkVisual(val id: String, val index: Int, val label: String, val xyz: List<Double>)

data class Visualization(val landmarks: List<LandmarkVisual> = emptyList())

data class FootResult(
    val measurements: List<MeasurementVisual> = emptyList(),
    val visualization: Visualization? = null,
    @Json(name = "mesh_available") val meshAvailable: Boolean = false,
    @Json(name = "mesh_method") val meshMethod: String? = null,
    @Json(name = "mesh_scale") val meshScale: String? = null,
    @Json(name = "mesh_messages") val meshMessages: List<String> = emptyList(),
    @Json(name = "volume_cm3") val volumeCm3: Double? = null,
)

data class Footwear(
    @Json(name = "eu_size") val euSize: Double? = null,
    @Json(name = "mondopoint_mm") val mondopointMm: Int? = null,
    @Json(name = "width_category") val widthCategory: String? = null,
    @Json(name = "instep_category") val instepCategory: String? = null,
    @Json(name = "heel_category") val heelCategory: String? = null,
    val provisional: Boolean = true,
    val notes: List<String> = emptyList(),
)

data class ScanSummary(
    @Json(name = "length_cm") val lengthCm: Double? = null,
    @Json(name = "width_cm") val widthCm: Double? = null,
    @Json(name = "length_status") val lengthStatus: String? = null,
    @Json(name = "width_status") val widthStatus: String? = null,
    @Json(name = "scan_confidence") val scanConfidence: Double = 0.0,
)

data class FramesUsed(
    val top: Int = 0,
    @Json(name = "top_tried") val topTried: Int = 0,
    val side: Int = 0,
    @Json(name = "side_tried") val sideTried: Int = 0,
)

data class VideoScanResult(
    @Json(name = "scan_id") val scanId: String = "",
    val foot: String = "right",
    val status: String = "",
    val accepted: Boolean = false,
    val retake: List<RetakeItem> = emptyList(),
    val warnings: List<String> = emptyList(),
    val summary: ScanSummary? = null,
    @Json(name = "foot_result") val footResult: FootResult? = null,
    val footwear: Footwear? = null,
    @Json(name = "frames_used") val framesUsed: FramesUsed? = null,
    val error: String? = null,
)

data class FootMesh(
    val vertices: List<List<Double>> = emptyList(),
    val faces: List<List<Int>> = emptyList(),
    val colors: List<List<Int>>? = null,
    val method: String? = null,
    @Json(name = "scale_source") val scaleSource: String? = null,
)
