package com.educkit.footscanner.network

import com.educkit.footscanner.BuildConfig
import com.squareup.moshi.Json
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path
import java.io.File
import java.util.concurrent.TimeUnit

data class CreateScanRequest(
    val foot: String,
    val device: Map<String, Any?> = emptyMap(),
    val capture: Map<String, Any?> = mapOf("weight_bearing" to true, "target_fps" to 30, "duration_seconds" to 30),
)

data class CreateScanResponse(
    @Json(name = "scan_id") val scanId: String,
    val status: String,
)

data class StatusResponse(
    @Json(name = "scan_id") val scanId: String,
    val status: String,
    @Json(name = "progress_percent") val progressPercent: Int = 0,
    val stage: String = "",
    val message: String = "",
    @Json(name = "error_code") val errorCode: String? = null,
    @Json(name = "frame_summary") val frameSummary: Map<String, Any?> = emptyMap(),
    val stages: List<StageItem> = emptyList(),
)

data class ProcessRequest(
    @Json(name = "target_frames") val targetFrames: Int = 70,
)

data class BestFrame(
    val rank: Int,
    val position: String,
    val label: String,
    @Json(name = "image_url") val imageUrl: String,
    @Json(name = "time_s") val timeS: Double = 0.0,
    @Json(name = "a4_detected") val a4Detected: Boolean = false,
    @Json(name = "tilt_deg") val tiltDeg: Double? = null,
    @Json(name = "quality_score") val qualityScore: Int = 0,
)

data class FramesResponse(
    @Json(name = "scan_id") val scanId: String,
    @Json(name = "original_frame_count") val originalFrameCount: Int,
    @Json(name = "selected_frame_count") val selectedFrameCount: Int,
    @Json(name = "best_frames") val bestFrames: List<BestFrame> = emptyList(),
)

interface ScannerApi {
    @POST("/api/v1/scans")
    suspend fun createScan(@Body body: CreateScanRequest): CreateScanResponse

    @Multipart
    @POST("/api/v1/scans/{scanId}/video")
    suspend fun uploadVideo(@Path("scanId") scanId: String, @Part file: MultipartBody.Part): StatusResponse

    @POST("/api/v1/scans/{scanId}/process")
    suspend fun process(@Path("scanId") scanId: String, @Body body: ProcessRequest = ProcessRequest()): StatusResponse

    @POST("/api/v1/scans/{scanId}/measure")
    suspend fun measure(@Path("scanId") scanId: String): StatusResponse

    @GET("/api/v1/scans/{scanId}/status")
    suspend fun status(@Path("scanId") scanId: String): StatusResponse

    @GET("/api/v1/scans/{scanId}/frames")
    suspend fun frames(@Path("scanId") scanId: String): FramesResponse

    @GET("/api/v1/scans/{scanId}/result")
    suspend fun result(@Path("scanId") scanId: String): VideoScanResult

    @GET("/api/v1/scans/{scanId}/mesh.json")
    suspend fun mesh(@Path("scanId") scanId: String): FootMesh

    @Multipart
    @POST("/api/v1/guide/frame")
    suspend fun guideFrame(
        @Part file: MultipartBody.Part,
        @Part("view") view: RequestBody,
        @Part("session_id") sessionId: RequestBody,
        @Part("focal_mm") focalMm: RequestBody?,
        @Part("sensor_long_mm") sensorLongMm: RequestBody?,
    ): GuideResponse
}

object ApiClient {
    @Volatile
    private var baseUrl: String = BuildConfig.DEFAULT_API_BASE.trimEnd('/') + "/"

    private val moshi: Moshi = Moshi.Builder()
        .add(KotlinJsonAdapterFactory())
        .build()

    private val okHttp: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .build()

    fun setBaseUrl(url: String) {
        baseUrl = url.trimEnd('/') + "/"
        retrofit = buildRetrofit()
        api = retrofit.create(ScannerApi::class.java)
    }

    fun currentBaseUrl(): String = baseUrl.trimEnd('/')

    fun absolute(path: String): String = currentBaseUrl() + path

    private fun buildRetrofit(): Retrofit = Retrofit.Builder()
        .baseUrl(baseUrl)
        .client(okHttp)
        .addConverterFactory(MoshiConverterFactory.create(moshi))
        .build()

    @Volatile
    private var retrofit: Retrofit = buildRetrofit()

    @Volatile
    var api: ScannerApi = retrofit.create(ScannerApi::class.java)
        private set

    fun videoPart(file: File): MultipartBody.Part =
        MultipartBody.Part.createFormData("file", file.name, file.asRequestBody("video/mp4".toMediaType()))

    fun jpegPart(name: String, bytes: ByteArray): MultipartBody.Part =
        MultipartBody.Part.createFormData(name, "frame.jpg", bytes.toRequestBody("image/jpeg".toMediaType()))

    fun text(value: Any?): RequestBody? = value?.toString()?.toRequestBody("text/plain".toMediaType())
}
