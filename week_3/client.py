"""
Meeting Audio Transcription & Summarization Client
===================================================
A local Gradio interface that connects to a Google Colab FastAPI backend.
Transcribes audio recordings, generates executive meeting summaries,
and exports the results as PDF and Markdown reports.
"""

import os
import io
import time
import tempfile
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

import requests
import gradio as gr
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors

# -----------------------------------------------------------------------------
# Global Configuration & Defaults
# -----------------------------------------------------------------------------
DEFAULT_BACKEND_URL: str = "https://plenty-panhandle-massive.ngrok-free.dev"
REQUEST_TIMEOUT_SECONDS: int = 300  # 5 minutes for processing longer audio files


# -----------------------------------------------------------------------------
# Backend API Communication Logic
# -----------------------------------------------------------------------------
def check_backend_health(backend_url: str) -> Tuple[str, str]:
    """
    Pings the Colab FastAPI backend health endpoint to verify connectivity.

    Args:
        backend_url: Base URL of the backend (e.g. https://xxxx.ngrok-free.app).

    Returns:
        Tuple containing a formatted Markdown status message and a status color indicator.
    """
    clean_url = backend_url.strip().rstrip("/")
    if not clean_url:
        return "⚠️ **Please enter a valid Backend URL.**", "color: #eab308;"

    health_url = f"{clean_url}/health"
    try:
        response = requests.get(health_url, timeout=10)
        if response.status_code == 200:
            data: Dict[str, Any] = response.json()
            mode = data.get("mode", "unknown").upper()
            asr_model = data.get("model_asr", "N/A")
            sum_model = data.get("model_summarizer", "N/A")
            msg = (
                f"✅ **Backend Connected Successfully!**\n\n"
                f"- **Inference Mode:** `{mode}`\n"
                f"- **ASR Model:** `{asr_model}`\n"
                f"- **Summarizer:** `{sum_model}`"
            )
            return msg, "color: #22c55e;"
        else:
            return (
                f"❌ **Backend returned HTTP {response.status_code}**: {response.text}",
                "color: #ef4444;",
            )
    except requests.exceptions.Timeout:
        return "❌ **Connection Timeout**: The backend took longer than 10s to respond.", "color: #ef4444;"
    except requests.exceptions.ConnectionError:
        return "❌ **Connection Error**: Could not reach backend URL. Is the Colab cell running?", "color: #ef4444;"
    except Exception as exc:
        return f"❌ **Error connecting to backend**: {str(exc)}", "color: #ef4444;"


def call_transcription_api(
    audio_path: Optional[str],
    backend_url: str,
    min_summary_length: int,
    max_summary_length: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[str, str, str, Optional[str], Optional[str]]:
    """
    Sends the audio recording to the FastAPI backend for transcription and summarization.

    Args:
        audio_path: Local filesystem path to the uploaded audio file.
        backend_url: Base URL of the backend server.
        min_summary_length: Minimum summary token count.
        max_summary_length: Maximum summary token count.
        progress: Gradio Progress tracker.

    Returns:
        Tuple of:
            - summary_text (str)
            - transcript_text (str)
            - metadata_markdown (str)
            - pdf_download_path (Optional[str])
            - md_download_path (Optional[str])
    """
    # 1. Input Validation
    if not audio_path or not os.path.exists(audio_path):
        raise gr.Error("Please upload or record an audio file before processing.")

    clean_url = backend_url.strip().rstrip("/")
    if not clean_url:
        raise gr.Error("Backend URL cannot be empty. Please enter your Colab ngrok URL.")

    endpoint_url = f"{clean_url}/process"

    # 2. Prepare Payload and Multi-part Form Request
    progress(0.1, desc="Preparing audio payload...")
    filename = os.path.basename(audio_path)
    form_data = {
        "min_length": str(int(min_summary_length)),
        "max_length": str(int(max_summary_length)),
    }

    try:
        progress(0.2, desc="Uploading audio to backend & running inference...")
        start_time = time.time()

        with open(audio_path, "rb") as audio_file:
            files = {"file": (filename, audio_file, "audio/mpeg")}
            response = requests.post(
                endpoint_url,
                data=form_data,
                files=files,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        client_total_time = round(time.time() - start_time, 2)

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                pass
            raise gr.Error(f"Backend Server Error (HTTP {response.status_code}): {error_detail}")

        progress(0.8, desc="Parsing model responses...")
        result_data: Dict[str, Any] = response.json()

        transcription: str = result_data.get("transcription", "").strip()
        summary: str = result_data.get("summary", "").strip()
        audio_duration: float = result_data.get("audio_duration_seconds", 0.0)
        metrics: Dict[str, Any] = result_data.get("metrics", {})

        asr_time = metrics.get("transcription_time_seconds", "N/A")
        sum_time = metrics.get("summarization_time_seconds", "N/A")
        backend_total_time = metrics.get("total_time_seconds", "N/A")

        # 3. Construct Metadata Dashboard
        metadata_md = f"""
### 📊 Processing Metrics & Telemetry
| Metric | Value |
| :--- | :--- |
| **Audio Duration** | `{audio_duration:.2f}s` ({round(audio_duration / 60, 2)} min) |
| **ASR Inference Time** | `{asr_time}s` |
| **Summarization Inference Time** | `{sum_time}s` |
| **Backend Total Time** | `{backend_total_time}s` |
| **Client Round-Trip Time** | `{client_total_time}s` |
| **Transcription Length** | `{len(transcription.split())}` words / `{len(transcription)}` chars |
| **Summary Length** | `{len(summary.split())}` words / `{len(summary)}` chars |
"""

        # 4. Generate Exportable Files
        progress(0.9, desc="Generating PDF and Markdown export files...")
        pdf_path = create_pdf_report(
            summary=summary,
            transcription=transcription,
            audio_duration=audio_duration,
            client_time=client_total_time,
            backend_metrics=metrics,
        )

        md_path = create_markdown_report(
            summary=summary,
            transcription=transcription,
            audio_duration=audio_duration,
            client_time=client_total_time,
            backend_metrics=metrics,
        )

        progress(1.0, desc="Done!")
        return summary, transcription, metadata_md, pdf_path, md_path

    except requests.exceptions.Timeout:
        raise gr.Error(
            f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s. "
            "For very long audio files, consider increasing REQUEST_TIMEOUT_SECONDS or using a smaller model."
        )
    except requests.exceptions.ConnectionError:
        raise gr.Error(f"Connection failed to '{clean_url}'. Please check if your Colab ngrok tunnel is active.")
    except Exception as exc:
        raise gr.Error(f"Processing error: {str(exc)}")


# -----------------------------------------------------------------------------
# Export File Generation (PDF & Markdown)
# -----------------------------------------------------------------------------
def create_pdf_report(
    summary: str,
    transcription: str,
    audio_duration: float,
    client_time: float,
    backend_metrics: Dict[str, Any],
) -> str:
    """
    Generates an executive PDF report containing the meeting summary and transcription.

    Args:
        summary: Generated executive summary.
        transcription: Full transcribed text.
        audio_duration: Duration of input audio in seconds.
        client_time: Total round-trip time in seconds.
        backend_metrics: Dictionary of backend timing metrics.

    Returns:
        Local path to the generated temporary PDF file.
    """
    temp_dir = tempfile.gettempdir()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = os.path.join(temp_dir, f"meeting_summary_{timestamp_str}.pdf")

    # Document setup with standard letter page size & 0.75-inch margins
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        name="DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        name="DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15,
    )

    section_heading = ParagraphStyle(
        name="SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        name="BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=10,
    )

    code_style = ParagraphStyle(
        name="TranscriptStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8,
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("Executive Meeting Intelligence Report", title_style))
    story.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | Automated ASR & Summarization Pipeline",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=15))

    # Metadata Table
    meta_data = [
        [
            Paragraph("<b>Audio Duration:</b>", body_style),
            Paragraph(f"{audio_duration:.2f}s ({round(audio_duration/60, 2)} min)", body_style),
            Paragraph("<b>ASR Time:</b>", body_style),
            Paragraph(f"{backend_metrics.get('transcription_time_seconds', 'N/A')}s", body_style),
        ],
        [
            Paragraph("<b>Summary Time:</b>", body_style),
            Paragraph(f"{backend_metrics.get('summarization_time_seconds', 'N/A')}s", body_style),
            Paragraph("<b>Total Client Time:</b>", body_style),
            Paragraph(f"{client_time:.2f}s", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[100, 150, 100, 150])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 15))

    # Section 1: Executive Summary
    story.append(Paragraph("📌 Executive Summary", section_heading))
    # Replace newlines with Paragraphs
    summary_paragraphs = summary.split("\n")
    for para in summary_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))
    story.append(Spacer(1, 10))

    # Section 2: Full Meeting Transcript
    story.append(Paragraph("📝 Full Meeting Transcription", section_heading))
    transcript_paragraphs = transcription.split("\n")
    for para in transcript_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), code_style))

    # Build PDF
    doc.build(story)
    return pdf_filename


def create_markdown_report(
    summary: str,
    transcription: str,
    audio_duration: float,
    client_time: float,
    backend_metrics: Dict[str, Any],
) -> str:
    """
    Generates a structured Markdown report file.

    Args:
        summary: Generated executive summary.
        transcription: Full transcribed text.
        audio_duration: Duration of input audio in seconds.
        client_time: Total round-trip time in seconds.
        backend_metrics: Dictionary of backend timing metrics.

    Returns:
        Local path to the generated temporary Markdown file.
    """
    temp_dir = tempfile.gettempdir()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_filename = os.path.join(temp_dir, f"meeting_summary_{timestamp_str}.md")

    content = f"""---
title: Executive Meeting Intelligence Report
date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
audio_duration_seconds: {audio_duration:.2f}
transcription_time_seconds: {backend_metrics.get('transcription_time_seconds', 'N/A')}
summarization_time_seconds: {backend_metrics.get('summarization_time_seconds', 'N/A')}
total_client_time_seconds: {client_time:.2f}
---

# 🎙️ Executive Meeting Intelligence Report

**Generated on:** {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}  
**Audio Duration:** {audio_duration:.2f}s ({round(audio_duration / 60, 2)} minutes)  
**Total Processing Time:** {client_time:.2f}s  

---

## 📌 Executive Summary

{summary}

---

## 📝 Full Meeting Transcription

{transcription}

---
*Report generated automatically by Speech-to-Text & Summarization Pipeline Client.*
"""

    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(content)

    return md_filename


# -----------------------------------------------------------------------------
# Gradio UI Construction
# -----------------------------------------------------------------------------
def build_interface() -> gr.Blocks:
    """
    Constructs and styles the Gradio Blocks UI application.
    """
    with gr.Blocks(title="Audio Meeting Summarizer") as demo:
        # Header section
        gr.Markdown(
            """
            # 🎙️ Speech-to-Text & Meeting Summarization Pipeline
            ### Upload audio recordings, generate executive summaries via Google Colab GPU backend, and export reports.
            """
        )

        # Connection Accordion
        with gr.Accordion("⚙️ Backend Connection Settings", open=True):
            with gr.Row():
                backend_url_input = gr.Textbox(
                    value=DEFAULT_BACKEND_URL,
                    label="Colab Backend URL (ngrok public URL)",
                    placeholder="https://xxxx-xx-xx-xx.ngrok-free.app",
                    scale=4,
                )
                test_conn_btn = gr.Button("🔗 Test Connection", variant="secondary", scale=1)
            conn_status = gr.Markdown("ℹ️ *Enter your ngrok URL and click 'Test Connection' to verify.*")

        with gr.Row():
            # Left Column: Audio Input & Inference Controls
            with gr.Column(scale=5):
                gr.Markdown("### 1. Upload Meeting Audio")
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Audio File (WAV, MP3, M4A, OGG, FLAC)",
                )

                with gr.Accordion("🛠️ Summarization Parameters", open=False):
                    min_len_slider = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=5,
                        label="Minimum Summary Length (tokens)",
                    )
                    max_len_slider = gr.Slider(
                        minimum=50,
                        maximum=300,
                        value=150,
                        step=10,
                        label="Maximum Summary Length (tokens)",
                    )

                process_btn = gr.Button(
                    "🚀 Transcribe & Summarize",
                    variant="primary",
                    size="lg",
                )

            # Right Column: Output Tabs & Exports
            with gr.Column(scale=7):
                gr.Markdown("### 2. Generated Intelligence")
                with gr.Tabs():
                    with gr.TabItem("📌 Executive Summary"):
                        summary_output = gr.Markdown(
                            value="*The executive summary will appear here once processing completes.*"
                        )

                    with gr.TabItem("📝 Full Transcription"):
                        # Configured with max_lines and autoscroll=False for a dedicated scrollable viewport
                        tb_kwargs = {
                            "label": "Transcribed Meeting Dialogue",
                            "placeholder": "Full transcription will appear here...",
                            "lines": 10,
                            "max_lines": 18,
                            "autoscroll": False,
                        }
                        if "buttons" in gr.Textbox.__init__.__code__.co_varnames:
                            tb_kwargs["buttons"] = ["copy"]
                        elif "show_copy_button" in gr.Textbox.__init__.__code__.co_varnames:
                            tb_kwargs["show_copy_button"] = True

                        transcript_output = gr.Textbox(**tb_kwargs)

                    with gr.TabItem("📊 Execution Telemetry"):
                        telemetry_output = gr.Markdown(
                            value="*Performance telemetry and model timing metrics will appear here.*"
                        )

                # Export Download Section
                gr.Markdown("### 3. Export Reports")
                with gr.Row():
                    pdf_download = gr.DownloadButton(
                        label="📄 Download PDF Report",
                        value=None,
                        variant="secondary",
                        visible=True,
                    )
                    md_download = gr.DownloadButton(
                        label="📝 Download Markdown Report",
                        value=None,
                        variant="secondary",
                        visible=True,
                    )

        # -------------------------------------------------------------------------
        # Event Handlers & Callbacks
        # -------------------------------------------------------------------------
        test_conn_btn.click(
            fn=lambda url: check_backend_health(url)[0],
            inputs=[backend_url_input],
            outputs=[conn_status],
        )

        process_btn.click(
            fn=call_transcription_api,
            inputs=[audio_input, backend_url_input, min_len_slider, max_len_slider],
            outputs=[summary_output, transcript_output, telemetry_output, pdf_download, md_download],
        )

    return demo


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    demo_app = build_interface()
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )
    # Launch locally on port 7860 (theme passed to launch() as required in Gradio 6.0+)
    demo_app.launch(
        theme=theme,
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
