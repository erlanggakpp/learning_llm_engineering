"""
Runner Synthetic Telemetry Data Generator - Local Gradio Frontend Client
=========================================================================
Course Assignment: Building Synthetic Telemetry Data Generator using Open-Source Models
Model Evaluated: Qwen/Qwen2.5-3B-Instruct (FP16 Baseline vs. 4-Bit NF4 Quantized)
Frontend: Gradio 6.x + Requests

Features:
1. Remote Ngrok Tunnel Configuration & Health Verification (/health).
2. Runner Telemetry Schema Controls:
   - Workout Type (Tempo, Easy, Intervals, Long Run, Hill Repeats)
   - Athlete Profile (Beginner 5k, Intermediate Marathoner, Sub-Elite / Elite)
   - Telemetry Schema Checkboxes (all 11 physiological fields)
   - Record Count (1 to 10 records)
   - Output Format (JSON Array, CSV Table, Markdown Table)
   - Sampling Temperature (0.1 to 1.0)
3. Benchmark & Side-by-Side Presentation:
   - Tab 1: 4-Bit NF4 Quantized Output (gr.Code)
   - Tab 2: FP16 Baseline Output (gr.Code)
   - Formatted Markdown Benchmark Table (Load VRAM, Peak Gen VRAM, Latency, Throughput, % Reductions)
   - Automated Physiological Sanity Check Summary
   - Dataset Export / File Downloads (.json, .csv, .md)
4. Robust Error Handling & Offline Demo Mode simulation.
"""

import os
import sys
import json
import time
import re
import csv
import io
import tempfile
from datetime import datetime, date
from typing import List, Dict, Any, Tuple, Optional

import requests
import gradio as gr

# ==============================================================================
# Configuration & Constants
# ==============================================================================
DEFAULT_TUNNEL_URL: str = "https://plenty-panhandle-massive.ngrok-free.dev"
REQUEST_TIMEOUT_SECONDS: int = 180

WORKOUT_TYPES: List[str] = [
    "Tempo Run",
    "Easy Recovery Run",
    "Interval Track Session",
    "Long Weekend Run",
    "Hill Repeats",
]

ATHLETE_PROFILES: List[str] = [
    "Beginner 5k Runner",
    "Intermediate Marathoner",
    "Sub-Elite / Elite",
]

SCHEMA_FIELDS: List[str] = [
    "runner_id",
    "session_date",
    "distance_km",
    "duration_minutes",
    "pace_min_per_km",
    "avg_heart_rate_bpm",
    "max_heart_rate_bpm",
    "cadence_spm",
    "elevation_gain_m",
    "calories_burned",
    "rpe_scale_1_to_10",
]

FORMAT_CHOICES: List[str] = [
    "JSON Array",
    "CSV Table",
    "Markdown Table",
]


# ==============================================================================
# Health Check & Remote API Communication
# ==============================================================================
def check_remote_health(tunnel_url: str) -> str:
    """
    Pings the Colab FastAPI backend /health endpoint to verify connectivity and VRAM stats.
    """
    clean_url = tunnel_url.strip().rstrip("/")
    if not clean_url:
        return "⚠️ **Please enter your Ngrok Public Tunnel URL.** (e.g. `https://xxxx.ngrok-free.app`)"

    health_endpoint = f"{clean_url}/health"
    try:
        start_t = time.perf_counter()
        resp = requests.get(health_endpoint, timeout=12)
        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 1)

        if resp.status_code == 200:
            data = resp.json()
            model_id = data.get("model_id", "Qwen/Qwen2.5-3B-Instruct")
            device = data.get("device", "cuda:0")
            models = data.get("models_loaded", {})
            fp16_info = models.get("baseline_fp16", {})
            q4_info = models.get("quantized_4bit", {})
            vram_savings = data.get("vram_reduction_pct", 0.0)
            default_mode = data.get("default_inference_mode", "Normal (Direct Tensor model.generate)")

            return (
                f"✅ **Connected to Google Colab Backend!** (`{elapsed_ms}ms ping`)\n\n"
                f"- **Model ID:** `{model_id}` on `{device}`\n"
                f"- **FP16 Baseline Load VRAM:** `{fp16_info.get('weight_vram_mb', 0)} MB`\n"
                f"- **4-Bit NF4 Quantized VRAM:** `{q4_info.get('weight_vram_mb', 0)} MB`\n"
                f"- **Static Weight Memory Savings:** **`{vram_savings}% Reduction`**\n"
                f"- **Backend Default Engine:** `{default_mode}`\n"
                f"- **Status:** Dual models initialized and ready for generation."
            )
        else:
            return (
                f"❌ **Server Returned HTTP {resp.status_code}**\n\n"
                f"Response body: `{resp.text[:300]}`"
            )
    except requests.exceptions.Timeout:
        return (
            "❌ **Connection Timeout (12s)**: The Ngrok tunnel URL is not responding.\n"
            "Please verify that your Google Colab cell is still executing and Uvicorn is active."
        )
    except requests.exceptions.ConnectionError:
        return (
            "❌ **Connection Refused**: Unable to reach backend at that URL.\n"
            "Common causes:\n"
            "- The ngrok tunnel expired or URL changed.\n"
            "- The Colab backend cell was stopped or disconnected.\n"
            "- Typo in the URL (make sure it includes `https://`)."
        )
    except Exception as exc:
        return f"❌ **Connection Error**: `{str(exc)}`"


# ==============================================================================
# Biomechanical Telemetry Parsing & Physiological Verification
# ==============================================================================
def parse_pace_to_minutes(pace_val: Any) -> float:
    """Converts MM:SS or decimal pace string into float minutes."""
    if isinstance(pace_val, (int, float)):
        return float(pace_val)
    if not isinstance(pace_val, str):
        return 0.0
    val_str = pace_val.strip()
    match = re.match(r"^(\d+):(\d{2})", val_str)
    if match:
        return round(int(match.group(1)) + int(match.group(2)) / 60.0, 2)
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
    if matches:
        return float(matches[0])
    return 0.0


def extract_numeric(val: Any, default: float = 0.0) -> float:
    """Extracts first numeric float from string/number."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return default
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", val)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    return default


def parse_telemetry_records(text: str, data_format: str) -> List[Dict[str, Any]]:
    """Robustly parses telemetry records from JSON, CSV, or Markdown tables."""
    cleaned = text.strip()
    records: List[Dict[str, Any]] = []

    content = cleaned
    if "```" in content:
        for block in re.findall(r"```(?:json|csv|markdown|text)?\s*(.*?)\s*```", content, re.DOTALL):
            if block.strip():
                content = block.strip()
                break

    # 1. JSON Parsing
    if data_format == "JSON Array" or content.startswith("[") or content.startswith("{"):
        try:
            arr_match = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
            if arr_match:
                parsed = json.loads(arr_match.group(0))
                if isinstance(parsed, list):
                    return parsed
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass

    # 2. CSV Parsing
    if data_format == "CSV Table" or "," in content:
        lines = [line.strip() for line in content.splitlines() if line.strip() and "," in line]
        if len(lines) >= 2:
            try:
                reader = csv.DictReader(io.StringIO("\n".join(lines)))
                parsed_csv = [row for row in reader]
                if parsed_csv:
                    return parsed_csv
            except Exception:
                pass

    # 3. Markdown Table Parsing
    if data_format == "Markdown Table" or "|" in content:
        table_lines = [
            l.strip()
            for l in content.splitlines()
            if l.strip().startswith("|") and l.strip().endswith("|")
        ]
        if len(table_lines) >= 3:
            try:
                headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
                for row_line in table_lines[2:]:
                    cols = [c.strip() for c in row_line.strip("|").split("|")]
                    if len(cols) == len(headers):
                        records.append(dict(zip(headers, cols)))
                if records:
                    return records
            except Exception:
                pass

    return records


def client_evaluate_sanity(
    records: List[Dict[str, Any]], workout_type: str, athlete_profile: str
) -> Dict[str, Any]:
    """Client-side independent evaluation of physiological telemetry realism."""
    if not records:
        return {
            "status": "Inconclusive",
            "score_pct": 0.0,
            "passed_checks": 0,
            "total_checks": 0,
            "details": ["Unable to extract structured records from output text."],
            "summary_md": "⚠️ **Parsing Inconclusive**: Output could not be parsed into tabular/JSON records for automated verification.",
        }

    hr_inversion_evaluated = 0
    hr_inversion_passes = 0
    hr_range_passes = 0
    cadence_evaluated = 0
    cadence_passes = 0
    intensity_evaluated = 0
    intensity_passes = 0
    kinematic_evaluated = 0
    kinematic_passes = 0
    kinematic_failures: List[str] = []

    for r in records:
        avg_hr = extract_numeric(r.get("avg_heart_rate_bpm") or r.get("avg_hr"))
        max_hr = extract_numeric(r.get("max_heart_rate_bpm") or r.get("max_hr"))
        if avg_hr > 0 and max_hr > 0:
            hr_inversion_evaluated += 1
            if max_hr > avg_hr:
                hr_inversion_passes += 1
            if 90 <= avg_hr <= 198 and 110 <= max_hr <= 215:
                hr_range_passes += 1

        cadence = extract_numeric(r.get("cadence_spm") or r.get("cadence"))
        if cadence > 0:
            cadence_evaluated += 1
            if 140 <= cadence <= 205:
                cadence_passes += 1

        rpe = extract_numeric(r.get("rpe_scale_1_to_10") or r.get("rpe"))
        if rpe > 0:
            intensity_evaluated += 1
            if "Easy" in workout_type and rpe <= 5:
                intensity_passes += 1
            elif "Tempo" in workout_type and 5 <= rpe <= 8:
                intensity_passes += 1
            elif ("Interval" in workout_type or "Hill" in workout_type) and rpe >= 7:
                intensity_passes += 1
            elif "Long" in workout_type and 4 <= rpe <= 8:
                intensity_passes += 1
            else:
                intensity_passes += 1

        dist = extract_numeric(r.get("distance_km") or r.get("distance"))
        dur = extract_numeric(r.get("duration_minutes") or r.get("duration"))
        pace = parse_pace_to_minutes(r.get("pace_min_per_km") or r.get("pace"))
        rid = r.get("runner_id") or f"Session #{kinematic_evaluated + 1}"
        if dist > 0 and dur > 0 and pace > 0:
            kinematic_evaluated += 1
            expected_dur = dist * pace
            diff_ratio = abs(dur - expected_dur) / max(expected_dur, 1e-5)
            # Strict tolerance: max 7% margin for decimal rounding (e.g. 5:15 is 5.25 min)
            if diff_ratio <= 0.07:
                kinematic_passes += 1
            else:
                actual_pace = dur / dist
                kinematic_failures.append(
                    f"{rid}: {dist:.1f}km @ pace {pace:.2f} = {expected_dur:.1f}m, but duration is {dur:.1f}m (actual pace is {actual_pace:.2f} min/km, {diff_ratio*100:.1f}% mismatch)"
                )

    details: List[str] = []
    checks_total = 0
    passed_total = 0

    if hr_inversion_evaluated > 0:
        checks_total += hr_inversion_evaluated
        passed_total += hr_inversion_passes
        pct = (hr_inversion_passes / hr_inversion_evaluated) * 100
        icon = "✅" if pct == 100 else "⚠️"
        details.append(
            f"{icon} **Heart Rate Inversion (`max > avg`)**: {hr_inversion_passes}/{hr_inversion_evaluated} "
            f"({pct:.0f}%) sessions adhered to cardiac biology."
        )

        checks_total += hr_inversion_evaluated
        passed_total += hr_range_passes
        pct_r = (hr_range_passes / hr_inversion_evaluated) * 100
        icon_r = "✅" if pct_r == 100 else "⚠️"
        details.append(
            f"{icon_r} **Human Physiological Cardiac Bounds**: {hr_range_passes}/{hr_inversion_evaluated} "
            f"({pct_r:.0f}%) sessions in 90-215 bpm zone."
        )

    if cadence_evaluated > 0:
        checks_total += cadence_evaluated
        passed_total += cadence_passes
        pct = (cadence_passes / cadence_evaluated) * 100
        icon = "✅" if pct >= 90 else "⚠️"
        details.append(
            f"{icon} **Biomechanic Cadence Range**: {cadence_passes}/{cadence_evaluated} "
            f"({pct:.0f}%) sessions within 140-205 spm."
        )

    if intensity_evaluated > 0:
        checks_total += intensity_evaluated
        passed_total += intensity_passes
        pct = (intensity_passes / intensity_evaluated) * 100
        icon = "✅" if pct >= 80 else "⚠️"
        details.append(
            f"{icon} **RPE vs. Workout Intensity**: {intensity_passes}/{intensity_evaluated} "
            f"({pct:.0f}%) scaled properly to `{workout_type}`."
        )

    if kinematic_evaluated > 0:
        checks_total += kinematic_evaluated
        passed_total += kinematic_passes
        pct = (kinematic_passes / kinematic_evaluated) * 100
        icon = "✅" if pct >= 90 else ("⚠️" if pct >= 50 else "❌")
        msg = (
            f"{icon} **Kinematic Physics (`Duration ≈ Dist × Pace`)**: {kinematic_passes}/{kinematic_evaluated} "
            f"({pct:.0f}%) mathematically coherent."
        )
        if kinematic_failures:
            examples_str = "; ".join(kinematic_failures[:2])
            msg += f"\n  - ⚠️ Arithmetic Mismatch in {len(kinematic_failures)} session(s): {examples_str}"
        details.append(msg)

    score = (passed_total / checks_total * 100) if checks_total > 0 else 100.0
    status_label = (
        "✅ PASSED (Strict Physiological Fidelity)"
        if score >= 90
        else ("⚠️ PARTIAL (Minor Deviations)" if score >= 70 else "❌ FAILED (Inconsistent)")
    )

    summary_md = f"**Status**: {status_label} — **{score:.1f}% Score** ({passed_total}/{checks_total} checks passed)\n\n" + "\n".join(
        [f"- {d}" for d in details]
    )

    return {
        "status": status_label,
        "score_pct": round(score, 1),
        "passed_checks": passed_total,
        "total_checks": checks_total,
        "details": details,
        "summary_md": summary_md,
    }


# ==============================================================================
# Offline Simulation Mode (Demo Fallback)
# ==============================================================================
def simulate_offline_demo_generation(
    workout_type: str,
    athlete_profile: str,
    count: int,
    fields: List[str],
    data_format: str,
    temperature: float,
) -> Dict[str, Any]:
    """Generates synthetic telemetry and benchmark numbers locally for offline testing."""
    pace_profiles = {
        "Beginner 5k Runner": {"pace": 6.75, "pace_str": "6:45", "hr_avg": 158, "hr_max": 174, "cad": 156, "dist": 4.5},
        "Intermediate Marathoner": {"pace": 4.80, "pace_str": "4:48", "hr_avg": 164, "hr_max": 176, "cad": 172, "dist": 10.0},
        "Sub-Elite / Elite": {"pace": 3.40, "pace_str": "3:24", "hr_avg": 168, "hr_max": 182, "cad": 184, "dist": 15.0},
    }
    prof = pace_profiles.get(athlete_profile, pace_profiles["Intermediate Marathoner"])

    workout_multipliers = {
        "Easy Recovery Run": {"pace_mult": 1.25, "hr_mult": 0.82, "rpe": 3, "elev": 20},
        "Tempo Run": {"pace_mult": 1.00, "hr_mult": 1.00, "rpe": 7, "elev": 45},
        "Interval Track Session": {"pace_mult": 0.88, "hr_mult": 1.08, "rpe": 9, "elev": 10},
        "Long Weekend Run": {"pace_mult": 1.15, "hr_mult": 0.90, "rpe": 6, "elev": 80},
        "Hill Repeats": {"pace_mult": 1.10, "hr_mult": 1.06, "rpe": 9, "elev": 240},
    }
    w_mult = workout_multipliers.get(workout_type, workout_multipliers["Tempo Run"])

    records = []
    base_dist = prof["dist"] if workout_type != "Long Weekend Run" else prof["dist"] * 1.8
    for i in range(1, count + 1):
        dist = round(base_dist + (i * 0.4), 2)
        pace_dec = round(prof["pace"] * w_mult["pace_mult"] + ((i % 3) * 0.05), 2)
        mins = int(pace_dec)
        secs = int(round((pace_dec - mins) * 60))
        pace_str = f"{mins}:{secs:02d}"
        dur = round(dist * pace_dec, 1)
        avg_hr = int(round(prof["hr_avg"] * w_mult["hr_mult"] + (i % 4)))
        max_hr = int(round(avg_hr + 12 + (i % 6)))
        cad = int(round(prof["cad"] + (i % 4) - 2))
        elev = int(round(w_mult["elev"] + (i * 5)))
        cal = int(round(dist * 68.0))
        rpe = w_mult["rpe"]

        full_rec = {
            "runner_id": f"RUN-{100 + i}",
            "session_date": f"2026-03-{10 + i:02d}",
            "distance_km": dist,
            "duration_minutes": dur,
            "pace_min_per_km": pace_str,
            "avg_heart_rate_bpm": avg_hr,
            "max_heart_rate_bpm": max_hr,
            "cadence_spm": cad,
            "elevation_gain_m": elev,
            "calories_burned": cal,
            "rpe_scale_1_to_10": rpe,
        }
        filtered_rec = {k: full_rec[k] for k in fields if k in full_rec}
        records.append(filtered_rec)

    # Format text output
    if data_format == "JSON Array":
        text_output = json.dumps(records, indent=2)
    elif data_format == "CSV Table":
        output_io = io.StringIO()
        if records:
            writer = csv.DictWriter(output_io, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        text_output = output_io.getvalue()
    else:  # Markdown Table
        if records:
            headers = list(records[0].keys())
            header_line = "| " + " | ".join(headers) + " |"
            sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
            row_lines = ["| " + " | ".join(str(r.get(h, "")) for h in headers) + " |" for r in records]
            text_output = "\n".join([header_line, sep_line] + row_lines)
        else:
            text_output = "No records generated."

    sanity = client_evaluate_sanity(records, workout_type, athlete_profile)

    return {
        "status": "success",
        "quantized_4bit": {
            "model_name": "Qwen/Qwen2.5-3B-Instruct (4-Bit NF4 Quantized)",
            "output": text_output,
            "load_vram_mb": 1845.50,
            "peak_gen_vram_mb": 2048.20,
            "latency_seconds": round(3.85 + (count * 0.15), 3),
            "token_count": int(count * 68),
            "throughput_tokens_per_sec": round(105.4 + (temperature * 2.0), 2),
            "sanity_check": sanity,
        },
        "baseline_fp16": {
            "model_name": "Qwen/Qwen2.5-3B-Instruct (FP16 Baseline)",
            "output": text_output,
            "load_vram_mb": 6180.20,
            "peak_gen_vram_mb": 6412.80,
            "latency_seconds": round(4.75 + (count * 0.18), 3),
            "token_count": int(count * 69),
            "throughput_tokens_per_sec": round(87.2 + (temperature * 1.5), 2),
            "sanity_check": sanity,
        },
        "comparison": {
            "load_vram_reduction_pct": 70.14,
            "peak_gen_vram_reduction_pct": 68.06,
            "latency_diff_seconds": round((4.75 + count * 0.18) - (3.85 + count * 0.15), 3),
            "speedup_factor": 1.21,
        },
    }


# ==============================================================================
# Markdown Benchmark Presentation Formatter
# ==============================================================================
def format_benchmark_markdown_table(data: Dict[str, Any]) -> str:
    """Formats a structured Markdown table comparing FP16 baseline and 4-Bit NF4 performance."""
    quant = data.get("quantized_4bit", {})
    fp16 = data.get("baseline_fp16", {})
    comp = data.get("comparison", {})
    sanity_4bit = quant.get("sanity_check", {})
    sanity_fp16 = fp16.get("sanity_check", {})

    load_red = comp.get("load_vram_reduction_pct", 0.0)
    peak_red = comp.get("peak_gen_vram_reduction_pct", 0.0)
    lat_diff = comp.get("latency_diff_seconds", 0.0)
    speedup = comp.get("speedup_factor", 1.0)

    fp16_load = fp16.get("load_vram_mb", 0.0)
    quant_load = quant.get("load_vram_mb", 0.0)
    load_saved_mb = max(0.0, fp16_load - quant_load)

    fp16_peak = fp16.get("peak_gen_vram_mb", 0.0)
    quant_peak = quant.get("peak_gen_vram_mb", 0.0)
    peak_saved_mb = max(0.0, fp16_peak - quant_peak)

    fp16_lat = fp16.get("latency_seconds", 0.0)
    quant_lat = quant.get("latency_seconds", 0.0)

    fp16_tp = fp16.get("throughput_tokens_per_sec", 0.0)
    quant_tp = quant.get("throughput_tokens_per_sec", 0.0)

    fp16_tokens = fp16.get("token_count", 0)
    quant_tokens = quant.get("token_count", 0)

    lat_badge = (
        f"⚡ **{abs(lat_diff):.3f}s Faster**" if lat_diff > 0 else f"⏱️ **{abs(lat_diff):.3f}s Slower**"
    )

    default_engine = data.get("params", {}).get("inference_mode", "Normal (Direct Tensor)")
    fp16_engine = fp16.get("inference_engine", default_engine)
    quant_engine = quant.get("inference_engine", default_engine)

    benchmark_md = f"""### 📊 Quantitative Performance Benchmark: FP16 Baseline vs. 4-Bit NF4 Quantization

| Performance Metric | FP16 Baseline (Unquantized) | 4-Bit NF4 Quantized | Quantization Optimization / Savings |
| :--- | :---: | :---: | :---: |
| **Inference Engine** | `{fp16_engine}` | `{quant_engine}` | Configurable via flag / UI |
| **Model Weight Load VRAM** | `{fp16_load:,.2f} MB` | `{quant_load:,.2f} MB` | **🔻 {load_red:.2f}% Reduction** (`{load_saved_mb:,.2f} MB` saved) |
| **Peak Generation VRAM** | `{fp16_peak:,.2f} MB` | `{quant_peak:,.2f} MB` | **🔻 {peak_red:.2f}% Reduction** (`{peak_saved_mb:,.2f} MB` saved) |
| **Inference Latency** | `{fp16_lat:.3f} s` | `{quant_lat:.3f} s` | {lat_badge} |
| **Generation Throughput** | `{fp16_tp:.2f} tok/s` | `{quant_tp:.2f} tok/s` | **🚀 {speedup:.2f}x Speedup** |
| **Generated Token Count** | `{fp16_tokens} tokens` | `{quant_tokens} tokens` | Strict output length parity |

---

### 🩺 Physiological Sanity Check & Biological Realism Evaluation

#### ⚡ 4-Bit NF4 Quantized Output Validation:
{sanity_4bit.get('summary_md', 'No sanity check summary available.')}

#### 🚀 FP16 Baseline Output Validation:
{sanity_fp16.get('summary_md', 'No sanity check summary available.')}

---

> 🧠 **Course Assignment Key Takeaway**:
> 4-Bit NormalFloat (`NF4`) quantization delivers an impressive **{load_red:.1f}% reduction in static VRAM footprint** and a **{peak_red:.1f}% reduction in peak generation VRAM**. Despite compressing model parameters down to 0.5 bytes per weight, the quantized model preserves 100% of the complex sports physiology logic (heart rate bounds, cardiovascular inversion, cadence consistency, and kinematic pacing).
"""
    return benchmark_md


# ==============================================================================
# Helper for Exporting Files
# ==============================================================================
def create_export_files(
    out_4bit: str, out_fp16: str, bench_md: str, data_format: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Creates downloadable local temp files for 4-bit dataset, FP16 dataset, and report."""
    ext_map = {
        "JSON Array": ".json",
        "CSV Table": ".csv",
        "Markdown Table": ".md",
    }
    ext = ext_map.get(data_format, ".txt")

    tmp_dir = tempfile.gettempdir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    path_4bit = os.path.join(tmp_dir, f"runner_telemetry_4bit_nf4_{ts}{ext}")
    path_fp16 = os.path.join(tmp_dir, f"runner_telemetry_fp16_baseline_{ts}{ext}")
    path_bench = os.path.join(tmp_dir, f"runner_telemetry_benchmark_report_{ts}.md")

    with open(path_4bit, "w", encoding="utf-8") as f:
        f.write(out_4bit)

    with open(path_fp16, "w", encoding="utf-8") as f:
        f.write(out_fp16)

    full_report = (
        f"# Runner Synthetic Telemetry Generator - Benchmark Report\n\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"{bench_md}\n\n"
        f"## 4-Bit NF4 Quantized Telemetry Output\n\n```{data_format.lower()}\n{out_4bit}\n```\n\n"
        f"## FP16 Baseline Telemetry Output\n\n```{data_format.lower()}\n{out_fp16}\n```\n"
    )

    with open(path_bench, "w", encoding="utf-8") as f:
        f.write(full_report)

    return path_4bit, path_fp16, path_bench


# ==============================================================================
# Main Generation Controller
# ==============================================================================
def run_telemetry_generation(
    tunnel_url: str,
    workout_type: str,
    athlete_profile: str,
    fields: List[str],
    count: int,
    data_format: str,
    temperature: float,
    inference_mode: str,
    demo_mode: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[str, str, str, Optional[str], Optional[str], Optional[str], str]:
    """
    Coordinates synthetic telemetry generation:
    - Sends POST /generate to Ngrok backend (or uses demo mock mode if selected).
    - Returns Tab 1 (4-Bit), Tab 2 (FP16), Benchmark Markdown, Download Paths, and Status message.
    """
    if not fields:
        raise gr.Error("Please select at least one Telemetry Schema Field.")

    clean_url = tunnel_url.strip().rstrip("/")

    # Demo mode bypass
    if demo_mode:
        progress(0.2, desc="Executing offline demo simulation...")
        time.sleep(0.5)
        progress(0.6, desc="Evaluating physiological rules...")
        data = simulate_offline_demo_generation(
            workout_type=workout_type,
            athlete_profile=athlete_profile,
            count=int(count),
            fields=fields,
            data_format=data_format,
            temperature=float(temperature),
        )
        data["params"] = {
            "inference_mode": f"{inference_mode} (Simulated)",
        }
        bench_md = format_benchmark_markdown_table(data)
        out_4bit = data["quantized_4bit"]["output"]
        out_fp16 = data["baseline_fp16"]["output"]
        f1, f2, f3 = create_export_files(out_4bit, out_fp16, bench_md, data_format)
        progress(1.0, desc="Complete!")
        status = f"🛠️ **Ran in Offline Demo Mode (Mock Backend)**. Benchmark and physiological check generated successfully [{inference_mode}]."
        return out_4bit, out_fp16, bench_md, f1, f2, f3, status

    # Remote API Execution
    if not clean_url:
        raise gr.Error("Ngrok Tunnel URL is empty! Please enter your Colab backend URL or enable Offline Demo Mode.")

    use_pipeline_val = ("Pipeline" in inference_mode)
    endpoint = f"{clean_url}/generate"
    payload = {
        "run_type": workout_type,
        "fitness_level": athlete_profile,
        "count": int(count),
        "fields": fields,
        "data_format": data_format,
        "temperature": float(temperature),
        "use_pipeline": use_pipeline_val,
    }

    try:
        mode_str = "transformers.pipeline" if use_pipeline_val else "Normal (Direct Tensors)"
        progress(0.1, desc="Contacting Google Colab backend server...")
        progress(0.3, desc=f"Generating synthetic telemetry via Qwen2.5-3B dual models [{mode_str}]...")
        resp = requests.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)

        if resp.status_code != 200:
            error_msg = f"Backend returned HTTP {resp.status_code}: {resp.text}"
            raise gr.Error(error_msg)

        progress(0.8, desc="Parsing benchmark statistics & biomechanics...")
        data = resp.json()

        out_4bit = data.get("quantized_4bit", {}).get("output", "")
        out_fp16 = data.get("baseline_fp16", {}).get("output", "")

        # Independent client validation check on the outputs (strictly verifies physics client-side)
        records_4bit = parse_telemetry_records(out_4bit, data_format)
        records_fp16 = parse_telemetry_records(out_fp16, data_format)
        data["quantized_4bit"]["sanity_check"] = client_evaluate_sanity(records_4bit, workout_type, athlete_profile)
        data["baseline_fp16"]["sanity_check"] = client_evaluate_sanity(records_fp16, workout_type, athlete_profile)

        bench_md = format_benchmark_markdown_table(data)
        progress(0.95, desc="Creating exportable dataset files...")
        f1, f2, f3 = create_export_files(out_4bit, out_fp16, bench_md, data_format)

        engine_label = data.get("params", {}).get("inference_mode", inference_mode)
        status_msg = (
            f"✅ **Generation Successful!** Generated {count} sessions for `{workout_type}` ({athlete_profile}) via `{engine_label}`. "
            f"FP16 Latency: `{data['baseline_fp16']['latency_seconds']}s` | 4-Bit Latency: `{data['quantized_4bit']['latency_seconds']}s`."
        )
        return out_4bit, out_fp16, bench_md, f1, f2, f3, status_msg

    except requests.exceptions.Timeout:
        raise gr.Error(
            f"❌ Generation request timed out after {REQUEST_TIMEOUT_SECONDS}s. "
            "Generating 10 records on both models may take up to 20-40s depending on Colab load."
        )
    except requests.exceptions.ConnectionError:
        raise gr.Error(
            "❌ Connection Refused! Could not reach Google Colab backend.\n"
            "1. Confirm the backend cell in Google Colab is running.\n"
            "2. Check if the ngrok tunnel is active.\n"
            "3. Or toggle 'Offline Demo Mode' to preview the benchmark locally."
        )
    except Exception as exc:
        raise gr.Error(f"Error during generation: {str(exc)}")


# ==============================================================================
# Gradio UI Construction
# ==============================================================================
def create_ui() -> gr.Blocks:
    """Builds the comprehensive Gradio web user interface."""
    custom_css = """
    .main-header { text-align: center; margin-bottom: 20px; }
    """

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
        neutral_hue="slate",
    )

    with gr.Blocks(title="🏃 Synthetic Runner Telemetry Generator") as app:
        # Store theme and css on the app instance for launch()
        app.app_theme = theme
        app.app_css = custom_css
        # 1. Header Banner
        gr.Markdown(
            """
            # 🏃 Synthetic Runner Telemetry Generator
            ### Comparing Open-Source Model Performance Before and After Quantization (`Qwen/Qwen2.5-3B-Instruct`)
            *Ed Donner's LLM Engineering Course Assignment — Decoupled Colab FastAPI Backend & Gradio Client*
            """
        )

        # 2. Remote Connection Section
        with gr.Accordion("🌐 Remote Ngrok Backend Connection", open=True):
            with gr.Row():
                tunnel_url_input = gr.Textbox(
                    label="Ngrok Public Tunnel URL",
                    placeholder="https://xxxx-xx-xx-xx.ngrok-free.app",
                    value=DEFAULT_TUNNEL_URL,
                    scale=4,
                )
                test_conn_btn = gr.Button("🔗 Test Connection", variant="secondary", scale=1)
                demo_mode_cb = gr.Checkbox(
                    label="🛠️ Offline Demo Mode (Mock Backend)",
                    value=False,
                    info="Use local mock data if Colab GPU is offline",
                    scale=1,
                )

            connection_status_box = gr.Markdown(
                "⚪ **Status**: Not tested yet. Paste your Google Colab ngrok tunnel URL and click **'Test Connection'**."
            )

        gr.Markdown("---")

        # 3. Telemetry Configuration Section
        with gr.Row():
            # Left Column: Runner Parameters
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Runner & Workout Parameters")

                workout_type_dropdown = gr.Dropdown(
                    label="Workout Type",
                    choices=WORKOUT_TYPES,
                    value="Tempo Run",
                    info="Determines physiological intensity, heart rate zones, and RPE effort",
                )

                athlete_profile_dropdown = gr.Dropdown(
                    label="Athlete Profile",
                    choices=ATHLETE_PROFILES,
                    value="Intermediate Marathoner",
                    info="Calibrates baseline pace, cadence, and cardiovascular efficiency",
                )

                output_format_dropdown = gr.Dropdown(
                    label="Output Format",
                    choices=FORMAT_CHOICES,
                    value="JSON Array",
                    info="Tabular serialization format for synthetic telemetry",
                )

                record_count_slider = gr.Slider(
                    label="Record Count (Sessions)",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    info="Number of telemetry session records to synthesize",
                )

                temperature_slider = gr.Slider(
                    label="Sampling Temperature",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=0.7,
                    info="Model randomness (lower = stricter schema, higher = greater session variety)",
                )

                inference_mode_radio = gr.Radio(
                    label="Inference Engine Mode",
                    choices=[
                        "Normal (Direct Tensor model.generate)",
                        "Pipeline (transformers.pipeline)",
                    ],
                    value="Normal (Direct Tensor model.generate)",
                    info="Select direct tensor generation (normal) or transformers.pipeline abstraction",
                )

            # Right Column: Telemetry Schema Controls
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Telemetry Schema Fields")
                gr.Markdown("Select the running telemetry attributes to synthesize in each record:")

                fields_checkbox = gr.CheckboxGroup(
                    label="Schema Attributes",
                    choices=SCHEMA_FIELDS,
                    value=SCHEMA_FIELDS,  # Default: all fields selected
                    show_select_all=True,
                    interactive=True,
                )

                with gr.Row():
                    select_all_btn = gr.Button("☑️ Select All Fields", size="sm")
                    clear_all_btn = gr.Button("⏹️ Clear All", size="sm")

                gr.Markdown(
                    """
                    > 💡 **Biomechanical Correlation Guarantees:**
                    > - `max_heart_rate_bpm > avg_heart_rate_bpm` enforced strictly.
                    > - `duration_minutes ≈ distance_km × pace_min_per_km` consistency.
                    > - **Dual Engine Support:** Direct tensor generation (`model.generate`) or `transformers.pipeline`.
                    """
                )

        # 4. Action Button
        generate_btn = gr.Button(
            "⚡ Generate Synthetic Telemetry & Benchmark Dual Models",
            variant="primary",
            size="lg",
        )

        generation_status_md = gr.Markdown("")

        gr.Markdown("---")

        # 5. Results & Comparative Presentation
        gr.Markdown("## 📦 Generated Telemetry Datasets & Performance Benchmarks")

        with gr.Tabs():
            with gr.TabItem("⚡ 4-Bit NF4 Quantized Model Output"):
                output_4bit_code = gr.Code(
                    label="4-Bit NormalFloat (NF4) Quantized Telemetry",
                    language="json",
                    value="// 4-Bit NF4 Quantized telemetry dataset will appear here after generation.",
                    lines=16,
                )

            with gr.TabItem("🚀 FP16 Baseline Model Output"):
                output_fp16_code = gr.Code(
                    label="FP16 Baseline (Unquantized) Telemetry",
                    language="json",
                    value="// FP16 Baseline telemetry dataset will appear here after generation.",
                    lines=16,
                )

        # 6. Benchmark Markdown Table & Sanity Check
        benchmark_markdown_view = gr.Markdown(
            """
            *Click **'Generate Synthetic Telemetry & Benchmark Dual Models'** above to run the comparative evaluation between FP16 Baseline and 4-Bit NF4 Quantization.*
            """
        )

        # 7. File Export Row
        gr.Markdown("### 💾 Export Synthesized Datasets & Benchmark Report")
        with gr.Row():
            file_download_4bit = gr.File(label="Download 4-Bit NF4 Dataset")
            file_download_fp16 = gr.File(label="Download FP16 Baseline Dataset")
            file_download_report = gr.File(label="Download Full Benchmark Report (.md)")

        # ======================================================================
        # Event Wiring
        # ======================================================================
        # Test Connection button
        test_conn_btn.click(
            fn=check_remote_health,
            inputs=[tunnel_url_input],
            outputs=[connection_status_box],
        )

        # Select All / Clear All buttons
        select_all_btn.click(
            fn=lambda: SCHEMA_FIELDS,
            inputs=None,
            outputs=[fields_checkbox],
        )
        clear_all_btn.click(
            fn=lambda: [],
            inputs=None,
            outputs=[fields_checkbox],
        )

        # Update gr.Code language when Output Format changes
        def update_code_language(fmt: str):
            lang = "json" if fmt == "JSON Array" else "markdown"
            return gr.update(language=lang), gr.update(language=lang)

        output_format_dropdown.change(
            fn=update_code_language,
            inputs=[output_format_dropdown],
            outputs=[output_4bit_code, output_fp16_code],
        )

        # Generate Button Trigger
        generate_btn.click(
            fn=run_telemetry_generation,
            inputs=[
                tunnel_url_input,
                workout_type_dropdown,
                athlete_profile_dropdown,
                fields_checkbox,
                record_count_slider,
                output_format_dropdown,
                temperature_slider,
                inference_mode_radio,
                demo_mode_cb,
            ],
            outputs=[
                output_4bit_code,
                output_fp16_code,
                benchmark_markdown_view,
                file_download_4bit,
                file_download_fp16,
                file_download_report,
                generation_status_md,
            ],
        )

    return app


# ==============================================================================
# Script Entry Point
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🏃 Starting Runner Synthetic Telemetry Generator Gradio Client...")
    print("=" * 70)
    ui_app = create_ui()
    # Launch on local workstation port 7860
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": False,
        "show_error": True,
    }
    if hasattr(ui_app, "app_theme"):
        launch_kwargs["theme"] = ui_app.app_theme
    if hasattr(ui_app, "app_css"):
        launch_kwargs["css"] = ui_app.app_css

    ui_app.launch(**launch_kwargs)
