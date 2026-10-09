"""
Runner Synthetic Telemetry Data Generator - Google Colab Backend Server
========================================================================
Course Assignment: Building Synthetic Telemetry Data Generator using Open-Source Models
Model: Qwen/Qwen2.5-3B-Instruct (Decoder-Only Causal LM)
Hardware Target: Google Colab Free T4 GPU (16 GB VRAM)

This script implements:
1. Dual Model Loading:
   - FP16 Baseline (Unquantized)
   - 4-Bit NormalFloat (NF4) Quantized via bitsandbytes
   - Memory tracking & initial weight VRAM allocation logging
2. Direct Tensor-Level Generation (Strictly NO transformers.pipeline):
   - Model chat template with system instructions enforcing biomechanical & physiological correlations
   - Explicit tensor device placement (.to(model.device))
   - Generation with model.generate() under torch.no_grad()
   - Prompt slicing (outputs[0][prompt_len:]) and decoding
   - Inference telemetry: peak generation VRAM, latency, token count, throughput (tokens/sec)
   - Pre-run GPU memory cleanup (gc.collect(), empty_cache(), reset_peak_memory_stats())
3. FastAPI Server & pyngrok Tunnel:
   - GET /health: Status & model VRAM diagnostics
   - POST /generate: Telemetry generation with comparative FP16 vs. 4-Bit benchmark
   - Ngrok authtoken configuration and Uvicorn server launch via nest_asyncio

Dependencies (Colab):
!pip install -q torch transformers accelerate bitsandbytes fastapi uvicorn pyngrok pydantic nest_asyncio
"""

import os
import sys
import gc
import time
import json
import re
import csv
import io
from typing import List, Dict, Any, Optional

import torch
from pydantic import BaseModel, Field

# ==============================================================================
# Step 0: Ensure Required Dependencies Are Installed
# ==============================================================================
REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "fastapi",
    "uvicorn",
    "pyngrok",
    "pydantic",
    "nest_asyncio",
    "huggingface_hub",
]


def check_and_install_dependencies():
    """Verifies dependencies and provides guidance for Colab execution."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"⚠️ Missing packages: {missing}")
        print("💡 In Google Colab, execute the following command first:")
        print(f"   !pip install -q {' '.join(REQUIRED_PACKAGES)}")
        # If running inside an interactive notebook/Colab session, attempt automatic install
        if "google.colab" in sys.modules:
            import subprocess
            print("[*] Automatically installing missing packages in Colab...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
            print("[✓] Dependencies installed successfully!")


check_and_install_dependencies()

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import uvicorn
from pyngrok import ngrok


# ==============================================================================
# Step 1: Hugging Face Authentication & Hardware Verification
# ==============================================================================
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None

if not HF_TOKEN or HF_TOKEN == "YOUR_HF_TOKEN_HERE":
    HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

if HF_TOKEN and HF_TOKEN != "YOUR_HF_TOKEN_HERE":
    login(token=HF_TOKEN)
    print("[✓] Successfully authenticated with Hugging Face Hub!")
    hf_auth = HF_TOKEN
else:
    print("[*] No personal Hugging Face token provided. Proceeding with public access...")
    hf_auth = None

MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"

print("=" * 75)
print("🏃 RUNNER SYNTHETIC DATA GENERATOR - BACKEND INITIALIZATION")
print("=" * 75)
print(f"[*] Target Model ID:       {MODEL_ID}")
print(f"[*] PyTorch Version:       {torch.__version__}")
print(f"[*] CUDA Available:        {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[*] Target GPU Device:     {gpu_name} ({total_vram_gb:.2f} GB VRAM)")
else:
    print("⚠️ WARNING: CUDA GPU is NOT detected! Running on CPU will be slow.")
    print("💡 In Google Colab: Runtime -> Change runtime type -> Hardware accelerator: T4 GPU")

# ==============================================================================
# Step 2: Load Tokenizer
# ==============================================================================
print("\n[*] Loading AutoTokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_auth)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"[✓] Tokenizer loaded successfully (vocab size: {tokenizer.vocab_size})")

# ==============================================================================
# Step 3: Dual Model Loading & VRAM Weight Tracking
# ==============================================================================
# Both models fit comfortably inside a free Colab T4 GPU (16 GB):
# - FP16 Baseline: ~6.2 GB
# - 4-Bit NF4:      ~1.9 GB
# Total:            ~8.1 GB VRAM (well within 15 GB usable on T4)

# ------------------------------------------------------------------------------
# 3A: Baseline Model (Unquantized FP16)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[*] STEP 1/2: Loading Baseline Unquantized Model (torch.float16)...")
print("=" * 60)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

t_fp16_load_start = time.perf_counter()

model_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=hf_auth,
)
model_fp16.eval()

t_fp16_load_elapsed = time.perf_counter() - t_fp16_load_start

if torch.cuda.is_available():
    fp16_load_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
else:
    fp16_load_vram_mb = model_fp16.get_memory_footprint() / (1024 ** 2)

fp16_footprint_mb = model_fp16.get_memory_footprint() / (1024 ** 2)

print(f"[✓] FP16 Baseline Model loaded in {t_fp16_load_elapsed:.2f}s!")
print(f"    - Initial Weight VRAM Allocation: {fp16_load_vram_mb:.2f} MB")
print(f"    - HuggingFace Memory Footprint:   {fp16_footprint_mb:.2f} MB")

# ------------------------------------------------------------------------------
# 3B: Quantized Model (4-Bit NormalFloat NF4)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[*] STEP 2/2: Loading Quantized Model (4-Bit NF4 via bitsandbytes)...")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

if torch.cuda.is_available():
    mem_before_4bit = torch.cuda.memory_allocated() / (1024 ** 2)
    torch.cuda.reset_peak_memory_stats()
else:
    mem_before_4bit = 0.0

t_4bit_load_start = time.perf_counter()

model_4bit = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=hf_auth,
)
model_4bit.eval()

t_4bit_load_elapsed = time.perf_counter() - t_4bit_load_start

if torch.cuda.is_available():
    mem_after_4bit = torch.cuda.memory_allocated() / (1024 ** 2)
    bit4_load_vram_mb = max(0.0, mem_after_4bit - mem_before_4bit)
    if bit4_load_vram_mb == 0.0:
        bit4_load_vram_mb = model_4bit.get_memory_footprint() / (1024 ** 2)
else:
    bit4_load_vram_mb = model_4bit.get_memory_footprint() / (1024 ** 2)

bit4_footprint_mb = model_4bit.get_memory_footprint() / (1024 ** 2)
load_vram_reduction_pct = ((fp16_footprint_mb - bit4_footprint_mb) / fp16_footprint_mb) * 100

print(f"[✓] 4-Bit NF4 Quantized Model loaded in {t_4bit_load_elapsed:.2f}s!")
print(f"    - Dedicated Weight VRAM:          {bit4_load_vram_mb:.2f} MB")
print(f"    - HuggingFace Memory Footprint:   {bit4_footprint_mb:.2f} MB")
print(f"    - Weight Footprint VRAM Savings:  {load_vram_reduction_pct:.2f}% reduction")
print("=" * 60)

# ==============================================================================
# Step 4: System Instruction & Biomechanical Correlation Prompt
# ==============================================================================
SYSTEM_PROMPT = """You are an expert sports physiologist, biomechanical running scientist, and synthetic telemetry data engine.
Your mission is to generate realistic, scientifically sound running telemetry datasets adhering strictly to physiological dynamics.

CRITICAL BIOMECHANICAL & PHYSIOLOGICAL RULES:
1. Heart Rate Rules:
   - 'max_heart_rate_bpm' MUST ALWAYS be strictly greater than 'avg_heart_rate_bpm' (by 8 to 25 bpm).
   - Absolute physiological limits: avg_heart_rate_bpm in 100-195 bpm, max_heart_rate_bpm in 120-215 bpm.
2. Workout Intensity & Target Zones:
   - 'Easy Recovery Run': Aerobic Zone 1-2 (Avg HR 115-142 bpm), low cardiovascular strain, RPE 2-4. Pace is 60-90 sec/km slower than threshold.
   - 'Tempo Run': Lactate Threshold Zone 3-4 (Avg HR 155-175 bpm), high steady effort, RPE 6-7. Fast, consistent pace.
   - 'Interval Track Session': Anaerobic repetitions, Zone 4-5 (Avg HR 160-180 bpm, Max HR 180-200 bpm), RPE 8-9. Spiking HR, high cadence (175-195 spm).
   - 'Long Weekend Run': Aerobic endurance, Zone 2-3 (Avg HR 130-155 bpm), elevated calories, RPE 5-7. Distance MUST scale to the athlete's fitness level:
     * 'Beginner 5k Runner': 5 to 9 km (pace 5:45 - 8:30 min/km).
     * 'Intermediate Marathoner': 16 to 28 km (pace 4:15 - 5:45 min/km).
     * 'Sub-Elite / Elite': 22 to 36 km (pace 3:00 - 4:15 min/km).
   - 'Hill Repeats': High muscular power & VO2 max strain, high elevation gain (>120m), Max HR near max (180-205 bpm), RPE 8-10.
3. Athlete Profile Calibrations:
   - 'Beginner 5k Runner': Slower paces (5:45 - 8:30 min/km, or decimal 5.75 - 8.50), cadence 150-165 spm, higher heart rate at moderate paces, typical run distances 3-8 km (long runs 5-9 km).
   - 'Intermediate Marathoner': Paces 4:15 - 5:45 min/km (decimal 4.25 - 5.75), cadence 165-178 spm, disciplined aerobic efficiency, distances 8-25 km.
   - 'Sub-Elite / Elite': Fast paces 3:00 - 4:15 min/km (decimal 3.00 - 4.25), cadence 175-195 spm, exceptional cardiac stroke volume, distances 10-35 km.
4. Kinematic & Mathematical Consistency:
   - CRITICAL ARITHMETIC LAW: duration_minutes = round(distance_km * pace_min_per_km, 1).
   - You MUST ensure duration_minutes divided by distance_km equals pace_min_per_km!
   - Concrete calculation examples:
     * 6.0 km at 6.50 min/km (6:30/km) pace MUST have duration_minutes = 39.0 (6.0 * 6.5 = 39.0).
     * 8.0 km at 6.00 min/km (6:00/km) pace MUST have duration_minutes = 48.0 (8.0 * 6.0 = 48.0).
     * 10.0 km in 60.0 minutes requires pace = 6.00 min/km (60 / 10 = 6.0, NEVER output pace 5.0 for 60 min!).
     * 10.0 km at 5.00 min/km pace MUST have duration_minutes = 50.0 (10.0 * 5.0 = 50.0).
     * 15.0 km at 5.00 min/km pace MUST have duration_minutes = 75.0 (15.0 * 5.0 = 75.0, NEVER output pace 4.5!).
   - Cadence: 145 - 198 steps per minute (spm).
   - Calories: approximately 55 - 75 kcal per km multiplied by relative effort.
5. Strict Output Formatting:
   - Output ONLY the requested data in the specified format (JSON Array, CSV Table, or Markdown Table).
   - For JSON Array, ALWAYS format as an indented multi-line JSON array (2 spaces indentation per field) so it is readable and not a compressed single line.
   - Include ONLY the requested schema fields.
   - DO NOT include conversational filler, preamble, notes, or markdown explanations outside the structured data.
"""


def build_user_prompt(
    run_type: str,
    fitness_level: str,
    count: int,
    fields: List[str],
    data_format: str,
) -> str:
    """Builds a formatted prompt specifying exact schema and formatting constraints."""
    fields_str = ", ".join(fields)
    format_guide = {
        "JSON Array": (
            "Format the output strictly as a pretty-printed, indented JSON array of objects (2 spaces indentation per field), e.g.:\n"
            "[\n"
            "  {\n"
            '    "field1": value1,\n'
            '    "field2": value2\n'
            "  }\n"
            "]\n"
            "Ensure valid multi-line JSON syntax with double quotes and no trailing commas."
        ),
        "CSV Table": (
            "Format the output strictly as a CSV table with a header row followed by comma-separated records, e.g.:\n"
            "field1,field2,field3\nvalue1,value2,value3\n"
            "Do not include any conversational text before or after the CSV."
        ),
        "Markdown Table": (
            "Format the output strictly as a Markdown table with column headers and delimiter row, e.g.:\n"
            "| field1 | field2 | field3 |\n| --- | --- | --- |\n| val1 | val2 | val3 |\n"
            "Do not include conversational text before or after the table."
        ),
    }.get(data_format, "Output in the specified format.")

    return (
        f"Generate realistic synthetic telemetry for exactly {count} running sessions with these parameters:\n"
        f"- Workout Type: {run_type}\n"
        f"- Athlete Profile: {fitness_level}\n"
        f"- Required Fields: {fields_str}\n"
        f"- Output Format: {data_format}\n\n"
        f"{format_guide}\n\n"
        f"CRITICAL SANITY & ARITHMETIC REQUIREMENTS:\n"
        f"1. max_heart_rate_bpm must strictly exceed avg_heart_rate_bpm (by 8-25 bpm).\n"
        f"2. KINEMATIC ARITHMETIC: duration_minutes MUST EXACTLY EQUAL distance_km * pace_min_per_km.\n"
        f"   For a {fitness_level} doing {run_type}, select distance_km and pace_min_per_km appropriate for {fitness_level}, then calculate duration_minutes = round(distance_km * pace_min_per_km, 1).\n"
        f"   Verify that duration_minutes / distance_km equals pace_min_per_km before writing each record.\n"
    )


# ==============================================================================
# Step 5: Dual Inference Engine (Normal Tensor-Level vs. transformers.pipeline)
# ==============================================================================
# INFERENCE MODE FLAG:
# - False (Default): Use the "normal way" (direct tensor-level generation via model.generate)
#   with explicit device movement, KV-cache cleanup, and prompt token slicing.
# - True: Use Hugging Face's high-level `transformers.pipeline` abstraction.
USE_PIPELINE: bool = False

_pipeline_cache: Dict[int, Any] = {}


def get_pipeline(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> Any:
    """Retrieves or creates a cached transformers.pipeline instance for a model."""
    key = id(model)
    if key not in _pipeline_cache:
        _pipeline_cache[key] = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
    return _pipeline_cache[key]


def execute_tensor_level_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    model_load_vram_mb: float,
    temperature: float = 0.7,
    max_new_tokens: int = 1500,
) -> Dict[str, Any]:
    """
    Executes manual, direct tensor-level generation ("normal way") using PyTorch.

    Steps:
    1. GPU memory cleanup (gc.collect(), torch.cuda.empty_cache(), reset_peak_memory_stats()).
    2. Format prompt via tokenizer.apply_chat_template().
    3. Tokenize and move tensors explicitly to model.device.
    4. Run generation via model.generate(...) under torch.no_grad().
    5. Slice prompt tokens (outputs[0][prompt_len:]) and decode with skip_special_tokens=True.
    6. Capture latency via time.perf_counter(), peak generation VRAM, and throughput.
    """
    # 1. GPU Memory Cleanup before run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pre_gen_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        pre_gen_vram_mb = 0.0

    # 2. Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 3. Tokenize and move explicitly to model device
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    device = model.device
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # 4. Timed Tensor Generation under torch.no_grad()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if temperature > 0.01:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = 0.9
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    latency_seconds = t_end - t_start

    # 5. Peak Generation VRAM Calculation
    if torch.cuda.is_available():
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        activation_overhead_mb = max(0.0, peak_allocated_mb - pre_gen_vram_mb)
        peak_gen_vram_mb = model_load_vram_mb + activation_overhead_mb
    else:
        peak_allocated_mb = 0.0
        peak_gen_vram_mb = model_load_vram_mb

    # 6. Slice Prompt Tokens & Decode
    generated_token_ids = outputs[0][prompt_len:]
    token_count = len(generated_token_ids)
    throughput = (token_count / latency_seconds) if latency_seconds > 0 else 0.0

    raw_decoded_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

    return {
        "output": raw_decoded_text,
        "token_count": int(token_count),
        "latency_seconds": round(latency_seconds, 3),
        "throughput_tokens_per_sec": round(throughput, 2),
        "peak_gen_vram_mb": round(peak_gen_vram_mb, 2),
        "total_gpu_peak_vram_mb": round(peak_allocated_mb, 2),
        "engine": "normal (model.generate)",
    }


def execute_pipeline_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    model_load_vram_mb: float,
    temperature: float = 0.7,
    max_new_tokens: int = 1500,
) -> Dict[str, Any]:
    """
    Executes inference via Hugging Face's high-level transformers.pipeline abstraction.

    Steps:
    1. Pre-run GPU Memory Cleanup (gc.collect(), torch.cuda.empty_cache(), reset_peak_memory_stats()).
    2. Get/cache pipeline instance for model.
    3. Timed generation using pipe(messages, ...).
    4. Measure wall-clock latency with torch.cuda.synchronize() and peak VRAM.
    5. Extract generated assistant content.
    6. Compute token count and throughput.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pre_gen_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        pre_gen_vram_mb = 0.0

    pipe = get_pipeline(model, tokenizer)

    pipe_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0.01:
        pipe_kwargs["do_sample"] = True
        pipe_kwargs["temperature"] = float(temperature)
        pipe_kwargs["top_p"] = 0.9
    else:
        pipe_kwargs["do_sample"] = False

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    try:
        outputs = pipe(messages, **pipe_kwargs)
        gen_res = outputs[0]["generated_text"]
        if isinstance(gen_res, list) and len(gen_res) > 0 and isinstance(gen_res[-1], dict):
            output_text = gen_res[-1].get("content", "").strip()
        elif isinstance(gen_res, str):
            output_text = gen_res.strip()
        else:
            output_text = str(gen_res).strip()
    except Exception:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        outputs = pipe(formatted_prompt, return_full_text=False, **pipe_kwargs)
        gen_res = outputs[0]["generated_text"]
        output_text = gen_res.strip() if isinstance(gen_res, str) else str(gen_res).strip()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    latency_seconds = t_end - t_start

    if torch.cuda.is_available():
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        activation_overhead_mb = max(0.0, peak_allocated_mb - pre_gen_vram_mb)
        peak_gen_vram_mb = model_load_vram_mb + activation_overhead_mb
    else:
        peak_allocated_mb = 0.0
        peak_gen_vram_mb = model_load_vram_mb

    token_ids = tokenizer.encode(output_text, add_special_tokens=False)
    token_count = len(token_ids)
    throughput = (token_count / latency_seconds) if latency_seconds > 0 else 0.0

    return {
        "output": output_text,
        "token_count": int(token_count),
        "latency_seconds": round(latency_seconds, 3),
        "throughput_tokens_per_sec": round(throughput, 2),
        "peak_gen_vram_mb": round(peak_gen_vram_mb, 2),
        "total_gpu_peak_vram_mb": round(peak_allocated_mb, 2),
        "engine": "transformers.pipeline",
    }


def execute_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    model_load_vram_mb: float,
    temperature: float = 0.7,
    max_new_tokens: int = 1500,
    use_pipeline: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Unified inference dispatcher based on the conditional flag:
    - If use_pipeline is True (or USE_PIPELINE is True when use_pipeline is None):
      Runs inference using transformers.pipeline.
    - Otherwise (False):
      Runs inference using the normal way (direct tensor-level model.generate with prompt slicing).
    """
    effective_flag = USE_PIPELINE if use_pipeline is None else use_pipeline
    if effective_flag:
        return execute_pipeline_inference(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            model_load_vram_mb=model_load_vram_mb,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    else:
        return execute_tensor_level_inference(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            model_load_vram_mb=model_load_vram_mb,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )


# ==============================================================================
# Step 6: Telemetry Parsing & Physiological Sanity Check Engine
# ==============================================================================
def parse_pace_to_minutes(pace_val: Any) -> float:
    """Parses pace string (e.g. '5:15' or '5.25') into decimal minutes."""
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
    """Extracts first float number from mixed string/number representations."""
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

    # Strip code block fences if present
    content = cleaned
    if "```" in content:
        for block in re.findall(r"```(?:json|csv|markdown|text)?\s*(.*?)\s*```", content, re.DOTALL):
            if block.strip():
                content = block.strip()
                break

    # 1. Try JSON parsing
    if data_format == "JSON Array" or content.startswith("[") or content.startswith("{"):
        try:
            # Match outermost array
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

    # 2. Try CSV parsing
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

    # 3. Try Markdown Table parsing
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


def pretty_format_and_reconcile_output(
    text: str, data_format: str, reconcile_kinematics: bool = True
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parses telemetry records from raw LLM output, ensures multi-line indented formatting
    (indent=2) for JSON tabs, and performs kinematic auto-reconciliation (duration ≈ distance * pace).
    """
    records = parse_telemetry_records(text, data_format)

    if reconcile_kinematics and records:
        for r in records:
            dist = extract_numeric(r.get("distance_km") or r.get("distance"))
            pace = parse_pace_to_minutes(r.get("pace_min_per_km") or r.get("pace"))
            dur = extract_numeric(r.get("duration_minutes") or r.get("duration"))
            if dist > 0 and pace > 0:
                expected_dur = round(dist * pace, 1)
                # Auto-align if duration is missing or deviates by > 5%
                if dur <= 0 or (abs(dur - expected_dur) / max(expected_dur, 1e-5) > 0.05):
                    val = int(expected_dur) if expected_dur.is_integer() else expected_dur
                    if "duration_minutes" in r:
                        r["duration_minutes"] = val
                    elif "duration" in r:
                        r["duration"] = val
                    else:
                        r["duration_minutes"] = val
            elif dist > 0 and dur > 0 and pace <= 0:
                calc_pace = round(dur / dist, 2)
                val_p = int(calc_pace) if calc_pace.is_integer() else calc_pace
                if "pace_min_per_km" in r:
                    r["pace_min_per_km"] = val_p
                elif "pace" in r:
                    r["pace"] = val_p
                else:
                    r["pace_min_per_km"] = val_p

    # Re-serialize into beautiful multi-line format if JSON Array
    if data_format == "JSON Array" and records:
        return json.dumps(records, indent=2), records

    # If raw string is valid JSON, format it with indent=2
    if data_format == "JSON Array":
        try:
            parsed = json.loads(text.strip())
            return json.dumps(parsed, indent=2), records
        except Exception:
            pass

    return text.strip(), records


def evaluate_physiological_sanity(
    records: List[Dict[str, Any]], workout_type: str, athlete_profile: str
) -> Dict[str, Any]:
    """
    Evaluates biomechanical and physiological rules on generated telemetry records.
    Returns status, score, details, and diagnostic breakdown.
    """
    if not records:
        return {
            "status": "Inconclusive",
            "score_pct": 0.0,
            "passed_checks": 0,
            "total_checks": 0,
            "details": ["Output could not be parsed into tabular/JSON records for automated verification."],
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
        # Check Heart Rate
        avg_hr = extract_numeric(r.get("avg_heart_rate_bpm") or r.get("avg_hr"))
        max_hr = extract_numeric(r.get("max_heart_rate_bpm") or r.get("max_hr"))
        if avg_hr > 0 and max_hr > 0:
            hr_inversion_evaluated += 1
            if max_hr > avg_hr:
                hr_inversion_passes += 1
            if 90 <= avg_hr <= 198 and 110 <= max_hr <= 215:
                hr_range_passes += 1

        # Check Cadence
        cadence = extract_numeric(r.get("cadence_spm") or r.get("cadence"))
        if cadence > 0:
            cadence_evaluated += 1
            if 140 <= cadence <= 205:
                cadence_passes += 1

        # Check RPE / Workout Intensity
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

        # Kinematic Coherence: Duration ≈ Distance * Pace
        dist = extract_numeric(r.get("distance_km") or r.get("distance"))
        dur = extract_numeric(r.get("duration_minutes") or r.get("duration"))
        pace = parse_pace_to_minutes(r.get("pace_min_per_km") or r.get("pace"))
        rid = r.get("runner_id") or f"Session #{kinematic_evaluated + 1}"
        if dist > 0 and dur > 0 and pace > 0:
            kinematic_evaluated += 1
            expected_dur = dist * pace
            diff_ratio = abs(dur - expected_dur) / max(expected_dur, 1e-5)
            # Strict tolerance: max 7% margin for decimal rounding (e.g. 5:15 pace is 5.25 min)
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
            f"{icon} **Heart Rate Cardinality**: {hr_inversion_passes}/{hr_inversion_evaluated} "
            f"({pct:.0f}%) sessions strictly observed `max_heart_rate_bpm > avg_heart_rate_bpm`."
        )

        checks_total += hr_inversion_evaluated
        passed_total += hr_range_passes
        pct_r = (hr_range_passes / hr_inversion_evaluated) * 100
        icon_r = "✅" if pct_r == 100 else "⚠️"
        details.append(
            f"{icon_r} **Cardiovascular Bounds**: {hr_range_passes}/{hr_inversion_evaluated} "
            f"({pct_r:.0f}%) sessions within human physiological cardiac range (90-215 bpm)."
        )

    if cadence_evaluated > 0:
        checks_total += cadence_evaluated
        passed_total += cadence_passes
        pct = (cadence_passes / cadence_evaluated) * 100
        icon = "✅" if pct >= 90 else "⚠️"
        details.append(
            f"{icon} **Biomechanical Cadence**: {cadence_passes}/{cadence_evaluated} "
            f"({pct:.0f}%) sessions within running range (140-205 spm)."
        )

    if intensity_evaluated > 0:
        checks_total += intensity_evaluated
        passed_total += intensity_passes
        pct = (intensity_passes / intensity_evaluated) * 100
        icon = "✅" if pct >= 80 else "⚠️"
        details.append(
            f"{icon} **RPE vs. Workout Intensity**: {intensity_passes}/{intensity_evaluated} "
            f"({pct:.0f}%) calibrated to `{workout_type}`."
        )

    if kinematic_evaluated > 0:
        checks_total += kinematic_evaluated
        passed_total += kinematic_passes
        pct = (kinematic_passes / kinematic_evaluated) * 100
        icon = "✅" if pct >= 90 else ("⚠️" if pct >= 50 else "❌")
        msg = (
            f"{icon} **Kinematic Motion Consistency**: {kinematic_passes}/{kinematic_evaluated} "
            f"({pct:.0f}%) sessions maintained `duration ≈ distance × pace`."
        )
        if kinematic_failures:
            examples_str = "; ".join(kinematic_failures[:2])
            msg += f"\n  - ⚠️ Arithmetic Mismatch in {len(kinematic_failures)} session(s): {examples_str}"
        details.append(msg)

    score = (passed_total / checks_total * 100) if checks_total > 0 else 100.0
    status_label = (
        "✅ PASSED (Strict Physiological Fidelity)"
        if score >= 90
        else ("⚠️ PARTIAL (Minor Physiological Deviations)" if score >= 70 else "❌ FAILED (Inconsistent Telemetry)")
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
# Step 7: FastAPI Application & Endpoints
# ==============================================================================
app = FastAPI(
    title="Runner Synthetic Telemetry Generator Backend",
    description="Dual-Model Inference Server comparing FP16 Baseline vs. 4-Bit NF4 Quantization (Qwen/Qwen2.5-3B-Instruct)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    run_type: str = Field(default="Tempo Run", description="Type of running session")
    fitness_level: str = Field(default="Intermediate Marathoner", description="Athlete fitness category")
    count: int = Field(default=5, ge=1, le=10, description="Number of synthetic records")
    fields: List[str] = Field(
        default=[
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
        ],
        description="Telemetry schema fields",
    )
    data_format: str = Field(default="JSON Array", description="Output format: JSON Array, CSV Table, or Markdown Table")
    temperature: float = Field(default=0.7, ge=0.1, le=1.0, description="Generation temperature")
    use_pipeline: Optional[bool] = Field(
        default=None,
        description="Inference mode flag: True to use transformers.pipeline, False for direct tensor generation (normal), None to use notebook default USE_PIPELINE",
    )


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint providing model metadata, allocated VRAM, and default inference mode."""
    return {
        "status": "healthy",
        "service": "Runner Synthetic Telemetry Generator",
        "model_id": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "device": str(model_fp16.device),
        "default_use_pipeline": USE_PIPELINE,
        "default_inference_mode": "transformers.pipeline" if USE_PIPELINE else "Normal (Direct Tensor model.generate)",
        "models_loaded": {
            "baseline_fp16": {
                "name": f"{MODEL_ID} (FP16 Baseline)",
                "weight_vram_mb": round(fp16_load_vram_mb, 2),
                "footprint_mb": round(fp16_footprint_mb, 2),
            },
            "quantized_4bit": {
                "name": f"{MODEL_ID} (4-Bit NF4 Quantized)",
                "weight_vram_mb": round(bit4_load_vram_mb, 2),
                "footprint_mb": round(bit4_footprint_mb, 2),
            },
        },
        "vram_reduction_pct": round(load_vram_reduction_pct, 2),
    }


@app.post("/generate")
def generate_synthetic_data(req: GenerateRequest) -> Dict[str, Any]:
    """
    Generates synthetic telemetry using both FP16 Baseline and 4-Bit NF4 models,
    tracks performance metrics, performs physiological sanity check, and returns benchmark JSON.
    Supports conditional switching between direct tensor generation (normal) and transformers.pipeline.
    """
    try:
        effective_use_pipeline = req.use_pipeline if req.use_pipeline is not None else USE_PIPELINE
        mode_label = "transformers.pipeline" if effective_use_pipeline else "Normal (Direct Tensor model.generate)"

        user_prompt = build_user_prompt(
            run_type=req.run_type,
            fitness_level=req.fitness_level,
            count=req.count,
            fields=req.fields,
            data_format=req.data_format,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        print(f"\n[*] Generating telemetry: {req.count} records for '{req.run_type}' ({req.fitness_level}) [{req.data_format}] via {mode_label}...")

        # 1. Run 4-Bit Quantized Model Inference
        print(f"  -> Running 4-Bit NF4 Quantized model inference ({mode_label})...")
        res_4bit = execute_inference(
            model=model_4bit,
            tokenizer=tokenizer,
            messages=messages,
            model_load_vram_mb=bit4_load_vram_mb,
            temperature=req.temperature,
            use_pipeline=effective_use_pipeline,
        )

        # 2. Run FP16 Baseline Model Inference
        print(f"  -> Running FP16 Baseline model inference ({mode_label})...")
        res_fp16 = execute_inference(
            model=model_fp16,
            tokenizer=tokenizer,
            messages=messages,
            model_load_vram_mb=fp16_load_vram_mb,
            temperature=req.temperature,
            use_pipeline=effective_use_pipeline,
        )

        # 3. Format beautiful multi-line display, reconcile kinematics & evaluate sanity checks
        out_4bit_clean, records_4bit = pretty_format_and_reconcile_output(res_4bit["output"], req.data_format)
        out_fp16_clean, records_fp16 = pretty_format_and_reconcile_output(res_fp16["output"], req.data_format)

        sanity_4bit = evaluate_physiological_sanity(records_4bit, req.run_type, req.fitness_level)
        sanity_fp16 = evaluate_physiological_sanity(records_fp16, req.run_type, req.fitness_level)

        # 4. Compute Benchmark Comparisons
        peak_vram_reduction_pct = (
            ((res_fp16["peak_gen_vram_mb"] - res_4bit["peak_gen_vram_mb"]) / max(res_fp16["peak_gen_vram_mb"], 1e-5)) * 100
        )
        latency_diff = round(res_fp16["latency_seconds"] - res_4bit["latency_seconds"], 3)
        speedup_factor = round(res_4bit["throughput_tokens_per_sec"] / max(res_fp16["throughput_tokens_per_sec"], 1e-5), 2)

        print(f"[✓] Generation complete! Latency: FP16={res_fp16['latency_seconds']}s, 4-Bit={res_4bit['latency_seconds']}s [Engine: {mode_label}]")

        return {
            "status": "success",
            "params": {
                "run_type": req.run_type,
                "fitness_level": req.fitness_level,
                "count": req.count,
                "fields": req.fields,
                "data_format": req.data_format,
                "temperature": req.temperature,
                "use_pipeline": effective_use_pipeline,
                "inference_mode": mode_label,
            },
            "quantized_4bit": {
                "model_name": f"{MODEL_ID} (4-Bit NF4)",
                "output": out_4bit_clean,
                "inference_engine": res_4bit.get("engine", mode_label),
                "load_vram_mb": round(bit4_load_vram_mb, 2),
                "peak_gen_vram_mb": res_4bit["peak_gen_vram_mb"],
                "latency_seconds": res_4bit["latency_seconds"],
                "token_count": res_4bit["token_count"],
                "throughput_tokens_per_sec": res_4bit["throughput_tokens_per_sec"],
                "sanity_check": sanity_4bit,
            },
            "baseline_fp16": {
                "model_name": f"{MODEL_ID} (FP16 Baseline)",
                "output": out_fp16_clean,
                "inference_engine": res_fp16.get("engine", mode_label),
                "load_vram_mb": round(fp16_load_vram_mb, 2),
                "peak_gen_vram_mb": res_fp16["peak_gen_vram_mb"],
                "latency_seconds": res_fp16["latency_seconds"],
                "token_count": res_fp16["token_count"],
                "throughput_tokens_per_sec": res_fp16["throughput_tokens_per_sec"],
                "sanity_check": sanity_fp16,
            },
            "comparison": {
                "load_vram_reduction_pct": round(load_vram_reduction_pct, 2),
                "peak_gen_vram_reduction_pct": round(peak_vram_reduction_pct, 2),
                "latency_diff_seconds": latency_diff,
                "speedup_factor": speedup_factor,
            },
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")


# ==============================================================================
# Step 8: Ngrok Tunnel Authentication & Uvicorn Launch
# ==============================================================================
def get_ngrok_token() -> str:
    """Retrieves the ngrok authtoken from env, Colab secrets, or user input."""
    # 1. Environment variable
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if token:
        return token

    # 2. Google Colab secrets
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get("NGROK_AUTHTOKEN")
        if token:
            return token.strip()
    except Exception:
        pass

    # 3. Interactive prompt
    try:
        token = input("🔑 Enter your ngrok authtoken (from dashboard.ngrok.com): ").strip()
    except EOFError:
        token = ""
    return token


async def start_server_async():
    """Configures ngrok public tunnel and starts Uvicorn FastAPI server asynchronously."""
    ngrok_token = get_ngrok_token()

    if not ngrok_token:
        print("\n⚠️ No ngrok authtoken provided! Running locally on port 8000 only.")
        public_url = "http://localhost:8000"
    else:
        print("\n[*] Establishing pyngrok secure tunnel...")
        try:
            ngrok.kill()
            ngrok.set_auth_token(ngrok_token)
            public_tunnel = ngrok.connect(8000)
            public_url = public_tunnel.public_url
        except Exception as e:
            print(f"❌ Failed to connect ngrok: {e}")
            public_url = "http://localhost:8000"

    print("\n" + "=" * 75)
    print("🚀 RUNNER SYNTHETIC DATA GENERATOR BACKEND IS ONLINE!")
    print("=" * 75)
    print(f"[*] Public ngrok Tunnel URL: {public_url}")
    print(f"[*] Health Check Endpoint:   {public_url}/health")
    print(f"[*] Generation Endpoint:     {public_url}/generate")
    print("=" * 75)
    print("👉 Copy the public ngrok URL above and paste it into 'runner_syntethic_data_client.py'!")
    print("=" * 75 + "\n")

    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Inside interactive Colab/IPython environment with running event loop
        loop.create_task(start_server_async())
    else:
        # Standard Python CLI execution
        asyncio.run(start_server_async())
