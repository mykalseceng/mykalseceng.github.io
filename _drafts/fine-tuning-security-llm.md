---
layout: post
title: "Fine-Tuning a Security Vulnerability Analysis LLM: A Complete Guide"
date: 2025-12-24
categories: [ai, security, machine-learning]
tags: [llm, fine-tuning, qlora, ollama, hackerone]
description: "How we built a specialized security analysis model using QLoRA, HackerOne data, and Ollama"
---

*How we built a specialized security analysis model using QLoRA, HackerOne data, and Ollama*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why Fine-Tune Instead of Training from Scratch?](#why-fine-tune-instead-of-training-from-scratch)
3. [Key Concepts Explained](#key-concepts-explained)
   - [QLoRA (Quantized Low-Rank Adaptation)](#qlora-quantized-low-rank-adaptation)
   - [ShareGPT Format](#sharegpt-format)
   - [GGUF Format](#gguf-format)
   - [Quantization](#quantization)
4. [The Pipeline Overview](#the-pipeline-overview)
5. [Implementation](#implementation)
   - [Step 1: Scraping HackerOne Reports](#step-1-scraping-hackerone-reports)
   - [Step 2: Formatting for Training](#step-2-formatting-for-training)
   - [Step 3: Train/Eval Split](#step-3-traineval-split)
   - [Step 4: Training with QLoRA](#step-4-training-with-qlora)
   - [Step 5: Merging the Adapter](#step-5-merging-the-adapter)
   - [Step 6: Converting to GGUF](#step-6-converting-to-gguf)
   - [Step 7: Quantization](#step-7-quantization)
   - [Step 8: Creating the Ollama Model](#step-8-creating-the-ollama-model)
6. [Testing and Validation](#testing-and-validation)
7. [Results and Lessons Learned](#results-and-lessons-learned)
8. [Conclusion](#conclusion)

---

## Introduction

Large Language Models (LLMs) are incredibly powerful, but general-purpose models often lack deep expertise in specialized domains. Security vulnerability analysis is one such domain where precision, technical accuracy, and domain-specific knowledge are critical.

In this guide, we walk through building a specialized security vulnerability analysis model by fine-tuning an existing LLM on real-world bug bounty reports from HackerOne. The result is a model that can analyze vulnerabilities, identify root causes, assess impact, and suggest mitigations—all running locally on your machine.

**What we built:**
- A 7B parameter model fine-tuned on ~4,000 real security vulnerability reports
- Runs locally via Ollama in a 4.7GB package
- Trained in under 1 hour on a single A10 GPU (~$0.75/hour)

---

## Why Fine-Tune Instead of Training from Scratch?

When I first started this project, I considered training a security analysis model from scratch. After running the numbers, it became clear why that approach is reserved for well-funded AI labs.

Training an LLM from the ground up is a massive undertaking. You need billions of tokens of training data—terabytes of text that must be scraped, cleaned, and processed. The compute requirements are staggering: a 7B parameter model typically requires 1-2 trillion tokens and around 100,000 GPU hours on expensive A100 clusters. We're talking months of training time and costs easily exceeding $100,000.

Fine-tuning with QLoRA flips this equation entirely. We used just 4,000 well-curated examples—a few million tokens rather than trillions. Training completed in under an hour on a single A10 GPU. The total cost? Less than a dollar.

The reason this works is that pre-trained models like Qwen2.5-Coder have already learned the hard stuff: natural language understanding, code comprehension, general reasoning patterns, and technical vocabulary. We're not teaching the model to "think" from scratch. We're teaching an already-fluent model the specific format and domain knowledge for security vulnerability analysis. It's the difference between teaching someone a new language versus teaching a fluent speaker industry jargon.

---

## Key Concepts Explained

### QLoRA (Quantized Low-Rank Adaptation)

**The Problem:** Fine-tuning a 7B model normally requires ~28GB of GPU memory just for the model weights, plus optimizer states, gradients, and activations. This exceeds most consumer GPUs.

**The Solution:** QLoRA combines two techniques:

#### 1. Quantization (The "Q")
Instead of storing weights in 16-bit floating point (2 bytes per parameter), we compress them to 4-bit (0.5 bytes per parameter):

```
7B parameters × 2 bytes = 14GB (FP16)
7B parameters × 0.5 bytes ≈ 3.5GB (4-bit, weights only)
```

In practice, the actual VRAM footprint is higher—typically 4-4.5 GB for a 7B 4-bit model—due to quantization metadata (scales and zero points per tensor group). During training, activations and optimizer state dominate memory beyond the weights. During inference, KV cache grows with context length. The base model is frozen in this 4-bit NF4 format—we never modify these weights.

#### 2. Low-Rank Adaptation (The "LoRA")
Instead of updating all 7 billion parameters, we add small "adapter" matrices to specific layers:

```
Original: Y = X × W              (W is 4096 × 4096 = 16M parameters)
LoRA:     Y = X × W + X × A × B  (A is 4096 × 32, B is 32 × 4096 = 262K parameters)
```

With rank `r=32`, each adapted layer adds only ~0.3% parameters. In our setup, only LoRA adapter weights are trainable.

**Visual Representation:**

```
┌─────────────────────────────────────────────┐
│           Frozen Base Model (4-bit)          │
│  ┌─────────────────────────────────────────┐ │
│  │  Attention Layer                        │ │
│  │  ┌─────────┐    ┌─────────┐            │ │
│  │  │    W    │ +  │  A × B  │ ← Trainable│ │
│  │  │ (frozen)│    │ (LoRA)  │            │ │
│  │  └─────────┘    └─────────┘            │ │
│  └─────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────┐ │
│  │  FFN Layer                              │ │
│  │  ┌─────────┐    ┌─────────┐            │ │
│  │  │    W    │ +  │  A × B  │ ← Trainable│ │
│  │  │ (frozen)│    │ (LoRA)  │            │ │
│  │  └─────────┘    └─────────┘            │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘

Base Model: 7.6B parameters (frozen, 4-bit) ≈ 4-4.5GB VRAM
LoRA Adapters: ~80M parameters (trainable, 16-bit) ≈ 160MB VRAM
```

Note: The 80M trainable parameter count assumes LoRA is applied to all 28 transformer blocks across all 7 target modules (q/k/v/o projections + gate/up/down FFN layers). The embedding and LM head layers remain frozen and are not adapted.

### ShareGPT Format

ShareGPT is a JSON format for representing multi-turn conversations, named after a website where users shared ChatGPT conversations.

**Structure:**
```json
{
  "conversations": [
    {"from": "human", "value": "User's first message"},
    {"from": "gpt", "value": "Assistant's response"},
    {"from": "human", "value": "User's follow-up"},
    {"from": "gpt", "value": "Assistant's second response"}
  ]
}
```

**Why this format?**
- Standardized across many fine-tuning tools (Unsloth, Axolotl, LLaMA-Factory)
- Supports multi-turn conversations naturally
- Role labels (`human`/`gpt`) map cleanly to chat templates
- Easy to convert from various sources

### GGUF Format

GGUF (GPT-Generated Unified Format) is a binary file format designed for efficient LLM inference, created by the llama.cpp project.

**Benefits:**
- Single file contains model weights, tokenizer, and metadata
- Memory-mapped loading (fast startup, low RAM usage)
- Supports various quantization levels
- Cross-platform (CPU, CUDA, Metal, ROCm)
- Used by Ollama, llama.cpp, LM Studio, and others

**File Structure:**
```
┌────────────────────────────┐
│  Magic Number (GGUF)       │
├────────────────────────────┤
│  Version                   │
├────────────────────────────┤
│  Metadata (key-value)      │
│  - architecture: qwen2     │
│  - context_length: 32768   │
│  - tokenizer.model: gpt2   │
│  - ...                     │
├────────────────────────────┤
│  Tensor Info               │
│  - name, shape, type       │
├────────────────────────────┤
│  Tensor Data               │
│  - Quantized weights       │
└────────────────────────────┘
```

### Quantization

Quantization reduces model precision to decrease size and increase inference speed. Think of it as controlled compression—we sacrifice some numerical precision to make models small enough to run on consumer hardware.

**Important distinction:** There are two different quantization stages in our pipeline, and they serve different purposes:

1. **Training-time quantization (NF4/QLoRA):** During fine-tuning, we load the base model in 4-bit NF4 format using bitsandbytes. This reduces training VRAM requirements. The LoRA adapters themselves remain in 16-bit precision.

2. **Inference-time quantization (GGUF Q4_K_M):** After training and merging, we quantize the full merged model for deployment. This uses llama.cpp's k-quant methods, which are different from NF4.

These are separate steps with different formats and tools. The training quantization (bitsandbytes) is not preserved—we merge to FP16 first, then re-quantize for inference.

For inference, we chose Q4_K_M. The naming breaks down: "Q4" indicates 4-bit quantization, "K" refers to the k-quant method which groups weights intelligently, and "M" means medium quality. The k-quant approach applies higher precision (Q6_K) to critical layers like the output head while using Q4_K for attention and FFN layers.

At full FP16 precision, our model consumes 14 GB. Q4_K_M shrinks this to 4.4 GB while retaining good quality. Going lower (Q2_K) degrades quality noticeably—for security analysis where accuracy matters, Q4_K_M hits the right balance.

---

## The Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPARATION                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  HackerOne  │───▶│  Raw JSONL  │───▶│  ShareGPT   │                  │
│  │   Scraper   │    │  (h1.jsonl) │    │   Format    │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│        │                  │                   │                          │
│        │            5000 reports         3934 convos                     │
│        ▼                                      │                          │
│   hackerone-                                  ▼                          │
│   scraper-           format_dataset.py  ┌─────────────┐                  │
│   threaded.py                           │   Split     │                  │
│                                         │  95% / 5%   │                  │
│                                         └─────────────┘                  │
│                                               │                          │
│                                    ┌──────────┴──────────┐               │
│                                    ▼                     ▼               │
│                              train.jsonl           eval.jsonl            │
│                              (3737 rows)           (197 rows)            │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              TRAINING                                     │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Lambda Labs A10 GPU                           │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │    │
│  │  │  Qwen2.5    │    │   QLoRA     │    │   LoRA      │         │    │
│  │  │  7B (4-bit) │ +  │  Training   │ =  │  Adapter    │         │    │
│  │  │   Frozen    │    │  (Unsloth)  │    │  (300 MB)   │         │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘         │    │
│  │                                                                  │    │
│  │  Training: 936 steps, 2 epochs, ~49 minutes                     │    │
│  │  Loss: 1.53 → 0.54                                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│                                    ▼                                     │
│                            security_adapter/                             │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           POST-PROCESSING                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   Merge     │───▶│  Convert    │───▶│  Quantize   │                  │
│  │  Adapter    │    │  to GGUF    │    │  Q4_K_M     │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│        │                  │                   │                          │
│     14 GB              14 GB              4.4 GB                         │
│  (safetensors)       (F16 GGUF)        (Q4 GGUF)                        │
│                                               │                          │
│                                               ▼                          │
│                                      ┌─────────────┐                     │
│                                      │   Ollama    │                     │
│                                      │   Model     │                     │
│                                      │  (sec-vuln) │                     │
│                                      └─────────────┘                     │
│                                           4.7 GB                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Step 1: Scraping HackerOne Reports

HackerOne's Hacktivity page lists publicly disclosed bug bounty reports—a treasure trove of real-world vulnerability data. We built a threaded scraper to collect these efficiently while respecting rate limits.

The key insight is that while listing reports is fast, fetching full details requires individual API calls. We parallelize the enrichment step using a thread pool:

```python
with ThreadPoolExecutor(max_workers=self.workers) as executor:
    enriched = list(executor.map(self.fetch_report, reports))
```

Each worker fetches report details and respects a rate limit (0.5 seconds between requests) to avoid overwhelming the API. The scraper writes each enriched report as a JSON line, giving us a streaming-friendly format.

After about 45 minutes of polite scraping, we collected approximately 5,000 reports totaling around 50 MB. Each record contains the vulnerability title, severity rating, program name, disclosure date, and most importantly—the full report content with technical details.

---

### Step 2: Formatting for Training

Raw vulnerability reports aren't suitable for training—we need to convert them into conversations that teach the model both what questions to expect and how to structure its analysis.

The formatting script transforms each report into a ShareGPT-style conversation. The "human" turn presents the vulnerability context (program, title, severity, URL) and asks for analysis. The "gpt" turn provides a structured response with clear sections: vulnerability details, impact assessment, and a summary.

```python
return {
    "conversations": [
        {"from": "human", "value": human_prompt},   # Analysis request
        {"from": "gpt", "value": gpt_response}       # Structured response
    ]
}
```

The key design decision was the output format. By consistently structuring responses with markdown headers and sections, we teach the model to produce organized, scannable analyses rather than wall-of-text responses. The prompt biases the model away from step-by-step exploitation details—we want it to be more useful for defenders than attackers. Note that this training-time guidance reduces but doesn't prevent exploitation-focused outputs; determined prompt engineering could likely elicit such content.

After filtering out incomplete reports (missing severity, empty content, etc.), we ended up with 3,934 valid training conversations from our initial 5,000 scraped reports.

---

### Step 3: Train/Eval Split

Before training, we need to hold out some data for evaluation. A random split works, but we used deterministic hashing for reproducibility—the same data always ends up in the same split, even if you rerun the script or add new records later.

```python
h = hashlib.sha256(key_for(row).encode()).hexdigest()
x = int(h[:8], 16) / 0xFFFFFFFF  # Normalize to 0-1
if x < cutoff:
    # Goes to eval set
```

The hash is computed from a stable identifier (report ID or URL), converted to a number between 0 and 1, then compared against the split threshold. This approach has a nice property: when you add new training data later, existing records stay in their original split, so you're not accidentally contaminating your eval set.

We used a 95/5 split, giving us 3,737 training examples and 197 held out for evaluation.

---

### Step 4: Training with QLoRA

This is where the magic happens. We used Unsloth, a library that provides optimized QLoRA training with significant speed improvements over vanilla implementations.

The training script does four key things. First, it loads the base model in 4-bit quantization—this is the "Q" in QLoRA:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)
```

Next, it attaches LoRA adapters to specific layers. We target the attention projections (q, k, v, o) and feed-forward layers (gate, up, down). The rank of 32 gives us enough capacity to learn the security domain without going overboard on trainable parameters:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

The training itself uses Hugging Face's SFTTrainer. We ran for 2 epochs with a batch size of 2 and gradient accumulation of 4, giving an effective batch size of 8. The learning rate of 2e-4 is fairly standard for LoRA fine-tuning.

Since QLoRA as implemented here requires CUDA and bitsandbytes (AMD ROCm and Apple Metal paths are experimental or unsupported for training), we rented an A10 from Lambda Labs (~$0.75/hour). The training took 49 minutes to complete 936 steps.

We watched the training loss drop from 1.53 to 0.54. A word of caution: this is training loss only, not a generalization metric. With just 4,000 highly structured examples over 2 epochs, a sharp loss drop can indicate memorization rather than learning. We didn't track eval loss during training, so the later testing section is our primary validation. Ideally, you'd monitor eval loss to detect overfitting.

The headline number: in our configuration, we trained ~80.7 million parameters out of 7.7 billion—about 1.05% of the model. This small adapter is sufficient to shift the model's output format and domain vocabulary.

---

### Step 5: Merging the Adapter

After training completes, we have the base model (frozen) and the LoRA adapter weights (trained) as separate entities. For deployment, we need to merge them into a single model.

The merge operation "bakes in" the adaptations permanently by computing `W' = W + A×B` for each adapted layer. Once merged, the model behaves identically but no longer requires the PEFT infrastructure at inference time.

```python
model = model.merge_and_unload()
model.save_pretrained_merged("./security_merged_fp16", tokenizer, save_method="merged_16bit")
```

We saved in FP16 format for compatibility with the GGUF conversion pipeline. The merged model weighs in at approximately 14 GB, split across four safetensor files plus tokenizer assets. This is our full-fidelity checkpoint before any inference-time compression.

---

### Step 6: Converting to GGUF

The merged model is in HuggingFace's safetensors format—great for Python-based inference but not optimized for efficient local deployment. We need to convert it to GGUF, the format used by llama.cpp and Ollama.

GGUF (GPT-Generated Unified Format) is a single-file format that packages weights, tokenizer, and metadata together. It supports memory-mapped loading, meaning the model can start generating almost immediately without loading everything into RAM first.

The conversion uses llama.cpp's `convert_hf_to_gguf.py` script:

```bash
python llama.cpp/convert_hf_to_gguf.py security_merged_fp16 \
  --outtype f16 --outfile security_merged.f16.gguf
```

The script reads the safetensors, extracts architecture metadata (context length, embedding dimensions, attention heads), and writes everything into a single 14 GB GGUF file. This intermediate F16 file preserves full precision—we'll quantize it in the next step.

---

### Step 7: Quantization

A 14 GB model is unwieldy for local use—that's a lot of RAM to dedicate to a single application. The quantization step compresses the model to a fraction of its size.

```bash
./llama.cpp/build/bin/llama-quantize \
  security_merged.f16.gguf security_merged.Q4_K_M.gguf Q4_K_M
```

The quantizer processes each tensor, converting 16-bit floats to 4-bit integers with intelligent grouping. It applies different precision to different layers: the output head gets Q6_K (slightly higher precision) since errors there directly impact token probabilities, while attention and FFN layers use Q4_K.

The result is dramatic: 14 GB compresses to 4.4 GB—just 31% of the original size. Quality loss is often modest but task-dependent; we found it acceptable in our tests. The model still produces coherent, well-structured security analyses.

---

### Step 8: Creating the Ollama Model

The final step wraps our quantized model in Ollama's packaging format. A Modelfile specifies the base model, generation parameters, and system prompt:

```dockerfile
FROM ./security_merged.Q4_K_M.gguf
PARAMETER temperature 0.2
PARAMETER num_ctx 4096
SYSTEM """You are a security vulnerability analysis assistant.
Do not provide step-by-step exploitation.
Provide root cause, impact, constraints, mitigations, and detection ideas."""
```

We set a low temperature (0.2) for more deterministic, focused responses—important for technical analysis where we want consistency over creativity. The system prompt reinforces the model's role and biases outputs toward defensive analysis rather than exploitation guidance (though this is a soft constraint, not a hard guarantee).

Running `ollama create sec-vuln -f Modelfile` builds the model. Ollama creates a layered package: the GGUF weights, the system prompt, and the generation parameters each become separate layers. The final model weighs in at 4.7 GB and is immediately available for inference via `ollama run sec-vuln`.

---

## Testing and Validation

With the model deployed, we needed to verify that our fine-tuning actually worked. Did the model learn the security analysis patterns, or did we just waste an hour of GPU time?

### Test 1: Verify Training Data Incorporation

The first test used an exact prompt from the training data—essentially asking the model to recall what it learned:

**Input:**
```
Analyze this publicly disclosed vulnerability report.
Provide the likely root cause, affected component, and risk in plain language.
If impact is clear, summarize it briefly. Do not include step-by-step exploitation.

Context:
- Program: ibb
- Title: CVE-2022-32208: FTP-KRB bad message verification
- Severity: Low
- URL: https://hackerone.com/reports/1614332
```

**Model Output:**
```markdown
## CVE-2022-32208: FTP-KRB bad message verification

**Severity:** Low
**Program:** ibb

### Vulnerability Details

CVE-2022-32208: FTP-KRB bad message verification Project curl Security Advisory,
July 19th 2022 VULNERABILITY When using the curl library to perform a secure FTP
transfer with Kerberos authentication enabled, it is possible for an attacker to
cause the application to free memory that it has not allocated. This flaw arises
from a failure in the message verification routine...

### Impact

Man-in-the-middle attackers can inject data into FTP sessions.

### Summary

Improper message verification in curl's Kerberos FTP implementation.
```

The output matches the training format exactly—markdown headers, structured sections, clear severity rating. This confirms the model internalized the analysis structure we trained on.

### Test 2: Generalization to Novel Vulnerabilities

Memorization is easy; generalization is the real test. We fed the model a vulnerability type it definitely didn't see in training—a hypothetical SSRF in an image proxy:

**Input:**
```
Analyze: SSRF vulnerability in an image proxy that fetches user-provided URLs without validation
```

**Model Output:**
```markdown
## Server-Side Request Forgery in Image Proxy

**Severity:** High

### Vulnerability Details

The image proxy service accepts user-supplied URLs and fetches them server-side
without proper validation. This allows attackers to:
- Access internal network resources (169.254.169.254 for cloud metadata)
- Scan internal ports and services
- Bypass firewall restrictions

### Root Cause

Missing URL validation before server-side HTTP requests. The application trusts
user input without checking:
- Protocol scheme (allowing file://, gopher://, etc.)
- Host restrictions (allowing internal IPs)
- Port restrictions

### Impact

- Cloud credential theft via metadata endpoints
- Internal service enumeration
- Potential RCE through vulnerable internal services

### Mitigations

1. Allowlist permitted domains/protocols
2. Block RFC1918 and link-local addresses
3. Use a URL parser to validate before fetching
4. Implement request timeouts and size limits
```

The model applies a structured analysis to a novel scenario. It correctly identifies the vulnerability type, explains the attack surface (cloud metadata, internal networks), traces the root cause (missing validation), and suggests appropriate mitigations.

**Important caveat:** This test demonstrates the model produces reasonable security analysis, but it doesn't prove the improvement came from fine-tuning. The base Qwen2.5-Coder model likely has substantial security knowledge from its pre-training data. A rigorous evaluation would require an A/B comparison against the unfine-tuned base model to isolate the fine-tuning contribution. What we can confidently say is that the model learned our output format and structure—whether the domain knowledge improved significantly requires more controlled testing.

---

## Results and Lessons Learned

The pipeline produced a progression of artifacts, each building on the last. We started with 50 MB of raw scraped reports, formatted them into 40 MB of training conversations, then split off 5% for evaluation. The LoRA adapter weights are surprisingly compact at ~300 MB in our run—that's all the domain-specific knowledge we added. The merged FP16 model balloons to 14 GB, then quantization compresses it back down to 4.4 GB for the final Ollama-ready model.

The total cost was remarkable: under a dollar. Lambda Labs charges roughly $0.75/hour for an A10 GPU, and we finished training in 49 minutes. The scraping was free (just time), and all the tools (Unsloth, llama.cpp, Ollama) are open source.

Several lessons emerged from this project:

**Data quality trumps quantity.** We had just 4,000 examples, but they were real vulnerability reports with structured analyses. This outperformed experiments with larger but noisier datasets. If you're fine-tuning, spend time curating your examples.

**Output format matters more than you'd think.** By consistently formatting responses with markdown headers and sections, the model learned to produce scannable, organized analyses. The format is part of what you're teaching.

**QLoRA is enough for domain adaptation.** Training just 1% of parameters sounds limiting, but it's sufficient to shift the model's behavior toward your domain. We weren't trying to give the model new reasoning capabilities—we were teaching it security vocabulary and analysis patterns.

**For our use case, Q4_K_M was a good tradeoff.** The quantization reduced size by 70% with no perceptible quality degradation in our testing. Quality deltas can be subtle and task-dependent, so evaluate on your specific use case before committing to aggressive quantization.

---

## Conclusion

What started as an experiment in domain adaptation became a complete pipeline for building specialized AI models. We scraped 5,000 real bug bounty reports from HackerOne, formatted them into structured training conversations, fine-tuned with QLoRA on a rented A10 GPU for less than a dollar, quantized the result to a portable 4.4 GB package, and deployed it locally via Ollama.

The result is a model that reliably follows our security-analysis format and produces structured, security-relevant outputs—all running on a laptop without internet access. It won't replace human security researchers, but it's a useful tool for quickly analyzing vulnerability reports, brainstorming impacts, or learning how different vulnerability classes work.

```bash
ollama run sec-vuln "Analyze: XSS in markdown preview feature"
```

The broader lesson is that this approach generalizes. Any domain with structured examples can benefit from fine-tuning: legal document analysis, medical records processing, financial report summarization, or any specialized text understanding task. The key ingredients are quality training data, the right format, and a few hours of experimentation. The cost barrier that once made custom LLMs a luxury is effectively gone.

---

## Repository Structure

```
scrapers/
├── hackerone-scraper-threaded.py  # Step 1: Data collection
├── format_dataset.py              # Step 2: Format conversion
├── split_jsonl.py                 # Step 3: Train/eval split
├── train_unsloth_from_train_strict.py  # Step 4: Training
├── merge_adapter.py               # Step 5: Merge adapter
├── build_ollama_model.sh          # Full automation script
├── Modelfile                      # Ollama configuration
├── h1.jsonl                       # Raw data
├── train_strict.jsonl             # Formatted data
├── train_strict_train.jsonl       # Training split
├── train_strict_eval.jsonl        # Eval split
├── security_adapter/              # LoRA weights
├── security_merged_fp16/          # Merged model
├── security_merged.f16.gguf       # GGUF F16
├── security_merged.Q4_K_M.gguf    # Quantized model
└── llama.cpp/                     # Conversion tools
```
