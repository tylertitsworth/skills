---
name: huggingface-transformers
description: >
  Load, download, and manage HuggingFace models and datasets. Use when:
  (1) Downloading models and datasets (Python API, HF CLI, git-xet),
  (2) Loading models with AutoModel classes (dtype, device_map, quantization),
  (3) Using tokenizers (encoding, decoding, chat templates, special tokens),
  (4) Managing the HF cache (scan, delete, symlink structure),
  (5) Using the datasets library (load, filter, map, stream, save),
  (6) PEFT/LoRA adapter loading and merging,
  (7) Generation configuration (temperature, top_p, top_k, beam search),
  (8) Model quantization (BitsAndBytes, GPTQ, AWQ),
  (9) Using pipelines for quick inference.
---

# HuggingFace Transformers

HuggingFace ecosystem for model and dataset management. Core libraries: `transformers`, `datasets`, `huggingface_hub`, `tokenizers`, `peft`. Version: **4.46.x+**.

## Downloading Models and Datasets

### HF CLI (`hf` command)

The `hf` CLI is the fastest way to download from the Hub. Add to container image: `huggingface_hub` (includes `hf` CLI) and `hf-xet` (Xet backend for faster downloads).

```bash
# Download entire model
hf download meta-llama/Llama-3.1-8B-Instruct

# Download specific files
hf download meta-llama/Llama-3.1-8B-Instruct config.json tokenizer.json

# Download with pattern matching
hf download meta-llama/Llama-3.1-8B-Instruct --include "*.safetensors" --exclude "*.bin"

# Download to specific directory (instead of cache)
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir /models/llama-8b

# Download a dataset
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

# Download specific dataset files
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset --include "data/train_sft*"

# Download a specific revision/branch
hf download meta-llama/Llama-3.1-8B-Instruct --revision main

# Quiet mode (just print path)
hf download gpt2 config.json --quiet
```

### HF CLI Download Options

| Option | Purpose |
|---|---|
| `--repo-type` | Repository type: `model` (default), `dataset`, `space` |
| `--revision` | Branch, tag, or commit hash |
| `--include` | Glob pattern for files to include |
| `--exclude` | Glob pattern for files to exclude |
| `--local-dir` | Download to directory (copies, no symlinks) |
| `--local-dir-use-symlinks` | Use symlinks instead of copies | 
| `--cache-dir` | Custom cache directory |
| `--token` | Auth token (or use `hf auth login`) |
| `--quiet` | Only print the path |
| `--force-download` | Re-download even if cached |
| `--resume-download` | Resume interrupted download |

### HF CLI Authentication

```bash
# Interactive login (stores token)
hf auth login

# Login with token from env var (for K8s containers)
hf auth login --token $HF_TOKEN --add-to-git-credential

# Check who you're logged in as
hf auth whoami
```

In K8s pods, set `HF_TOKEN` env var from a Secret — the SDK and CLI auto-detect it:

```yaml
env:
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-secret
      key: token
```

### Xet Storage (Faster Downloads)

Xet is HuggingFace's modern storage backend replacing Git LFS. It provides chunk-level deduplication and faster downloads.

**Python (automatic):** `huggingface_hub >= 0.32.0` installs `hf_xet` automatically. No code changes needed — `from_pretrained()`, `hf download`, and `snapshot_download()` all use Xet transparently.

**Git clone with Xet:**
```bash
# Install git-xet extension (in container image)
# curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | sh

# Clone uses Xet automatically for Xet-backed repos
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

**Xet cache:** Separate from the HF cache. Chunks are cached locally for deduplication across downloads. Manage with `huggingface-cli cache` commands.

### Python Download API

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download single file
path = hf_hub_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    filename="config.json",
    token=True,                    # use stored token
)

# Download entire repo
path = snapshot_download(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    allow_patterns=["*.safetensors", "*.json"],
    ignore_patterns=["*.bin"],
    cache_dir="/models/cache",
    token=True,
)

# Download dataset
path = snapshot_download(
    repo_id="HuggingFaceH4/ultrachat_200k",
    repo_type="dataset",
)
```

### Environment Variables for Downloads

| Variable | Purpose | Default |
|---|---|---|
| `HF_TOKEN` | Authentication token | None |
| `HF_HOME` | Root for all HF data | `~/.cache/huggingface` |
| `HF_HUB_CACHE` | Model/dataset cache dir | `$HF_HOME/hub` |
| `HF_HUB_ENABLE_HF_TRANSFER` | Use hf_transfer for faster downloads | `0` |
| `HF_HUB_OFFLINE` | Offline mode (cache only) | `0` |
| `HF_HUB_DOWNLOAD_TIMEOUT` | Download timeout in seconds | `10` |
| `TRANSFORMERS_CACHE` | Legacy cache dir (use HF_HUB_CACHE) | |
| `HF_DATASETS_CACHE` | Datasets cache dir | `$HF_HOME/datasets` |

## Cache Management

### HF CLI Cache Commands

```bash
# Scan cache (show disk usage by repo)
hf cache scan

# Delete specific revisions interactively
hf cache delete

# Delete specific repos
hf cache delete --repo-id meta-llama/Llama-3.1-8B-Instruct
```

### Python Cache Management

```python
from huggingface_hub import scan_cache_dir, HfApi

# Scan cache
cache = scan_cache_dir()
print(f"Total: {cache.size_on_disk_str}")
for repo in cache.repos:
    print(f"  {repo.repo_id}: {repo.size_on_disk_str} ({repo.nb_files} files)")

# Delete specific revisions
delete_strategy = cache.delete_revisions("abc123def456")
print(f"Will free {delete_strategy.expected_freed_size_str}")
delete_strategy.execute()
```

### Cache Structure

```
~/.cache/huggingface/hub/
├── models--meta-llama--Llama-3.1-8B-Instruct/
│   ├── refs/
│   │   └── main                    # points to snapshot hash
│   ├── snapshots/
│   │   └── abc123.../              # files (or symlinks to blobs)
│   └── blobs/
│       └── <sha256>                # actual file content
└── datasets--HuggingFaceH4--ultrachat_200k/
    └── ...
```

## Loading Models

### AutoModel Classes

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",                    # auto-detect dtype
    device_map="auto",                     # auto-shard across GPUs
    attn_implementation="flash_attention_2",  # or "sdpa", "eager"
    trust_remote_code=True,                # for custom model code
    cache_dir="/models/cache",             # custom cache location
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    padding_side="left",                   # left-pad for batch generation
    trust_remote_code=True,
)
```

### from_pretrained Settings

| Setting | Purpose | Values |
|---|---|---|
| `torch_dtype` | Weight dtype | `"auto"`, `torch.bfloat16`, `torch.float16` |
| `device_map` | Device placement | `"auto"`, `"cpu"`, `"cuda:0"`, custom dict |
| `attn_implementation` | Attention backend | `"flash_attention_2"`, `"sdpa"`, `"eager"` |
| `trust_remote_code` | Allow custom model code | `True`/`False` |
| `revision` | Model revision | Branch/tag/commit hash |
| `cache_dir` | Override cache directory | Path string |
| `low_cpu_mem_usage` | Reduce CPU memory during loading | `True`/`False` |
| `max_memory` | Max memory per device | `{"cuda:0": "20GiB", "cpu": "40GiB"}` |

### device_map Options

| Value | Behavior |
|---|---|
| `"auto"` | Automatically shard across all available GPUs, overflow to CPU |
| `"balanced"` | Evenly split layers across GPUs |
| `"balanced_low_0"` | Balance but leave GPU 0 lighter (for generation) |
| `"sequential"` | Fill GPUs sequentially |
| Custom dict | Manual layer-to-device mapping |

## Tokenizers

### Core Operations

```python
# Encode
tokens = tokenizer("Hello world", return_tensors="pt")
# tokens.input_ids, tokens.attention_mask

# Decode
text = tokenizer.decode(token_ids, skip_special_tokens=True)

# Batch encode
batch = tokenizer(
    ["text1", "text2"],
    padding=True,
    truncation=True,
    max_length=4096,
    return_tensors="pt",
)
```

### Chat Templates

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,                # return string, not tokens
    add_generation_prompt=True,    # add assistant turn prefix
)

# Tokenize directly
tokens = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
)
```

## Datasets Library

### Loading Datasets

```python
from datasets import load_dataset

# From Hub
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# Specific config/subset
ds = load_dataset("gsm8k", "main", split="train")

# From local files
ds = load_dataset("json", data_files="train.jsonl")
ds = load_dataset("parquet", data_files="data/*.parquet")
ds = load_dataset("csv", data_files="data.csv")

# Streaming (no download, lazy loading)
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
for example in ds:
    process(example)
```

### Dataset Operations

```python
# Filter
ds = ds.filter(lambda x: len(x["text"]) > 100)

# Map (transform)
ds = ds.map(tokenize_fn, batched=True, num_proc=8, remove_columns=["text"])

# Select columns
ds = ds.select_columns(["input_ids", "attention_mask", "labels"])

# Shuffle and select
ds = ds.shuffle(seed=42).select(range(10000))

# Train/test split
ds = ds.train_test_split(test_size=0.1, seed=42)

# Save processed dataset
ds.save_to_disk("/data/processed")
ds.to_parquet("/data/processed.parquet")

# Load from disk
from datasets import load_from_disk
ds = load_from_disk("/data/processed")
```

### Dataset Caching

The datasets library caches processed results automatically. Control with:

| Setting | Purpose |
|---|---|
| `HF_DATASETS_CACHE` | Cache directory |
| `ds.map(..., cache_file_name=...)` | Explicit cache file |
| `ds.map(..., load_from_cache_file=False)` | Disable cache |
| `datasets.disable_caching()` | Global cache disable |

## PEFT / LoRA

### Loading Adapters

```python
from peft import PeftModel, PeftConfig

# Load base model + adapter
config = PeftConfig.from_pretrained("my-org/llama-8b-lora")
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, "my-org/llama-8b-lora")

# Merge adapter into base model (for inference)
model = model.merge_and_unload()
```

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                              # rank
    lora_alpha=32,                     # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```

## Generation Configuration

### GenerationConfig Settings

| Setting | Purpose | Default |
|---|---|---|
| `max_new_tokens` | Max tokens to generate | Model default |
| `temperature` | Sampling randomness (0 = greedy) | `1.0` |
| `top_p` | Nucleus sampling threshold | `1.0` |
| `top_k` | Top-K sampling | `50` |
| `do_sample` | Enable sampling (vs greedy) | `False` |
| `num_beams` | Beam search width | `1` |
| `repetition_penalty` | Penalize repeated tokens | `1.0` |
| `no_repeat_ngram_size` | Block repeated n-grams | `0` |
| `stop_strings` | Stop on these strings | None |
| `eos_token_id` | End-of-sequence token(s) | Model default |

```python
output = model.generate(
    **tokens,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.1,
)
```

## Model Quantization

### BitsAndBytes (4-bit / 8-bit)

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # nf4 or fp4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### Pre-Quantized Models (GPTQ, AWQ)

```python
# GPTQ — auto-detected from config
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GPTQ")

# AWQ — auto-detected from config
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-AWQ")
```

## Pipelines (Quick Inference)

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

result = pipe(
    [{"role": "user", "content": "What is 2+2?"}],
    max_new_tokens=256,
    temperature=0.7,
)
```

## Debugging

See `references/troubleshooting.md` for common loading, download, and generation issues.

## Reference

- [Transformers docs](https://huggingface.co/docs/transformers/)
- [HuggingFace Hub docs](https://huggingface.co/docs/huggingface_hub/)
- [Datasets docs](https://huggingface.co/docs/datasets/)
- [PEFT docs](https://huggingface.co/docs/peft/)
- [HF CLI reference](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
- [Xet storage](https://huggingface.co/docs/hub/en/xet/using-xet-storage)
- `references/troubleshooting.md` — common errors and fixes
