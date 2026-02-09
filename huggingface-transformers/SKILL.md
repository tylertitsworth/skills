---
name: huggingface-transformers
description: >
  Work with Hugging Face Transformers — the standard library for pretrained models, fine-tuning,
  and inference. Use when: (1) Loading models and tokenizers from the Hub, (2) Fine-tuning with
  Trainer API, (3) Using PEFT/LoRA for parameter-efficient fine-tuning, (4) Building inference
  pipelines, (5) Working with tokenizers (encoding, padding, truncation, special tokens),
  (6) Managing datasets with the datasets library, (7) Pushing/pulling models from Hugging Face
  Hub, (8) Configuring training arguments (learning rate, batch size, mixed precision, gradient
  accumulation), (9) Debugging tokenization or model loading issues.
---

# Hugging Face Transformers

The standard Python library for pretrained models. Version: **4.46+**. Covers `transformers`, `datasets`, `peft`, `accelerate`, and the Hub.

## Setup

```bash
pip install transformers datasets accelerate peft
# Login for gated models (Llama, Gemma, etc.)
huggingface-cli login
```

## Loading Models

### AutoModel Pattern

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",          # uses model's native dtype
    device_map="auto",           # auto-shard across GPUs
    attn_implementation="sdpa",  # scaled_dot_product_attention
)
```

### AutoModel Variants

| Class | Task |
|-------|------|
| `AutoModelForCausalLM` | Text generation (GPT, Llama, Mistral) |
| `AutoModelForSeq2SeqLM` | Encoder-decoder (T5, BART) |
| `AutoModelForSequenceClassification` | Text classification |
| `AutoModelForTokenClassification` | NER, POS tagging |
| `AutoModelForQuestionAnswering` | Extractive QA |
| `AutoModelForMaskedLM` | Fill-mask (BERT, RoBERTa) |
| `AutoModel` | Base model (no task head) |

### Memory-Efficient Loading

```python
# 4-bit quantization with bitsandbytes
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# 8-bit loading
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
```

## Tokenizers

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Basic encoding
inputs = tokenizer("Hello, world!", return_tensors="pt")
# {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}

# Batch encoding with padding and truncation
inputs = tokenizer(
    ["Short text.", "A much longer piece of text that needs truncation."],
    padding=True,          # pad to longest in batch
    truncation=True,       # truncate to max_length
    max_length=512,
    return_tensors="pt",
)

# Decode
text = tokenizer.decode(token_ids, skip_special_tokens=True)
texts = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)

# Chat template (for instruct models)
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is LoRA?"},
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
```

### Tokenizer Configuration

```python
# Set pad token (required for batch training, many models lack one)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# Or add a new pad token
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))
```

## Pipelines (Quick Inference)

```python
from transformers import pipeline

# Text generation
gen = pipeline("text-generation", model=model_name, device_map="auto", torch_dtype="auto")
output = gen("Explain transformers:", max_new_tokens=200, temperature=0.7)

# Classification
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("This model is amazing!")  # [{'label': 'POSITIVE', 'score': 0.99}]

# Embeddings
embedder = pipeline("feature-extraction", model="BAAI/bge-small-en-v1.5")
embedding = embedder("Hello world", return_tensors=True)
```

## Generation

```python
inputs = tokenizer("The key to efficient training is", return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.1,
    num_beams=1,              # 1 = no beam search
    pad_token_id=tokenizer.pad_token_id,
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streaming generation
from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_special_tokens=True)
model.generate(**inputs, max_new_tokens=256, streamer=streamer)
```

## Datasets

```python
from datasets import load_dataset, Dataset

# Load from Hub
ds = load_dataset("tatsu-lab/alpaca")
ds = load_dataset("json", data_files="data.jsonl")
ds = load_dataset("csv", data_files="data.csv")

# Inspect
print(ds["train"].features)
print(ds["train"][0])

# Map / transform
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized = ds["train"].map(tokenize_fn, batched=True, num_proc=4, remove_columns=["text"])

# Filter
filtered = ds["train"].filter(lambda x: len(x["text"]) > 100)

# Train/test split
split = ds["train"].train_test_split(test_size=0.1, seed=42)

# Create from dict/pandas
ds = Dataset.from_dict({"text": texts, "label": labels})
ds = Dataset.from_pandas(df)
```

## Trainer API

### Basic Fine-Tuning

```python
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,    # effective batch = 8 × 4 = 32
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,                        # mixed precision (A100/H100)
    # fp16=True,                      # for older GPUs (V100/T4)
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="wandb",               # or "tensorboard", "none"
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,      # save memory at cost of speed
    torch_compile=True,               # PyTorch 2.0+ compilation
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./final_model")
```

### SFTTrainer (from TRL) for Instruction Tuning

```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./sft-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    bf16=True,
    gradient_checkpointing=True,
    max_seq_length=2048,
    packing=True,                # pack multiple samples per sequence
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    tokenizer=tokenizer,
)
trainer.train()
```

## PEFT / LoRA

Parameter-efficient fine-tuning — trains <1% of parameters:

```python
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# For quantized models
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                         # rank (8-64 typical)
    lora_alpha=32,                # scaling factor (usually 2×r)
    target_modules="all-linear",  # or ["q_proj", "v_proj", "k_proj", "o_proj"]
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 13M || all params: 8B || trainable%: 0.16%

# Train with Trainer as normal — only LoRA weights are updated

# Save adapter only (small — ~50MB for a 8B model)
model.save_pretrained("./lora-adapter")

# Load adapter later
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# Merge adapter into base (for deployment)
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
```

### QLoRA (Quantized LoRA)

```python
# Combine 4-bit quantization + LoRA = QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # 4-bit config from earlier
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
# Now train — uses ~6GB VRAM for a 7B model
```

## Text Generation Parameters

```python
output = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    do_sample=True,              # False = greedy
    num_beams=1,                 # >1 = beam search
    num_return_sequences=1,
    stop_strings=["<|end|>"],
    tokenizer=tokenizer,         # required for stop_strings
)
```

### Chat Templates

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Or tokenize directly
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
output = model.generate(inputs.to(model.device), max_new_tokens=256)
response = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
```

## Model Quantization

### GPTQ (Post-Training Quantization)

```python
from transformers import GPTQConfig

quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quantization_config, device_map="auto"
)
```

### AWQ

```python
from transformers import AwqConfig

quantization_config = AwqConfig(bits=4, fuse_max_seq_len=512)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-AWQ", quantization_config=quantization_config, device_map="auto"
)
```

## Evaluation

```python
import evaluate

# Load metrics
accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Compute
results = accuracy.compute(predictions=preds, references=labels)

# With Trainer — pass compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return accuracy.compute(predictions=preds, references=labels)

trainer = Trainer(model=model, args=args, compute_metrics=compute_metrics, ...)
```

## Model Parallelism

```python
# Automatic device mapping (splits across GPUs)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # auto-split across available GPUs
    torch_dtype=torch.bfloat16,
)

# Custom device map
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={
        "model.embed_tokens": 0,
        "model.layers.0-15": 0,
        "model.layers.16-31": 1,
        "model.norm": 1,
        "lm_head": 1,
    },
)

# Check device map
print(model.hf_device_map)
```

## Hub Operations

```python
from huggingface_hub import HfApi

# Push model to Hub
model.push_to_hub("myorg/my-model", private=True)
tokenizer.push_to_hub("myorg/my-model")

# Push with a model card
from huggingface_hub import ModelCard
card = ModelCard.from_template(
    card_data={"license": "mit", "tags": ["pytorch", "llama"]},
    model_id="myorg/my-model",
)
card.push_to_hub("myorg/my-model")

# Download specific files
api = HfApi()
api.hf_hub_download("meta-llama/Llama-3.1-8B-Instruct", "config.json")

# Snapshot download (full model)
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-3.1-8B-Instruct", local_dir="./model")
```

## Debugging

See `references/troubleshooting.md` for:
- Model loading failures (OOM, auth, architecture mismatches)
- Tokenizer issues (pad token, chat template, special tokens)
- Training issues (loss not decreasing, NaN, gradient explosion)
- PEFT/LoRA problems (target modules, merge errors)
- Common generation issues (repetition, empty output, wrong tokens)

## Reference

- [Transformers docs](https://huggingface.co/docs/transformers)
- [PEFT docs](https://huggingface.co/docs/peft)
- [Datasets docs](https://huggingface.co/docs/datasets)
- [TRL docs](https://huggingface.co/docs/trl)
- [Hub Python Library](https://huggingface.co/docs/huggingface_hub)
- `references/troubleshooting.md` — common errors and fixes
