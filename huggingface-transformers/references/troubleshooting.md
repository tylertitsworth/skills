# Hugging Face Transformers Troubleshooting

## Model Loading Failures

### CUDA OOM during loading

```python
# Use device_map="auto" to shard across GPUs / offload to CPU
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Quantize to reduce memory
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

# Load in fp16/bf16 instead of fp32
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
```

### "401 Client Error: Repository is gated"

Model requires access approval:
1. Go to the model page on huggingface.co and accept the license
2. Login: `huggingface-cli login` with a token from https://huggingface.co/settings/tokens
3. Or set: `export HUGGING_FACE_HUB_TOKEN="hf_..."`

### "OSError: Can't load config for 'model-name'"

- Model name is wrong or doesn't exist
- Private model without auth
- Network issue — try `HF_HUB_OFFLINE=1` with a cached model

### "ValueError: Unrecognized configuration class"

Model uses custom code:
```python
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
```

### Architecture mismatch

Loading a causal LM with `AutoModelForSequenceClassification` (or vice versa):
```
Some weights of LlamaForSequenceClassification were not initialized from the model checkpoint
```
Use the correct `AutoModelFor*` class for your task.

## Tokenizer Issues

### "Token indices sequence length is longer than the specified maximum"

```python
# Always set truncation
inputs = tokenizer(text, truncation=True, max_length=model.config.max_position_embeddings)
```

### No pad token

Many models (Llama, GPT-2) lack a pad token:
```python
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
```

### Chat template not found

```
Jinja2 TemplateError: 'chat_template' not found
```

Model doesn't have a built-in chat template:
```python
# Use a generic one
tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{{ 'assistant: ' }}"

# Or pass a template file
tokenizer.apply_chat_template(messages, chat_template=open("template.jinja").read())
```

### Special tokens in generation output

```python
# Decode without special tokens
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Issues

### Loss not decreasing

1. **Learning rate too high or too low**: Start with `2e-5` for fine-tuning, `1e-4` for LoRA
2. **Data issue**: Inspect a few samples from the DataLoader manually
3. **Wrong data collator**: Ensure labels are set correctly
4. **Frozen parameters**: Check `model.print_trainable_parameters()` (PEFT)
5. **Gradient accumulation math**: Effective LR = LR × gradient_accumulation_steps (sometimes)

### NaN loss

```python
# Switch to fp32 or bf16 (fp16 can overflow)
training_args = TrainingArguments(bf16=True)  # instead of fp16=True

# Enable gradient clipping
training_args = TrainingArguments(max_grad_norm=1.0)

# Check for bad data
for batch in dataloader:
    assert not torch.isnan(batch["input_ids"].float()).any()
```

### OOM during training

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Enable `gradient_checkpointing=True`
4. Use PEFT/LoRA instead of full fine-tuning
5. Use QLoRA (4-bit + LoRA)
6. Enable `bf16=True` or `fp16=True`

### Trainer not saving best model

```python
training_args = TrainingArguments(
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",        # must match eval_strategy
    save_steps=100,               # must match eval_steps
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
```

## PEFT / LoRA Issues

### "ValueError: Target modules not found"

The `target_modules` don't match your model's layer names:
```python
# Inspect model layers
for name, _ in model.named_modules():
    print(name)

# Use "all-linear" for automatic detection
lora_config = LoraConfig(target_modules="all-linear", ...)
```

### Merge errors

```
RuntimeError: Cannot merge LORA layers when base model is on different device
```

```python
# Ensure model is on one device before merge
model = model.to("cpu")
merged = model.merge_and_unload()
```

### LoRA adapter not loading

```python
# Ensure base model matches the one used for training
base = AutoModelForCausalLM.from_pretrained("same-model-used-for-training")
model = PeftModel.from_pretrained(base, "./lora-adapter")

# Check adapter_config.json matches the base model
```

## Generation Issues

### Repetitive output

```python
model.generate(
    **inputs,
    repetition_penalty=1.2,      # penalize repeated tokens
    no_repeat_ngram_size=3,      # ban repeating 3-grams
    temperature=0.8,             # add randomness
)
```

### Empty or very short output

```python
# Ensure max_new_tokens is set (not max_length, which includes prompt)
model.generate(**inputs, max_new_tokens=256)

# Check that pad_token_id is set
model.generate(**inputs, pad_token_id=tokenizer.pad_token_id)

# Check that eos_token_id isn't being generated immediately
# (may happen with wrong chat template)
```

### Wrong tokens / gibberish

- Model and tokenizer mismatch — load both from the same model ID
- Wrong `torch_dtype` — use `torch_dtype="auto"` or the model's native dtype
- Corrupted checkpoint — re-download: `rm -rf ~/.cache/huggingface/hub/models--<model>/`

## Dataset Issues

### "pyarrow.lib.ArrowInvalid: Column ... expected"

Schema mismatch — ensure all rows have the same columns:
```python
# Force consistent features
ds = ds.cast_column("label", Value("int64"))
```

### Slow `.map()` processing

```python
# Use batched=True for vectorized transforms
ds = ds.map(fn, batched=True, batch_size=1000, num_proc=8)

# Save processed dataset to disk to avoid reprocessing
ds.save_to_disk("./processed_data")
ds = load_from_disk("./processed_data")
```

### Dataset too large for memory

```python
# Stream instead of loading all at once
ds = load_dataset("large-dataset", streaming=True)
for sample in ds:
    process(sample)
```
