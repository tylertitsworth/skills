---
name: hydra
description: >
  Configure ML applications with Meta's Hydra framework and OmegaConf. Use when:
  (1) Setting up Hydra config structure (config groups, defaults list, packages),
  (2) Writing override syntax (CLI overrides, append, delete, sweep),
  (3) Using structured configs (dataclasses) for type-safe configuration,
  (4) Object instantiation with hydra.utils.instantiate (_target_, _recursive_, _convert_),
  (5) Config composition (defaults list, _self_ ordering, config groups),
  (6) Using the Compose API for non-CLI usage (notebooks, tests, K8s jobs),
  (7) OmegaConf features (interpolation, resolvers, merge, MISSING),
  (8) Configuring sweepers and launchers for hyperparameter search,
  (9) Debugging config composition (--cfg, --info).
---

# Hydra

[Hydra](https://github.com/facebookresearch/hydra) (Meta) is a configuration framework for Python applications. Composes configs from multiple sources, supports CLI overrides, type-safe structured configs, and object instantiation. Built on [OmegaConf](https://github.com/omry/omegaconf). Version: **1.3.x**.

Used extensively in ML: verl, NeMo, fairseq, Detectron2, PyTorch Lightning CLI, and many training frameworks.

## Config Structure

### Directory Layout

```
conf/
├── config.yaml              # Primary config (has defaults list)
├── model/                   # Config group: model
│   ├── llama_7b.yaml
│   └── llama_70b.yaml
├── training/                # Config group: training
│   ├── sft.yaml
│   └── rlhf.yaml
├── data/                    # Config group: data
│   ├── gsm8k.yaml
│   └── openorca.yaml
└── experiment/              # Config group: experiment
    └── ablation_lr.yaml
```

### Primary Config

```yaml
# conf/config.yaml
defaults:
  - model: llama_7b
  - training: sft
  - data: gsm8k
  - _self_                   # This config's values override defaults

project_name: my-training
seed: 42
output_dir: /checkpoints/${project_name}
```

### Config Group Files

```yaml
# conf/model/llama_7b.yaml
name: meta-llama/Llama-3.1-8B-Instruct
hidden_size: 4096
num_layers: 32
dtype: bfloat16

# conf/training/sft.yaml
lr: 2e-5
epochs: 3
batch_size: 8
warmup_steps: 100
gradient_accumulation: 4
```

## Defaults List

The defaults list controls config composition — which config group options to load and in what order.

### Syntax

```yaml
defaults:
  - config_group: option          # Load conf/config_group/option.yaml
  - config_group: null            # Placeholder — must be overridden or ignored
  - optional config_group: option # No error if option doesn't exist
  - override config_group: option # Override a previously set default
  - /absolute/group: option       # Absolute path (from config root)
  - config_group@package: option  # Load into a specific package (location in output)
  - config_group@_global_: option # Load into root of output config
  - _self_                        # Position of this file's content in merge order
```

### _self_ and Composition Order

Configs are merged in defaults list order. Later entries override earlier ones.

```yaml
# _self_ at END (default) — this file's values WIN over defaults
defaults:
  - model: llama_7b
  - _self_                   # config.yaml values override model/llama_7b.yaml

model:
  dtype: float16             # overrides llama_7b's bfloat16
```

```yaml
# _self_ at START — defaults WIN over this file's values
defaults:
  - _self_
  - model: llama_7b          # model/llama_7b.yaml overrides config.yaml

model:
  dtype: float16             # overridden by llama_7b's bfloat16
```

### Config Groups as Overrides

```yaml
# conf/experiment/ablation_lr.yaml
# @package _global_
defaults:
  - override /training: sft
  - override /model: llama_7b

training:
  lr: 1e-5                   # experiment-specific override
```

## Override Grammar

### CLI Override Syntax

| Syntax | Action | Example |
|---|---|---|
| `key=value` | Override existing value | `training.lr=1e-4` |
| `+key=value` | Append new key | `+training.warmup_ratio=0.1` |
| `++key=value` | Append or override | `++training.lr=1e-4` |
| `~key` | Delete key | `~training.warmup_steps` |
| `~key=value` | Delete key (with value check) | `~training.warmup_steps=100` |
| `group=option` | Select config group option | `model=llama_70b` |
| `+group=option` | Append to defaults list | `+experiment=ablation_lr` |
| `~group` | Remove from defaults list | `~data` |

### Value Types

| Type | Examples |
|---|---|
| String | `name=llama`, `name="hello world"` |
| Int | `epochs=3`, `seed=42` |
| Float | `lr=1e-4`, `dropout=0.1` |
| Bool | `bf16=true`, `debug=false` |
| Null | `scheduler=null` |
| List | `gpus=[0,1,2,3]` |
| Dict | `optim={type:adam,lr:1e-4}` |
| Interpolation | `'output=${model.name}_${training.lr}'` |
| Choice sweep | `lr=1e-3,1e-4,1e-5` |

### Nested Overrides

```bash
# Dot notation for nested keys
training.optimizer.lr=1e-4
training.optimizer.weight_decay=0.01

# Dict override (replaces entire dict)
training.optimizer={type:adam,lr:1e-4,weight_decay:0.01}
```

## Structured Configs (Dataclasses)

Type-safe configs using Python dataclasses with runtime validation:

```python
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from omegaconf import MISSING

class ModelType(Enum):
    LLAMA = "llama"
    MISTRAL = "mistral"

@dataclass
class ModelConfig:
    name: str = MISSING                          # required field (no default)
    model_type: ModelType = ModelType.LLAMA
    hidden_size: int = 4096
    num_layers: int = 32
    dtype: str = "bfloat16"
    quantization: Optional[str] = None

@dataclass
class TrainingConfig:
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 8
    warmup_steps: int = 100
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    bf16: bool = True

@dataclass
class DataConfig:
    dataset: str = MISSING
    max_seq_length: int = 4096
    num_workers: int = 4
    split: str = "train"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    project_name: str = "my-training"
    seed: int = 42
    output_dir: str = "/checkpoints/${project_name}"
```

### Register with ConfigStore

```python
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="llama_7b", node=ModelConfig(
    name="meta-llama/Llama-3.1-8B-Instruct",
    hidden_size=4096, num_layers=32,
))
cs.store(group="model", name="llama_70b", node=ModelConfig(
    name="meta-llama/Llama-3.1-70B-Instruct",
    hidden_size=8192, num_layers=80,
))
```

### Structured Config as Schema (Validation)

Use a structured config to validate YAML files:

```python
# Register as schema for config group
cs.store(group="model", name="model_schema", node=ModelConfig)
```

```yaml
# conf/model/llama_7b.yaml
defaults:
  - model_schema      # validates against ModelConfig dataclass

name: meta-llama/Llama-3.1-8B-Instruct
hidden_size: 4096
num_layers: 32
```

Any invalid keys or wrong types raise errors at composition time.

## Object Instantiation

`hydra.utils.instantiate()` creates objects from config:

### _target_ Fields

| Field | Purpose | Default |
|---|---|---|
| `_target_` | Fully qualified class/function name | required |
| `_args_` | Positional arguments | `[]` |
| `_recursive_` | Instantiate nested _target_ configs | `True` |
| `_convert_` | Conversion strategy for OmegaConf containers | `"none"` |
| `_partial_` | Return `functools.partial` instead of calling | `False` |

### _convert_ Strategies

| Value | Behavior |
|---|---|
| `"none"` | Pass DictConfig/ListConfig as-is (default) |
| `"partial"` | Convert to dict/list, except Structured Configs |
| `"object"` | Convert to dict/list, Structured Configs → dataclass instances |
| `"all"` | Convert everything to plain dicts, lists, primitives |

### Usage

```yaml
# conf/config.yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: 1000
  eta_min: 1e-6

model:
  _target_: transformers.AutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
  torch_dtype: bfloat16
```

```python
import hydra
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg):
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
```

### Recursive Instantiation

```yaml
trainer:
  _target_: my_app.Trainer
  model:
    _target_: transformers.AutoModelForCausalLM.from_pretrained
    pretrained_model_name_or_path: ${model.name}
  optimizer:
    _target_: torch.optim.AdamW
    lr: ${training.lr}
```

All nested `_target_` configs are instantiated recursively by default.

### Partial Instantiation

```yaml
optimizer:
  _target_: torch.optim.AdamW
  _partial_: true              # returns functools.partial(AdamW, lr=1e-4, ...)
  lr: 1e-4
  weight_decay: 0.01
```

```python
# Later, when you have model parameters:
opt_fn = instantiate(cfg.optimizer)  # partial(AdamW, lr=1e-4, ...)
optimizer = opt_fn(params=model.parameters())
```

## OmegaConf Features

### Variable Interpolation

```yaml
model:
  name: llama-8b
  path: /models/${model.name}
  output: /checkpoints/${model.name}_${training.lr}
```

### Built-in Resolvers

| Resolver | Purpose | Example |
|---|---|---|
| `${oc.env:VAR}` | Environment variable | `${oc.env:HOME}` |
| `${oc.env:VAR,default}` | Env var with default | `${oc.env:LR,1e-4}` |
| `${oc.select:key,default}` | Config key with default | `${oc.select:model.quant,none}` |
| `${oc.decode:str}` | Decode YAML string | `${oc.decode:"[1,2,3]"}` |
| `${oc.deprecated:key}` | Deprecated key warning | |
| `${oc.create:dict}` | Create container from str | |

### Custom Resolvers

```python
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)

# In config:
# total_batch: ${mul:${batch_size},${gradient_accumulation}}
# dtype: ${if:${use_bf16},bfloat16,float32}
```

### MISSING Values

```python
from omegaconf import MISSING

@dataclass
class Config:
    model_name: str = MISSING     # Must be provided (no default)
    lr: float = MISSING
```

Accessing a MISSING value raises `MissingMandatoryValue`. Override via CLI or defaults.

### Key OmegaConf Operations

```python
from omegaconf import OmegaConf, DictConfig

# Convert to plain dict
plain = OmegaConf.to_container(cfg, resolve=True)

# Convert to YAML string
yaml_str = OmegaConf.to_yaml(cfg)

# Merge configs
merged = OmegaConf.merge(base_cfg, override_cfg)

# Check if value is missing
OmegaConf.is_missing(cfg, "model_name")

# Make config read-only
OmegaConf.set_readonly(cfg, True)

# Set struct flag (no new keys allowed)
OmegaConf.set_struct(cfg, True)
```

## Compose API (Non-CLI Usage)

For notebooks, tests, K8s jobs, and programmatic usage:

```python
from hydra import compose, initialize, initialize_config_dir

# Option 1: Relative config path
with initialize(version_base=None, config_path="conf"):
    cfg = compose(
        config_name="config",
        overrides=["model=llama_70b", "training.lr=1e-5", "+experiment=ablation_lr"],
    )

# Option 2: Absolute config path (for K8s jobs with mounted configs)
with initialize_config_dir(version_base=None, config_dir="/etc/config"):
    cfg = compose(config_name="config", overrides=["training.epochs=5"])

# Option 3: Global initialization (for scripts)
from hydra import initialize_config_module
initialize_config_module(version_base=None, config_module="my_package.conf")
cfg = compose(config_name="config")
```

## @hydra.main Decorator

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    print(cfg.model.name)
    print(cfg.training.lr)

if __name__ == "__main__":
    train()
```

### version_base

| Value | Behavior |
|---|---|
| `None` | Use latest Hydra defaults |
| `"1.1"` | Hydra 1.1 compatibility defaults |
| `"1.2"` | Hydra 1.2 compatibility defaults |
| `"1.3"` | Hydra 1.3 defaults |

Always set `version_base=None` for new projects.

## Sweepers (Hyperparameter Search)

### Basic Sweep

```bash
# Sweep over lr and batch_size combinations
python train.py --multirun training.lr=1e-3,1e-4,1e-5 training.batch_size=8,16,32
```

### Optuna Sweeper

```yaml
# conf/config.yaml
defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
    direction: minimize
    n_trials: 50
    params:
      training.lr:
        type: float
        low: 1e-6
        high: 1e-3
        log: true
      training.batch_size:
        type: categorical
        choices: [8, 16, 32]
```

### Launchers

| Launcher | Purpose |
|---|---|
| `basic` (default) | Sequential local execution |
| `joblib` | Parallel local execution |
| `ray` | Distributed via Ray cluster |
| `submitit_slurm` | Submit to SLURM scheduler |

## Debugging

### Inspection Flags

| Flag | Shows |
|---|---|
| `--cfg job` | Final composed config (job portion) |
| `--cfg hydra` | Hydra's own config |
| `--cfg all` | Both job and hydra config |
| `--info defaults-tree` | Defaults tree (composition structure) |
| `--info defaults` | Final defaults list (flat, ordered) |
| `--info config` | Config search path and sources |
| `--help` | App help with config structure |
| `--hydra-help` | Hydra's help |

### Common Errors

| Error | Cause | Fix |
|---|---|---|
| `MissingMandatoryValue` | MISSING field not set | Provide value via override or defaults |
| `ConfigAttributeError` | Accessing non-existent key (struct mode) | Add key with `+key=value` or disable struct |
| `ConfigCompositionException` | Invalid defaults list | Check group names, option names, `_self_` position |
| `Could not load config` | Wrong config_path or missing file | Verify paths relative to calling module |
| `Key not found` in interpolation | Referenced key doesn't exist | Check interpolation targets exist in composed config |

## Hydra Output Directory

Hydra creates an output directory per run:

```
outputs/
└── 2024-01-15/
    └── 14-30-00/
        ├── .hydra/
        │   ├── config.yaml       # resolved config
        │   ├── overrides.yaml    # CLI overrides used
        │   └── hydra.yaml        # hydra config
        └── train.log
```

Override with: `hydra.run.dir=/custom/path` or `hydra.run.dir=.` to disable.

For `--multirun`: `hydra.sweep.dir=/custom/sweep/path`.

## Reference

- [Hydra docs](https://hydra.cc/docs/intro/)
- [Hydra GitHub](https://github.com/facebookresearch/hydra)
- [OmegaConf docs](https://omegaconf.readthedocs.io/)
- [Override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [Instantiate API](https://hydra.cc/docs/advanced/instantiate_objects/overview/)
- [Defaults List](https://hydra.cc/docs/advanced/defaults_list/)
- [Compose API](https://hydra.cc/docs/advanced/compose_api/)
