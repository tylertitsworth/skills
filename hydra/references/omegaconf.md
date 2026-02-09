# OmegaConf Reference

## Variable Interpolation

```yaml
model:
  name: llama-8b
  path: /models/${model.name}
  output: /checkpoints/${model.name}_${training.lr}
```

## Built-in Resolvers

| Resolver | Purpose | Example |
|---|---|---|
| `${oc.env:VAR}` | Environment variable | `${oc.env:HOME}` |
| `${oc.env:VAR,default}` | Env var with default | `${oc.env:LR,1e-4}` |
| `${oc.select:key,default}` | Config key with default | `${oc.select:model.quant,none}` |
| `${oc.decode:str}` | Decode YAML string | `${oc.decode:"[1,2,3]"}` |
| `${oc.deprecated:key}` | Deprecated key warning | |
| `${oc.create:dict}` | Create container from str | |

## Custom Resolvers

```python
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)

# In config:
# total_batch: ${mul:${batch_size},${gradient_accumulation}}
# dtype: ${if:${use_bf16},bfloat16,float32}
```

## MISSING Values

```python
from omegaconf import MISSING

@dataclass
class Config:
    model_name: str = MISSING     # Must be provided (no default)
    lr: float = MISSING
```

Accessing a MISSING value raises `MissingMandatoryValue`. Override via CLI or defaults.

## Key Operations

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
