# Skills

OpenClaw skills for ML/AI infrastructure on Kubernetes. Built for agents helping with cloud-native ML workloads — model training, serving, orchestration, and tooling.

## What Are Skills?

Skills are modular instruction packages that teach AI agents how to work with specific tools and frameworks. Each skill is a folder containing a `SKILL.md` with YAML frontmatter (name + description for triggering) and markdown instructions, plus optional `scripts/`, `references/`, and `assets/` directories.

Skills are loaded dynamically when an agent detects a matching task. The `description` field in frontmatter is the trigger mechanism — write it to match the user requests you want to handle.

## Available Skills

| Skill | Description |
|-------|-------------|
| [kueue](kueue/) | Kubernetes-native job queueing with GPU quota management |
| [kuberay](kuberay/) | Deploy and manage Ray clusters on Kubernetes |
| [ray-core](ray-core/) | Distributed computing primitives with Ray |
| [ray-data](ray-data/) | Scalable data loading and preprocessing for ML |
| [ray-serve](ray-serve/) | Model serving and inference with Ray Serve |
| [ray-train](ray-train/) | Distributed training with Ray Train |
| [flyte-deployment](flyte-deployment/) | Deploy and operate Flyte on Kubernetes |
| [flyte-sdk](flyte-sdk/) | Author ML workflows with Flytekit |
| [wandb](wandb/) | Experiment tracking with Weights & Biases |
| [operator-sdk](operator-sdk/) | Build Kubernetes operators for ML workloads |
| [pytorch](pytorch/) | PyTorch model development and GPU training |
| [vllm](vllm/) | High-throughput LLM serving with vLLM |
| [openai-api](openai-api/) | OpenAI-compatible API integration |
| [huggingface-transformers](huggingface-transformers/) | Transformers library for fine-tuning and inference |
| [fsdp](fsdp/) | Fully Sharded Data Parallel distributed training |
| [verl](verl/) | RLHF/GRPO training with vERL |
| [megatron-lm](megatron-lm/) | Large-scale model training with Megatron-LM |
| [triton-inference-server](triton-inference-server/) | Multi-framework model serving with Triton |
| [llm-evaluation](llm-evaluation/) | LLM evaluation and benchmarking |

## Installation

### OpenClaw CLI

```bash
# Install a single skill
openclaw install github:tylertitsworth/skills/<skill-name>

# Install from a local clone
openclaw install /path/to/skills/<skill-name>
```

### Manual

Copy the skill folder into your OpenClaw skills directory:

```bash
cp -r <skill-name> ~/.openclaw/skills/
```

## Skill Structure

```
skill-name/
├── SKILL.md              # Required — frontmatter + instructions
├── references/           # Optional — detailed docs loaded on demand
├── scripts/              # Optional — executable code
└── assets/               # Optional — templates, configs, etc.
```

### Conventions

- **SKILL.md** stays under 500 lines. Use `references/` for deep dives.
- **Frontmatter** `description` field determines when the skill triggers — be specific about use cases.
- **Progressive disclosure**: core instructions in SKILL.md, detailed topics in reference files.
- **Cross-references**: skills reference each other (e.g., `vllm` → `openai-api`) instead of duplicating content.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding or modifying skills.

## License

[MIT](LICENSE)
