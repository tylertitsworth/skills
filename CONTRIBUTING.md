# Contributing

## Adding a New Skill

1. **Create a branch** from `main`: `feat/<skill-name>`
2. **Create the skill directory** with at minimum a `SKILL.md`
3. **Follow the structure**:
   ```
   skill-name/
   ├── SKILL.md              # Required
   ├── references/           # Optional — detailed documentation
   ├── scripts/              # Optional — executable code
   └── assets/               # Optional — templates, output files
   ```
4. **Open a PR** targeting `main` with title: `feat: <skill-name> — <one-line purpose>`

## SKILL.md Requirements

### Frontmatter (Required)

```yaml
---
name: skill-name
description: >
  What the skill does and WHEN to use it.
  This is the trigger mechanism — be specific about matching conditions.
---
```

### Body Guidelines

- **Under 500 lines.** Move detailed content to `references/` files.
- **GPU-first examples.** Use `nvidia.com/gpu` resource requests, bf16/fp16 defaults.
- **Practical over theoretical.** Show commands and configs that actually work.
- **Pin versions.** Specify exact API versions and tool releases.
- **Cross-reference** other skills in this repo instead of duplicating content.

## What NOT to Include

- `README.md` inside skill directories (SKILL.md serves this purpose)
- `CHANGELOG.md` per skill (git history is the changelog)
- Theoretical explanations without actionable examples
- Content the model already knows well (basic Python, general CLI usage)

## Commit Messages

Use imperative mood with the skill name:

```
feat: add kueue SKILL.md with GPU quota examples
feat: add kueue references/troubleshooting.md
fix: correct vllm tensor-parallel config example
```

## PR Description

Include:
- **Summary**: What the skill does and why
- **Design decisions**: Key choices and trade-offs
- **File structure**: Tree view of files created
- **Testing notes**: Example prompts to verify the skill works

## Code Review

All PRs are reviewed before merge. Common feedback areas:
- Frontmatter `description` not specific enough for triggering
- SKILL.md too long (move content to references)
- Missing version pins
- Examples that won't work in practice
