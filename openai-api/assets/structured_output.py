"""OpenAI API structured output examples.

Demonstrates:
  - Pydantic model → JSON schema for response_format
  - Responses API with reasoning (GPT-5)
  - Streaming with structured output
  - Multi-turn with tool calling

Usage:
    export OPENAI_API_KEY=sk-...
    python structured_output.py
"""

from enum import Enum
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field


# --- Structured Output with Pydantic ---

class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class K8sIssue(BaseModel):
    """Kubernetes troubleshooting diagnosis."""
    resource_type: str = Field(description="K8s resource type (Pod, Deployment, etc.)")
    resource_name: str
    namespace: str
    issue: str = Field(description="One-line issue summary")
    severity: Severity
    root_cause: str = Field(description="Root cause analysis")
    fix: str = Field(description="Recommended fix")
    commands: list[str] = Field(description="kubectl commands to apply the fix")


def structured_output_example():
    """Use response_format with json_schema for structured output."""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a Kubernetes troubleshooting expert."},
            {"role": "user", "content": "Pod 'training-worker-0' in namespace 'ml' is stuck in ImagePullBackOff for image nvcr.io/nvidia/pytorch:24.12-py3"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "k8s_diagnosis",
                "schema": K8sIssue.model_json_schema(),
            },
        },
    )

    diagnosis = K8sIssue.model_validate_json(response.choices[0].message.content)
    print(f"Issue: {diagnosis.issue}")
    print(f"Severity: {diagnosis.severity.value}")
    print(f"Root cause: {diagnosis.root_cause}")
    print(f"Fix: {diagnosis.fix}")
    for cmd in diagnosis.commands:
        print(f"  $ {cmd}")


# --- Responses API with Reasoning (GPT-5) ---

class ArchitectureRecommendation(BaseModel):
    """ML system architecture recommendation."""
    pattern: str = Field(description="Architecture pattern name")
    components: list[str]
    gpu_count: int
    estimated_cost_per_hour: Optional[float] = None
    reasoning: str = Field(description="Why this architecture fits the requirements")
    tradeoffs: list[str]


def reasoning_example():
    """Use Responses API with GPT-5 reasoning for complex analysis."""
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "medium"},
        input=[
            {"role": "user", "content": "Design an inference architecture for serving a 70B parameter model with <200ms p99 latency at 100 req/s. Budget: 8 H100 GPUs."},
        ],
        text={"format": {
            "type": "json_schema",
            "name": "architecture",
            "schema": ArchitectureRecommendation.model_json_schema(),
        }},
    )

    rec = ArchitectureRecommendation.model_validate_json(response.output_text)
    print(f"Pattern: {rec.pattern}")
    print(f"Components: {', '.join(rec.components)}")
    print(f"GPUs: {rec.gpu_count}")
    print(f"Reasoning: {rec.reasoning}")


if __name__ == "__main__":
    print("=== Structured Output (Chat Completions) ===\n")
    structured_output_example()
    print("\n=== Reasoning (Responses API) ===\n")
    reasoning_example()
