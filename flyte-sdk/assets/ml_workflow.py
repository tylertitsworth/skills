"""Example Flyte ML workflow: data prep → training → evaluation.

Demonstrates:
  - @task with ImageSpec for containerized execution
  - Typed inputs/outputs with Annotated
  - Caching and retries
  - GPU resource requests
  - Workflow composition

Register and run:
    pyflyte register ml_workflow.py
    pyflyte run --remote ml_workflow.py training_pipeline --model_name bert-base
"""

from flytekit import task, workflow, ImageSpec, Resources
from flytekit.types.file import FlyteFile
from typing import Annotated

# Container image spec — Flyte builds this automatically
training_image = ImageSpec(
    name="ml-training",
    python_version="3.11",
    packages=[
        "torch>=2.4",
        "transformers>=4.45",
        "datasets>=2.0",
        "scikit-learn",
    ],
    cuda="12.4",
    registry="ghcr.io/myorg",
)


@task(
    container_image=training_image,
    cache=True,
    cache_version="1.0",
    retries=2,
    requests=Resources(cpu="4", mem="16Gi"),
)
def prepare_data(dataset_name: str, split: str = "train") -> FlyteFile:
    """Download and preprocess a dataset."""
    from datasets import load_dataset
    import json

    ds = load_dataset(dataset_name, split=split)

    output_path = "/tmp/processed_data.jsonl"
    with open(output_path, "w") as f:
        for item in ds:
            f.write(json.dumps(item) + "\n")

    return FlyteFile(output_path)


@task(
    container_image=training_image,
    retries=1,
    requests=Resources(cpu="8", mem="32Gi", gpu="1"),
    limits=Resources(gpu="1"),
)
def train_model(
    data: FlyteFile,
    model_name: str,
    learning_rate: float = 2e-5,
    epochs: int = 3,
) -> FlyteFile:
    """Fine-tune a model on the prepared data."""
    # Placeholder: replace with actual training logic
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training {model_name} on {device}")
    print(f"  LR: {learning_rate}, Epochs: {epochs}")
    print(f"  Data: {data.path}")

    # Save model checkpoint
    output_path = "/tmp/model_checkpoint.pt"
    torch.save({"model_name": model_name, "epochs": epochs}, output_path)
    return FlyteFile(output_path)


@task(
    container_image=training_image,
    cache=True,
    cache_version="1.0",
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(gpu="1"),
)
def evaluate_model(
    model_checkpoint: FlyteFile,
    test_data: FlyteFile,
) -> Annotated[float, "accuracy"]:
    """Evaluate the trained model and return accuracy."""
    import torch

    ckpt = torch.load(model_checkpoint.path, weights_only=True)
    print(f"Evaluating model: {ckpt['model_name']}")

    # Placeholder: replace with actual evaluation
    accuracy = 0.85
    return accuracy


@workflow
def training_pipeline(
    model_name: str = "bert-base-uncased",
    dataset_name: str = "glue",
    learning_rate: float = 2e-5,
    epochs: int = 3,
) -> float:
    """End-to-end ML training pipeline."""
    train_data = prepare_data(dataset_name=dataset_name, split="train")
    test_data = prepare_data(dataset_name=dataset_name, split="test")
    checkpoint = train_model(
        data=train_data,
        model_name=model_name,
        learning_rate=learning_rate,
        epochs=epochs,
    )
    accuracy = evaluate_model(
        model_checkpoint=checkpoint,
        test_data=test_data,
    )
    return accuracy
