import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Paths / config
# -----------------------------

MODEL_PATH = "/home/eric/base_models/embedding_model"
DATASET_PATH = "test1.jsonl"
OUTPUT_PATH = "experience_embeddings.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Load model
# -----------------------------

print(f"Loading embedding model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True,
)

model = model.to(DEVICE)
model.eval()

print(f"Using device: {DEVICE}")


# -----------------------------
# Embedding helper
# -----------------------------

@torch.no_grad()
def embed_text(text: str) -> torch.Tensor:
    """
    Convert one role-prefixed message into one semantic vector.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = model(**inputs)

    # Token-level hidden states:
    # [batch, sequence_length, hidden_dim]
    hidden = outputs.last_hidden_state

    # Attention-mask-aware mean pooling.
    mask = inputs["attention_mask"].unsqueeze(-1)

    masked_hidden = hidden * mask

    summed = masked_hidden.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)

    embedding = summed / counts

    # Normalize so vector magnitude does not dominate comparison.
    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1,
    )

    # Move back to CPU and remove batch dimension.
    return embedding.squeeze(0).float().cpu()


# -----------------------------
# Parse dataset
# -----------------------------

dataset_path = Path(DATASET_PATH)

experiences = []

with dataset_path.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        line = line.strip()

        if not line:
            continue

        try:
            example = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Skipping malformed JSON on line {line_number}: {exc}")
            continue

        messages = example.get("messages")

        if not messages:
            print(f"Skipping line {line_number}: no messages field")
            continue

        experiences.append(messages)


print(f"\nLoaded {len(experiences)} experiences")


# -----------------------------
# Embed experiences
# -----------------------------

embedded_experiences = []

for experience_index, messages in enumerate(experiences):

    message_vectors = []
    message_metadata = []

    for message_index, message in enumerate(messages):

        role = message.get("role", "unknown")
        content = message.get("content", "")

        # Preserve role as part of the semantic representation.
        text = f"{role.upper()}: {content}"

        vector = embed_text(text)

        message_vectors.append(vector)

        message_metadata.append(
            {
                "role": role,
                "content": content,
            }
        )

    # Shape:
    # [number_of_messages, embedding_dimension]
    experience_tensor = torch.stack(message_vectors)

    embedded_experiences.append(
        {
            "experience_id": experience_index,
            "messages": message_metadata,
            "embeddings": experience_tensor,
        }
    )

    print(
        f"Experience {experience_index:03d}: "
        f"{len(messages)} messages -> "
        f"{tuple(experience_tensor.shape)}"
    )


# -----------------------------
# Save
# -----------------------------

torch.save(
    {
        "model_path": MODEL_PATH,
        "num_experiences": len(embedded_experiences),
        "experiences": embedded_experiences,
    },
    OUTPUT_PATH,
)

print("\nDone.")
print(f"Saved embeddings to: {OUTPUT_PATH}")

if embedded_experiences:
    first = embedded_experiences[0]["embeddings"]

    print(f"Embedding dimension: {first.shape[-1]}")
    print(
        f"First experience shape: "
        f"{tuple(first.shape)}"
    )
