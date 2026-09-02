import math

import torch
import torch.nn.functional as F


# ------------------------------------------------
# Config
# ------------------------------------------------

INPUT_PATH = "experience_embeddings.pt"
OUTPUT_PATH = "experience_vectors.pt"

# Your longest current experience is 32 messages.
# We map every experience onto the same relative start -> end range.
POSITION_RANGE = 31.0

# Position should affect an event without overwhelming its semantics.
POSITION_STRENGTH = 0.25

# Because our input embeddings are unit-normalized, ordinary
# / sqrt(d) attention would become nearly flat.
# This temperature sharpens cosine-attention relationships.
ATTENTION_TEMPERATURE = 0.10


# ------------------------------------------------
# Relative positional encoding
# ------------------------------------------------

def relative_positional_encoding(
    sequence_length: int,
    embedding_dim: int,
) -> torch.Tensor:
    """
    Build sinusoidal positional encodings where:

        first event -> position 0
        final event -> position POSITION_RANGE

    This means experiences of different lengths still have
    comparable START and END positions.
    """

    if sequence_length == 1:
        relative_positions = torch.zeros(1)
    else:
        relative_positions = torch.linspace(
            0.0,
            POSITION_RANGE,
            steps=sequence_length,
        )

    pe = torch.zeros(sequence_length, embedding_dim)

    # Frequencies for the even dimensions.
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / embedding_dim)
    )

    angles = relative_positions.unsqueeze(1) * div_term.unsqueeze(0)

    pe[:, 0::2] = torch.sin(angles)

    # Handle odd embedding dimensions safely.
    if embedding_dim > 1:
        pe[:, 1::2] = torch.cos(
            angles[:, :pe[:, 1::2].shape[1]]
        )

    # Important:
    # Our semantic embeddings have norm ~= 1.
    # Raw sinusoidal PE has a much larger norm.
    #
    # Normalize PE so POSITION_STRENGTH has an understandable scale.
    pe = F.normalize(pe, p=2, dim=-1)

    return pe


# ------------------------------------------------
# Masked attention
# ------------------------------------------------

def masked_attention(
    x: torch.Tensor,
    direction: str,
):
    """
    Parameter-free cosine attention.

    x shape:
        [T, D]

    forward:
        event i can attend to events 0..i

    backward:
        event i can attend to events i..T-1
    """

    sequence_length = x.shape[0]

    # x is normalized, so x @ x.T is cosine similarity.
    scores = x @ x.T

    scores = scores / ATTENTION_TEMPERATURE

    if direction == "forward":

        # Lower triangular:
        #
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
            )
        )

    elif direction == "backward":

        # Upper triangular:
        #
        # 1 1 1 1
        # 0 1 1 1
        # 0 0 1 1
        # 0 0 0 1

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
            )
        )

    else:
        raise ValueError(
            "direction must be 'forward' or 'backward'"
        )

    scores = scores.masked_fill(
        ~mask,
        float("-inf"),
    )

    weights = F.softmax(scores, dim=-1)

    contextualized = weights @ x

    return contextualized, weights


# ------------------------------------------------
# Complete experience encoder
# ------------------------------------------------

def encode_experience(
    embeddings: torch.Tensor,
):
    """
    embeddings:
        [T, 1024]

    returns:
        e:
            [1024]

        plus intermediate representations for inspection.
    """

    sequence_length, embedding_dim = embeddings.shape

    # --------------------------------------------
    # 1. Add relative position
    # --------------------------------------------

    pe = relative_positional_encoding(
        sequence_length,
        embedding_dim,
    )

    positioned = (
        embeddings
        + POSITION_STRENGTH * pe
    )

    # Return vectors to unit length.
    positioned = F.normalize(
        positioned,
        p=2,
        dim=-1,
    )

    # --------------------------------------------
    # 2. Forward attention
    # --------------------------------------------

    forward_context, forward_weights = masked_attention(
        positioned,
        direction="forward",
    )

    # --------------------------------------------
    # 3. Backward attention
    # --------------------------------------------

    backward_context, backward_weights = masked_attention(
        positioned,
        direction="backward",
    )

    # --------------------------------------------
    # 4. Combine both directions
    # --------------------------------------------

    complete_context = (
        forward_context
        + backward_context
    ) / 2.0

    # --------------------------------------------
    # 5. Pool whole completed trajectory
    # --------------------------------------------
    #
    # Every row now contains information gathered
    # from one direction and the other direction.
    #
    # Mean pooling here is NOT averaging the original
    # raw message embeddings.
    #
    # We are averaging contextualized event states.

    e = complete_context.mean(dim=0)

    e = F.normalize(
        e,
        p=2,
        dim=0,
    )

    return {
        "e": e,
        "positioned": positioned,
        "forward_context": forward_context,
        "backward_context": backward_context,
        "complete_context": complete_context,
        "forward_weights": forward_weights,
        "backward_weights": backward_weights,
    }


# ------------------------------------------------
# Load embedded experiences
# ------------------------------------------------

data = torch.load(
    INPUT_PATH,
    weights_only=False,
)

experiences = data["experiences"]

print(
    f"Loaded {len(experiences)} embedded experiences"
)


# ------------------------------------------------
# Build e for every experience
# ------------------------------------------------

encoded_experiences = []

for experience in experiences:

    experience_id = experience["experience_id"]

    embeddings = experience["embeddings"].float()

    result = encode_experience(
        embeddings
    )

    e = result["e"]

    encoded_experiences.append(
        {
            "experience_id": experience_id,
            "messages": experience["messages"],
            "e": e,

            # Keep these for research/debugging.
            "forward_weights":
                result["forward_weights"],

            "backward_weights":
                result["backward_weights"],
        }
    )

    print(
        f"Experience {experience_id:03d}: "
        f"{tuple(embeddings.shape)} "
        f"-> e {tuple(e.shape)} "
        f"| norm={e.norm().item():.4f}"
    )


# ------------------------------------------------
# Save
# ------------------------------------------------

torch.save(
    {
        "source": INPUT_PATH,
        "num_experiences":
            len(encoded_experiences),

        "embedding_dimension":
            encoded_experiences[0]["e"].shape[0],

        "position_range":
            POSITION_RANGE,

        "position_strength":
            POSITION_STRENGTH,

        "attention_temperature":
            ATTENTION_TEMPERATURE,

        "experiences":
            encoded_experiences,
    },
    OUTPUT_PATH,
)


print()
print("Done.")
print(
    f"Saved experience vectors to: "
    f"{OUTPUT_PATH}"
)

print(
    f"Experience vector dimension: "
    f"{encoded_experiences[0]['e'].shape[0]}"
)


# ------------------------------------------------
# Inspect experience 0 attention
# ------------------------------------------------

first = encoded_experiences[0]

print()
print("Experience 0 forward attention:")
print(first["forward_weights"])

print()
print("Experience 0 backward attention:")
print(first["backward_weights"])
