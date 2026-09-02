import torch
import torch.nn.functional as F


DATA_PATH = "experience_vectors.pt"

data = torch.load(
    DATA_PATH,
    weights_only=False,
)

experiences = data["experiences"]

print(f"Loaded {len(experiences)} experience vectors")


# ------------------------------------------------
# Stack all e vectors
# ------------------------------------------------

E = torch.stack([
    exp["e"].float()
    for exp in experiences
])

print(f"E shape: {tuple(E.shape)}")


# ------------------------------------------------
# Full cosine similarity matrix
# ------------------------------------------------

similarity = E @ E.T


# ------------------------------------------------
# Remove self-similarity
# ------------------------------------------------

num_experiences = E.shape[0]

mask = ~torch.eye(
    num_experiences,
    dtype=torch.bool,
)

off_diagonal = similarity[mask]


print()
print("Similarity statistics between DIFFERENT experiences:")
print(f"Min:    {off_diagonal.min().item():.4f}")
print(f"Max:    {off_diagonal.max().item():.4f}")
print(f"Mean:   {off_diagonal.mean().item():.4f}")
print(f"Std:    {off_diagonal.std().item():.4f}")


# ------------------------------------------------
# Strongest matching experience pairs
# ------------------------------------------------

pairs = []

for i in range(num_experiences):
    for j in range(i + 1, num_experiences):

        score = similarity[i, j].item()

        pairs.append(
            (
                score,
                i,
                j,
            )
        )

pairs.sort(reverse=True)


print()
print("Most similar experience pairs:")
print()

for score, i, j in pairs[:10]:

    user_i = experiences[i]["messages"][0]["content"]
    user_j = experiences[j]["messages"][0]["content"]

    preview_i = user_i.replace("\n", " ")[:120]
    preview_j = user_j.replace("\n", " ")[:120]

    print(f"{score:.4f}")
    print(f"  Experience {i:03d}: {preview_i}")
    print(f"  Experience {j:03d}: {preview_j}")
    print()


# ------------------------------------------------
# Least similar experience pairs
# ------------------------------------------------

pairs.sort()


print()
print("Least similar experience pairs:")
print()

for score, i, j in pairs[:10]:

    user_i = experiences[i]["messages"][0]["content"]
    user_j = experiences[j]["messages"][0]["content"]

    preview_i = user_i.replace("\n", " ")[:120]
    preview_j = user_j.replace("\n", " ")[:120]

    print(f"{score:.4f}")
    print(f"  Experience {i:03d}: {preview_i}")
    print(f"  Experience {j:03d}: {preview_j}")
    print()
