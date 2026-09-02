import torch
import torch.nn.functional as F


DATA_PATH = "experience_embeddings.pt"

data = torch.load(DATA_PATH, weights_only=False)

experiences = data["experiences"]

print(f"Loaded {len(experiences)} experiences")


# ------------------------------------------------
# Inspect one experience
# ------------------------------------------------

experience = experiences[0]

messages = experience["messages"]
embeddings = experience["embeddings"]

print()
print(f"Experience ID: {experience['experience_id']}")
print(f"Shape: {tuple(embeddings.shape)}")

print("\nMessages:")

for i, message in enumerate(messages):
    role = message["role"]
    content = message["content"]

    preview = content.replace("\n", " ")[:120]

    print(f"{i:02d} [{role}] {preview}")


# ------------------------------------------------
# Pairwise cosine similarity
# ------------------------------------------------

similarity = F.cosine_similarity(
    embeddings.unsqueeze(1),
    embeddings.unsqueeze(0),
    dim=-1,
)

print("\nCosine similarity matrix:\n")

print(similarity)


# ------------------------------------------------
# Find strongest relationships excluding self
# ------------------------------------------------

print("\nStrongest message relationships:\n")

num_messages = embeddings.shape[0]

pairs = []

for i in range(num_messages):
    for j in range(i + 1, num_messages):

        score = similarity[i, j].item()

        pairs.append(
            (
                score,
                i,
                j,
            )
        )

pairs.sort(reverse=True)


for score, i, j in pairs[:10]:

    role_i = messages[i]["role"]
    role_j = messages[j]["role"]

    text_i = messages[i]["content"].replace("\n", " ")[:80]
    text_j = messages[j]["content"].replace("\n", " ")[:80]

    print(f"{score:.4f}")
    print(f"  {i:02d} [{role_i}] {text_i}")
    print(f"  {j:02d} [{role_j}] {text_j}")
    print()
