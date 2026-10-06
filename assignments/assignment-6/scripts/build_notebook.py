"""Build notebooks/assignment6_ssl_stl10.ipynb."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "assignment6_ssl_stl10.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


code("""
SID4 = 5330
SEED = SID4
SLICE = SID4 % 1000
HP_ID = SID4 % 6
CLS_A = SID4 % 10
CLS_B = (CLS_A + 1 + ((SID4 // 10) % 9)) % 10
print(f"SID4={SID4} SEED={SEED} SLICE={SLICE} HP_ID={HP_ID} CLS_A={CLS_A} CLS_B={CLS_B}")
""")

md("# Assignment 6: Self Supervised and Contrastive Learning on STL10\n\nOnly SEED is used. All parts use ResNet18 with no pretrained weights.")

code("""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms as T

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

train = datasets.STL10("../data", split="train", download=True)
test = datasets.STL10("../data", split="test", download=True)
unl = datasets.STL10("../data", split="unlabeled", download=True)
CLASSES = train.classes

rng = np.random.default_rng(SEED)
idx = np.concatenate([rng.choice(np.where(train.labels == c)[0], 50, replace=False) for c in range(10)])
Xl, yl = torch.from_numpy(train.data[idx]), torch.from_numpy(train.labels[idx]).long()
Xt, yt = torch.from_numpy(test.data), torch.from_numpy(test.labels).long()
Xu = torch.from_numpy(unl.data)
print(len(Xl), len(Xt), len(Xu), np.bincount(yl.numpy()))

MEAN = torch.tensor([0.447, 0.440, 0.407], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.260, 0.257, 0.271], device=device).view(1, 3, 1, 1)


def prep(x):
    return (x.to(device).float() / 255 - MEAN) / STD


def encoder():
    m = models.resnet18(weights=None)
    m.fc = nn.Identity()
    return m.to(device)


def batches(n, bs, shuffle=True):
    return (torch.randperm(n) if shuffle else torch.arange(n)).split(bs)


@torch.no_grad()
def embed(net, X):
    net.eval()
    return torch.cat([net(prep(X[b])).cpu() for b in batches(len(X), 256, False)])


def linear_probe(enc, epochs=20):
    for p in enc.parameters():
        p.requires_grad = False
    Ftr, Fte = embed(enc, Xl), embed(enc, Xt)
    clf = nn.Linear(512, 10)
    opt = torch.optim.Adam(clf.parameters(), 1e-3)
    for ep in range(epochs):
        for b in batches(len(Ftr), 32):
            loss = F.cross_entropy(clf(Ftr[b]), yl[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return (clf(Fte).argmax(1) == yt).float().mean().item()
""")

md("## Part A: Supervised with 500 Labels\n\n50 images per class, trained end to end for 15 epochs with random horizontal flips.")

code("""
sup = nn.Sequential(encoder(), nn.Linear(512, 10).to(device))
opt = torch.optim.Adam(sup.parameters(), 1e-3)
for ep in range(15):
    sup.train()
    for b in batches(len(Xl), 32):
        x = prep(Xl[b])
        x = torch.where(torch.rand(len(x), 1, 1, 1, device=device) < 0.5, x.flip(3), x)
        loss = F.cross_entropy(sup(x), yl[b].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {ep + 1} loss {loss.item():.3f}")
acc = {"Supervised": (embed(sup, Xt).argmax(1) == yt).float().mean().item()}
print(f"Part A test accuracy: {acc['Supervised']:.4f}")
""")

md("## Part B: Rotation Prediction\n\nEach unlabeled image (all 100,000) gets one random rotation of 0, 90, 180 or 270 degrees per epoch. Then the rotation head is dropped, the encoder is frozen and only a linear layer is trained on the 500 labels.")

code("""
rot = nn.Sequential(encoder(), nn.Linear(512, 4).to(device))
opt = torch.optim.Adam(rot.parameters(), 1e-3)
for ep in range(15):
    rot.train()
    correct = 0
    for b in batches(len(Xu), 256):
        x = prep(Xu[b])
        r = torch.randint(0, 4, (len(x),), device=device)
        for k in range(1, 4):
            x[r == k] = torch.rot90(x[r == k], k, (2, 3))
        out = rot(x)
        loss = F.cross_entropy(out, r)
        opt.zero_grad()
        loss.backward()
        opt.step()
        correct += (out.argmax(1) == r).sum().item()
    print(f"epoch {ep + 1} loss {loss.item():.3f} rotation acc {correct / len(Xu):.3f}")
acc["Rotation"] = linear_probe(rot[0])
print(f"Part B test accuracy: {acc['Rotation']:.4f}")
""")

md("## Part C: SimCLR\n\n20,000 unlabeled images. Two views per image from the four demo augmentations: random resized crop, horizontal flip, color jitter and random grayscale. NT Xent loss on cosine similarity with temperature 0.2 and an MLP projection head.")

code("""
aug = T.Compose([
    T.RandomResizedCrop(96, scale=(0.2, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
])
Xs = Xu[torch.from_numpy(rng.choice(len(Xu), 20000, replace=False))]


def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = (z @ z.T / tau).masked_fill(torch.eye(len(z), dtype=torch.bool, device=device), float("-inf"))
    n = len(z1)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(device)
    return F.cross_entropy(sim, target)


simclr = encoder()
head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128)).to(device)
opt = torch.optim.Adam(list(simclr.parameters()) + list(head.parameters()), 1e-3)
for ep in range(20):
    simclr.train()
    for b in batches(len(Xs), 256):
        v1 = prep(torch.stack([aug(x) for x in Xs[b]]))
        v2 = prep(torch.stack([aug(x) for x in Xs[b]]))
        loss = nt_xent(head(simclr(v1)), head(simclr(v2)))
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {ep + 1} loss {loss.item():.3f}")
acc["SimCLR"] = linear_probe(simclr)
print(f"Part C test accuracy: {acc['SimCLR']:.4f}")
""")

md("## Part D: Nearest Neighbors\n\nTest set embeddings are the 512 dimensional encoder outputs. Neighbors are ranked by cosine similarity. Precision at 5 (share of the 5 neighbors with the query's class) over all 8,000 test images is also reported.")

code("""
encoders = {"Supervised": sup[0], "Rotation": rot[0], "SimCLR": simclr}
queries = [int(rng.choice(np.where(yt.numpy() == c)[0])) for c in rng.choice(10, 3, replace=False)]
top = {}
for name, enc in encoders.items():
    E = F.normalize(embed(enc, Xt), dim=1)
    S = E @ E.T
    S.fill_diagonal_(-2)
    top[name] = S.topk(5).indices
    print(f"{name}: linear acc {acc[name]:.4f}, precision@5 {(yt[top[name]] == yt[:, None]).float().mean():.4f}")

fig, ax = plt.subplots(9, 6, figsize=(12, 19))
for i, q in enumerate(queries):
    for j, name in enumerate(encoders):
        row = ax[3 * i + j]
        for c, t in enumerate([q] + top[name][q].tolist()):
            row[c].imshow(Xt[t].permute(1, 2, 0))
            row[c].set_title(("Query: " if c == 0 else "") + CLASSES[yt[t]], fontsize=9,
                             color="black" if c == 0 or yt[t] == yt[q] else "red")
            row[c].axis("off")
        row[0].text(-10, 48, name, rotation=90, va="center", ha="right", fontsize=10)
plt.tight_layout()
plt.show()
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
