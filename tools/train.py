#!/usr/bin/env python3
"""Train the tiny 3-vs-4 leaflet classifier.

Kept deliberately small (~47k parameters) because the forward pass has to be
hand-written in JS and the weights embedded in the artifact — no ONNX runtime,
no TF.js, nothing the CSP would block.

BatchNorm is used for training and folded into the preceding convolution at
export time, so the JS side only ever needs conv / relu / maxpool / linear.
"""

import os, sys, json, math, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CROPS = os.path.join(BASE, 'data', 'crops')   # overridden by --data
SIZE = 96


class TinyNet(nn.Module):
    """No padding anywhere, on purpose.

    With 'same' padding a 96x96 crop sees zeros at its border while the same
    window inside a whole frame sees real pixels, so a dense whole-frame pass
    scores differently from the crops the network was validated on — measured
    at 14 points mean, 50 points worst. Valid convolutions make the two
    mathematically identical, so what gets tested is what ships.

    96 -> 94 -> 47 -> 45 -> 22 -> 20 -> 10 -> 8 -> 4
    """
    def __init__(self, nc=2):
        super().__init__()
        ch = [3, 16, 32, 48, 64]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(4):
            self.convs.append(nn.Conv2d(ch[i], ch[i+1], 3, padding=0, bias=False))
            self.bns.append(nn.BatchNorm2d(ch[i+1]))
        self.drop = nn.Dropout(0.35)
        self.fc = nn.Linear(ch[-1], nc)

    def forward(self, x):
        for c, b in zip(self.convs, self.bns):
            x = F.max_pool2d(F.relu(b(c(x))), 2)
        x = x.mean(dim=(2, 3))
        return self.fc(self.drop(x))


def loaders(batch=64, workers=4, root=None):
    # full-circle rotation matters: a four-leaf is 90-degree symmetric and a
    # three-leaf 120-degree, so orientation must never become the cue
    train_tf = transforms.Compose([
        transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(SIZE, scale=(0.72, 1.0), ratio=(0.9, 1.11)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.35, 0.35, 0.35, 0.06),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([transforms.ToTensor()])
    root = root or CROPS
    tr = datasets.ImageFolder(os.path.join(root, 'train'), train_tf)
    va = datasets.ImageFolder(os.path.join(root, 'val'), val_tf)
    return (DataLoader(tr, batch, shuffle=True, num_workers=workers, drop_last=True),
            DataLoader(va, batch, shuffle=False, num_workers=workers),
            tr.classes)


def evaluate(model, dl, classes, four_idx):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            p = F.softmax(model(x), 1)[:, four_idx]
            probs.append(p); labels.append(y)
    p = torch.cat(probs); y = torch.cat(labels)
    is_four = (y == four_idx)
    acc = ((p > 0.5) == is_four).float().mean().item()

    rows = []
    for t in (0.5, 0.7, 0.9, 0.95, 0.99):
        pred = p > t
        tp = (pred & is_four).sum().item()
        fp = (pred & ~is_four).sum().item()
        fn = (~pred & is_four).sum().item()
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append((t, prec, rec, tp, fp))
    return acc, rows, p, is_four


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--out', default=os.path.join(BASE, 'data', 'tinynet.pt'))
    ap.add_argument('--data', default=CROPS)
    a = ap.parse_args()

    torch.manual_seed(20260813); random.seed(20260813)
    tr, va, classes = loaders(a.batch, root=a.data)
    four_idx = classes.index('four')
    print('classes:', classes, ' four_idx =', four_idx)

    model = TinyNet(len(classes))
    n_par = sum(p.numel() for p in model.parameters())
    print(f'parameters: {n_par:,}  ({n_par*2/1024:.0f} KB as float16)')

    # hard-negative mining leaves the classes lopsided; weight the loss back
    counts = torch.tensor(
        [len(os.listdir(os.path.join(a.data, 'train', c))) for c in classes],
        dtype=torch.float)
    cls_w = counts.sum() / (len(classes) * counts)
    print('train counts:', {c: int(n) for c, n in zip(classes, counts)},
          ' weights:', [round(float(w), 2) for w in cls_w])

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(tr))

    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = cnt = 0.0
        for x, y in tr:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=cls_w, label_smoothing=0.05)
            loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y); cnt += len(y)
        acc, rows, _, _ = evaluate(model, va, classes, four_idx)
        p90 = next(r for r in rows if r[0] == 0.9)
        print(f'ep{ep:3d}  loss {tot/cnt:.4f}  val_acc {acc*100:5.1f}%   '
              f'@0.9 prec {p90[1]*100 if p90[1]==p90[1] else 0:5.1f}% rec {p90[2]*100:5.1f}%')
        if acc > best:
            best = acc
            torch.save(dict(state=model.state_dict(), classes=classes,
                            four_idx=four_idx, size=SIZE), a.out)

    print(f'\nbest val accuracy {best*100:.1f}%  -> {a.out}')
    ck = torch.load(a.out, weights_only=False)
    model.load_state_dict(ck['state'])
    acc, rows, _, _ = evaluate(model, va, classes, four_idx)
    print(f'\n{"thresh":>7s} {"precision":>10s} {"recall":>8s} {"TP":>5s} {"FP":>5s}')
    for t, prec, rec, tp, fp in rows:
        ps = f'{prec*100:.1f}%' if prec == prec else '   n/a'
        print(f'{t:7.2f} {ps:>10s} {rec*100:7.1f}% {tp:5d} {fp:5d}')


if __name__ == '__main__':
    main()
