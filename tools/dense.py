#!/usr/bin/env python3
"""Fully-convolutional evaluation: one pass over the whole frame.

TinyNet is conv/pool/global-average-pool/linear, so the classifier head is a
1x1 convolution and the global pool is just a 6x6 average. Feed a large image
instead of a 96x96 crop and the same maths produces a dense grid of window
scores at stride 16 — every position the sliding window would have visited,
for a fraction of the work.

  96x96 crop           -> 26M MACs -> 1 score
  256x192 whole frame  -> 142M MACs -> 192 scores

This module verifies the two agree, then measures how the detector behaves
over whole frames so the on-screen threshold can be set from evidence.
"""

import os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TinyNet, SIZE

CELL = 16          # 4 max-pools -> the score grid steps 16 input pixels


def feat_size(model, size=SIZE):
    """Side of the final feature map for one training crop (4 for valid convs)."""
    with torch.no_grad():
        x = torch.zeros(1, 3, size, size)
        for c, b in zip(model.convs, model.bns):
            x = F.max_pool2d(F.relu(b(c(x))), 2)
    return x.shape[-1]


def load(path=os.path.join(BASE, 'data', 'valid2.pt')):
    ck = torch.load(path, weights_only=False)
    m = TinyNet(len(ck['classes']))
    m.load_state_dict(ck['state'])
    m.eval()
    return m, ck['four_idx']


@torch.no_grad()
def dense_scores(model, four_idx, img_t):
    """img_t: (1,3,H,W) float. Returns (h,w) map of P(four) at stride CELL."""
    x = img_t
    for c, b in zip(model.convs, model.bns):
        x = F.max_pool2d(F.relu(b(c(x))), 2)
    # the training crop's global average pool is a fixed-size average here
    x = F.avg_pool2d(x, feat_size(model), stride=1)
    # the linear head applied at every location == 1x1 convolution
    w = model.fc.weight.view(model.fc.out_features, model.fc.in_features, 1, 1)
    logits = F.conv2d(x, w, model.fc.bias)
    return F.softmax(logits, dim=1)[0, four_idx].numpy()


@torch.no_grad()
def window_scores(model, four_idx, img_t, stride=CELL):
    """The slow way, for checking: crop every window and run the net on it."""
    _, _, H, W = img_t.shape
    out = []
    for y in range(0, H - SIZE + 1, stride):
        row = []
        for x in range(0, W - SIZE + 1, stride):
            p = F.softmax(model(img_t[:, :, y:y+SIZE, x:x+SIZE]), 1)[0, four_idx]
            row.append(float(p))
        out.append(row)
    return np.array(out)


def to_tensor(img):
    return torch.from_numpy(
        np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.0)


if __name__ == '__main__':
    model, fi = load()
    im = Image.open(os.path.join(BASE, 'testdata', 'p3-four-in-hand.jpg')).convert('RGB')
    im = im.resize((256, 192), Image.LANCZOS)
    t = to_tensor(im)

    d = dense_scores(model, fi, t)
    w = window_scores(model, fi, t)
    print('dense map  ', d.shape)
    print('window map ', w.shape)
    n = min(d.shape[0], w.shape[0]), min(d.shape[1], w.shape[1])
    a, b = d[:n[0], :n[1]], w[:n[0], :n[1]]

    # 'same' padding makes the border cells see zeros the crop never had, so
    # the interior is where the two must agree exactly
    inner_a, inner_b = a[1:-1, 1:-1], b[1:-1, 1:-1]
    print(f'interior max abs diff : {np.abs(inner_a - inner_b).max():.3e}')
    print(f'border   max abs diff : {np.abs(a - b).max():.3e}')
    print(f'positions evaluated   : {a.size} in one pass')
