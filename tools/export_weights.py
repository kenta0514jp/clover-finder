#!/usr/bin/env python3
"""Export TinyNet for the browser.

BatchNorm is folded into the preceding convolution so the JS side only needs
conv / relu / maxpool / global-average-pool / linear — no inference runtime,
nothing for the artifact CSP to block.

Writes a JSON file holding base64 float32 tensors, ready to inline.
"""

import os, sys, json, base64, argparse
import numpy as np
import torch

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TinyNet, SIZE


def fold(conv_w, bn):
    """conv (no bias) + BN  ->  conv with weight and bias."""
    gamma = bn['weight'].numpy()
    beta = bn['bias'].numpy()
    mean = bn['running_mean'].numpy()
    var = bn['running_var'].numpy()
    eps = 1e-5
    scale = gamma / np.sqrt(var + eps)
    w = conv_w.numpy() * scale[:, None, None, None]
    b = beta - mean * scale
    return w.astype(np.float32), b.astype(np.float32)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr, np.float32).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(BASE, 'data', 'valid2.pt'))
    ap.add_argument('--out', default=os.path.join(BASE, 'model.json'))
    a = ap.parse_args()

    ck = torch.load(a.model, weights_only=False)
    sd = ck['state']
    net = TinyNet(len(ck['classes']))
    net.load_state_dict(sd)
    net.eval()

    layers = []
    for i in range(4):
        cw = sd[f'convs.{i}.weight']
        bn = {k: sd[f'bns.{i}.{k}'] for k in
              ('weight', 'bias', 'running_mean', 'running_var')}
        w, b = fold(cw, bn)
        layers.append(dict(type='conv3', cin=int(w.shape[1]), cout=int(w.shape[0]),
                           w=b64(w), b=b64(b)))

    fw = sd['fc.weight'].numpy().astype(np.float32)
    fb = sd['fc.bias'].numpy().astype(np.float32)

    from dense import feat_size, CELL
    model = dict(
        size=SIZE,
        feat=feat_size(net),
        cell=CELL,
        classes=ck['classes'],
        four_idx=int(ck['four_idx']),
        layers=layers,
        fc=dict(cin=int(fw.shape[1]), cout=int(fw.shape[0]), w=b64(fw), b=b64(fb)),
        note='deton/detect4clover (CC BY 4.0) trained crops; BN folded into conv',
    )
    with open(a.out, 'w') as f:
        json.dump(model, f)

    n = sum(l['cin'] * l['cout'] * 9 + l['cout'] for l in layers)
    n += fw.size + fb.size
    print(f'{n:,} weights -> {os.path.getsize(a.out)/1024:.0f} KB json ({a.out})')

    # sanity: folded maths must match the torch model on random input
    x = torch.rand(2, 3, SIZE, SIZE)
    with torch.no_grad():
        ref = torch.softmax(net(x), 1).numpy()

    def relu(v): return np.maximum(v, 0)

    def conv3(inp, w, b):
        C, H, W = inp.shape
        O = w.shape[0]
        Ho, Wo = H - 2, W - 2            # valid: no padding
        out = np.empty((O, Ho, Wo), np.float32)
        for o in range(O):
            acc = np.full((Ho, Wo), b[o], np.float32)
            for c in range(C):
                for dy in range(3):
                    for dx in range(3):
                        acc += w[o, c, dy, dx] * inp[c, dy:dy+Ho, dx:dx+Wo]
            out[o] = acc
        return out

    def pool2(inp):
        C, H, W = inp.shape
        return inp[:, :H//2*2, :W//2*2].reshape(C, H//2, 2, W//2, 2).max(axis=(2, 4))

    got = []
    for k in range(x.shape[0]):
        v = x[k].numpy()
        for l in layers:
            w = np.frombuffer(base64.b64decode(l['w']), np.float32).reshape(
                l['cout'], l['cin'], 3, 3)
            bb = np.frombuffer(base64.b64decode(l['b']), np.float32)
            v = pool2(relu(conv3(v, w, bb)))
        v = v.mean(axis=(1, 2))
        logits = fw @ v + fb
        e = np.exp(logits - logits.max())
        got.append(e / e.sum())
    got = np.array(got)
    err = float(np.abs(got - ref).max())
    print(f'folded-vs-torch max prob error: {err:.2e}  {"OK" if err < 2e-3 else "MISMATCH"}')


if __name__ == '__main__':
    main()
