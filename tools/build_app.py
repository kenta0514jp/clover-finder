#!/usr/bin/env python3
"""Inline infer.js and the model weights into a single self-contained page.

The artifact CSP forbids fetching anything, so the network has to travel inside
the HTML itself.
"""
import json, os, sys

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
tpl = open(os.path.join(BASE, 'app.html')).read()
infer = open(os.path.join(BASE, 'infer.js')).read()
infer = infer.replace(
    "if (typeof module !== 'undefined') module.exports = { buildModel, decodeF32 };", "")
model = open(os.path.join(BASE, 'model.json')).read()

for tag, val in (('/*__INFER_JS__*/', infer), ('/*__MODEL_JSON__*/', model)):
    if tag not in tpl:
        sys.exit(f'placeholder {tag} missing from app.html')
    tpl = tpl.replace(tag, val)

out = os.path.join(BASE, 'index.html')
open(out, 'w').write(tpl)
print(f'{out}  {len(tpl)/1024:.0f} KB')
