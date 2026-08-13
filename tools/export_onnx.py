#!/usr/bin/env python3
"""Export the detector to ONNX and prove the exported graph still agrees.

Every stage of this project has been checked against its reference before being
trusted — BatchNorm folding, the dense pass, the hand-written JS — each to
about 1e-7. The ONNX export gets the same treatment: run the same image through
PyTorch and through onnxruntime and compare the raw output tensor.
"""

import os, sys, argparse, shutil
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=os.path.join(
        BASE, 'data', 'yolo_runs', 'v1', 'weights', 'best.pt'))
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--half', action='store_true',
                    help='fp16 weights — half the download, tiny accuracy cost')
    ap.add_argument('--out', default=os.path.join(BASE, 'model.onnx'))
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    print('classes:', model.names)

    path = model.export(format='onnx', imgsz=a.imgsz, simplify=True,
                        half=a.half, dynamic=False, opset=12)
    print('exported:', path)
    shutil.copy(path, a.out)
    print(f'{a.out}  {os.path.getsize(a.out)/1024/1024:.1f} MB')

    # ---- agreement check -------------------------------------------------
    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime not installed — skipping the agreement check')
        return

    import torch
    rng = np.random.default_rng(7)
    x = rng.random((1, 3, a.imgsz, a.imgsz), dtype=np.float32)

    sess = ort.InferenceSession(a.out, providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    ort_out = sess.run(None, {iname: x.astype(np.float16 if a.half else np.float32)})[0]

    torch_model = model.model.float().eval()
    with torch.no_grad():
        t_out = torch_model(torch.from_numpy(x))
    t_out = (t_out[0] if isinstance(t_out, (list, tuple)) else t_out).numpy()

    o = np.asarray(ort_out, np.float32)
    if o.shape != t_out.shape:
        print(f'shape differs: onnx {o.shape} vs torch {t_out.shape}')
        return
    diff = np.abs(o - t_out)
    print(f'output shape {o.shape}')
    print(f'onnx-vs-torch  max {diff.max():.3e}   mean {diff.mean():.3e}  '
          f'{"OK" if diff.max() < (2e-2 if a.half else 1e-3) else "MISMATCH"}')
    print('\noutput layout: [1, 4 + n_classes, n_anchors] — rows 0..3 are '
          'cx,cy,w,h in pixels, the rest are per-class scores')


if __name__ == '__main__':
    main()
