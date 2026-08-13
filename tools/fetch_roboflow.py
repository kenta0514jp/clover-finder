#!/usr/bin/env python3
"""Download the public four-leaf clover datasets from Roboflow Universe.

Reads the API key from ~/.roboflow_key so it never appears in a command line
or a transcript. Downloads in Pascal VOC form, which matches the annotation
format tools/make_crops.py already reads for the deton set.
"""

import os, sys, json, shutil, argparse, urllib.request, urllib.parse, urllib.error

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(BASE, 'data', 'roboflow')

# workspace / project / version, as they appear in the Universe URL
PROJECTS = [
    ('test-ara07', '4-leaf-clover-detect'),
    ('adam-fonagy', 'hunting-for-four-leaf-clovers'),
]


def key():
    p = os.path.expanduser('~/.roboflow_key')
    if not os.path.exists(p):
        sys.exit('~/.roboflow_key not found — write the API key there first')
    return open(p).read().strip()


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {'_http': e.code, '_body': e.read()[:400].decode('utf8', 'replace')}
    except Exception as e:
        return {'_error': str(e)}


def describe(k, ws, proj):
    info = get(f'https://api.roboflow.com/{ws}/{proj}?api_key={k}')
    if '_http' in info or '_error' in info:
        return None, info
    p = info.get('project', {})
    versions = info.get('versions', [])
    return dict(name=p.get('name'), images=p.get('images'),
                classes=p.get('classes'), license=p.get('license'),
                versions=[(v.get('id', '').split('/')[-1], v.get('images')) for v in versions]), info


def download(k, ws, proj, version, fmt='voc', tries=40, wait=15):
    """Roboflow builds the export asynchronously; poll until the link appears."""
    import time
    url = f'https://api.roboflow.com/{ws}/{proj}/{version}/{fmt}?api_key={k}'
    link = None
    for attempt in range(tries):
        meta = get(url)
        if '_http' in meta or '_error' in meta:
            return None, meta
        link = (meta.get('export') or {}).get('link')
        if link:
            break
        print(f'    building export… {meta.get("progress", 0):.0%} '
              f'(attempt {attempt+1}/{tries})', flush=True)
        time.sleep(wait)
    if not link:
        return None, {'timeout': 'export never finished'}
    dest_zip = os.path.join(OUT, f'{ws}__{proj}.zip')
    os.makedirs(OUT, exist_ok=True)
    urllib.request.urlretrieve(link, dest_zip)
    dest_dir = os.path.join(OUT, f'{ws}__{proj}')
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.unpack_archive(dest_zip, dest_dir)
    os.remove(dest_zip)
    return dest_dir, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list-only', action='store_true')
    a = ap.parse_args()
    k = key()

    for ws, proj in PROJECTS:
        d, raw = describe(k, ws, proj)
        if d is None:
            print(f'{ws}/{proj}: unavailable — {str(raw)[:160]}')
            continue
        print(f'\n{ws}/{proj}')
        print(f'  name     : {d["name"]}')
        print(f'  images   : {d["images"]}')
        print(f'  classes  : {d["classes"]}')
        print(f'  license  : {d["license"]}')
        print(f'  versions : {d["versions"]}')
        if a.list_only or not d['versions']:
            continue
        ver = d['versions'][0][0]          # the list is newest-first
        dest, meta = download(k, ws, proj, ver)
        if dest is None:
            print(f'  download failed: {str(meta)[:200]}')
            continue
        n_img = sum(1 for r, _, fs in os.walk(dest) for f in fs
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        n_xml = sum(1 for r, _, fs in os.walk(dest) for f in fs if f.endswith('.xml'))
        print(f'  -> {dest}  ({n_img} images, {n_xml} annotations)')


if __name__ == '__main__':
    main()
