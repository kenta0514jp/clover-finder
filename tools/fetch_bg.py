#!/usr/bin/env python3
"""Fetch background photographs — images with no clover in them at all.

Every image in all three training sets is a photograph of clover. The detector
therefore never received a gradient saying "this is not a clover", and on a
desk it happily reported four-leaf at 82%. Ultralytics treats an image with no
label file as background, which is exactly the signal that is missing.

The searches below deliberately cover both halves of the failure:

  indoors / man-made   what the phone sees when it is not pointed at a lawn
  outdoors, no clover  grass, moss, ivy, gravel — the near misses that matter

Wikimedia Commons only, so everything here is freely licensed; the per-file
attribution is written to CREDITS.tsv beside the images.
"""

import os, sys, json, time, argparse, urllib.parse, urllib.request

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
API = 'https://commons.wikimedia.org/w/api.php'
"""Wikimedia's robot policy rejects a User-Agent with no way to reach the
operator; without the contact URL every download returns 429."""
UA = ('clover-finder/1.0 (https://github.com/kenta0514jp/clover-finder) '
      'python-urllib')
PAUSE = 1.0          # be a polite guest on a donated server

QUERIES = [
    # indoor and man-made: where the phone actually was when it misfired
    'computer keyboard desk', 'office desk workspace', 'living room interior',
    'kitchen counter', 'notebook paper handwriting', 'computer monitor screen',
    'bookshelf books', 'wooden table top', 'city street pavement',
    'brick wall texture', 'car parking lot', 'stairs indoor',
    # outdoors without clover: the near misses
    'lawn grass turf', 'moss forest floor', 'ivy leaves wall',
    'fern leaves', 'gravel path', 'dandelion meadow',
    'hedge shrub leaves', 'tree bark', 'fallen autumn leaves',
    'vegetable garden bed', 'wildflower meadow', 'bamboo leaves',
]


def api(params):
    url = API + '?' + urllib.parse.urlencode(dict(params, format='json'))
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def search(q, n):
    """Image titles for a query, largest-relevance first."""
    try:
        d = api({'action': 'query', 'generator': 'search',
                 'gsrsearch': f'filetype:bitmap {q}', 'gsrnamespace': 6,
                 'gsrlimit': n, 'prop': 'imageinfo',
                 'iiprop': 'url|extmetadata|size', 'iiurlwidth': 900})
    except Exception as e:
        print(f'  search failed: {e}')
        return []
    out = []
    for p in (d.get('query', {}).get('pages', {}) or {}).values():
        ii = (p.get('imageinfo') or [{}])[0]
        url = ii.get('thumburl') or ii.get('url')
        if not url or ii.get('width', 0) < 640:
            continue
        meta = ii.get('extmetadata', {})
        out.append({
            'title': p['title'],
            'url': url,
            'license': meta.get('LicenseShortName', {}).get('value', '?'),
            'author': meta.get('Artist', {}).get('value', '?')[:120],
            'page': 'https://commons.wikimedia.org/wiki/' +
                    urllib.parse.quote(p['title'].replace(' ', '_')),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-query', type=int, default=14)
    ap.add_argument('--out', default=os.path.join(BASE, 'data', 'bg'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import re
    strip = re.compile(r'<[^>]+>')
    seen, rows, n = set(), [], 0
    for q in QUERIES:
        time.sleep(PAUSE)
        got = search(q, a.per_query)
        print(f'{q:32s} {len(got):3d}')
        for it in got:
            if it['title'] in seen:
                continue
            seen.add(it['title'])
            name = f'bg{n:04d}.jpg'
            path = os.path.join(a.out, name)
            if not os.path.exists(path):
                try:
                    req = urllib.request.Request(it['url'], headers={'User-Agent': UA})
                    with urllib.request.urlopen(req, timeout=40) as r, \
                         open(path, 'wb') as f:
                        f.write(r.read())
                except Exception as e:
                    print(f'    skip {it["title"][:40]}: {e}')
                    continue
                time.sleep(PAUSE)
            rows.append((name, q, strip.sub('', it['author']).strip(),
                         it['license'], it['page']))
            n += 1

    with open(os.path.join(a.out, 'CREDITS.tsv'), 'w') as f:
        f.write('file\tquery\tauthor\tlicense\tsource\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print(f'\n{n} background images -> {a.out}')


if __name__ == '__main__':
    main()
