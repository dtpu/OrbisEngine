#!/usr/bin/env python3
"""Generate a Marble world in multi-image reconstruction mode, from frames chosen by
scripts/select_world_mode.py, with a prompt written by scripts/world_prompt.py.

Same flow, flags and ops log as marble_image_world.py; only the prompt body differs:

  {"type": "multi-image", "reconstruct_images": true, "disable_recaption": true,
   "text_prompt": "...", "multi_image_prompt": [{"azimuth": <deg>, "content": {...}}, ...]}

`reconstruct_images` is the documented reconstruction path (up to 8 images, 4 otherwise) and is the
only Marble mode that uses real pixels from more than one direction. `disable_recaption` IS accepted
on the multi-image variant -- share/QUALITY-RESEARCH.md says it is not listed, which was read off
the prose rather than the schema; MultiImagePrompt-Input carries both `disable_recaption` and
`text_prompt`.

  WLT_API_KEY=... python3 scripts/marble_multi_world.py submit \
      --images dir/f_0003.png:359.7 dir/f_0049.png:340.7 ... \
      --name gym-multi --marble-dir .context/marble \
      --spz public/marble-gym-multi.spz --thumb share/gym-multi-thumb.png \
      --prompt-file .context/prompt/gym.json [--seed 7]
  ... poll <operation_id> --name ... ; ... fetch <world_id> --name ...
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marble_image_world import call, fetch, log, poll                      # noqa: E402


def upload(a, path: Path) -> str:
    ext = path.suffix.lstrip('.').lower()
    _, prep = call('POST', '/marble/v1/media-assets:prepare_upload',
                   {'file_name': path.name, 'kind': 'image', 'extension': ext})
    asset = prep['media_asset']['media_asset_id']
    status, _ = call('PUT', prep['upload_info']['upload_url'], raw=path.read_bytes(),
                     headers={'x-goog-content-length-range': '0,104857600',
                              'Content-Type': f'image/{"jpeg" if ext in ("jpg", "jpeg") else ext}'}, timeout=600)
    log(a, f'image {path.name} ({path.stat().st_size} bytes, asset {asset}, upload {status})')
    return asset


def submit(a):
    items = []
    for spec in a.images:
        p, _, az = spec.partition(':')
        asset = upload(a, Path(p))
        entry = {'content': {'source': 'media_asset', 'media_asset_id': asset}}
        if az:
            entry['azimuth'] = float(az)
        items.append(entry)
    prompt = a.prompt
    if a.prompt_file:
        prompt = json.loads(Path(a.prompt_file).read_text())['text_prompt']
    body = {'display_name': a.name, 'model': a.model,
            'world_prompt': {'type': 'multi-image', 'reconstruct_images': True,
                             'disable_recaption': True, 'text_prompt': prompt,
                             'multi_image_prompt': items}}
    if a.seed is not None:
        body['seed'] = a.seed
    (Path(a.marble_dir) / f'{a.name}-request.json').write_text(json.dumps(body, indent=1))
    _, op = call('POST', '/marble/v1/worlds:generate', body)
    op_id = op['operation_id']
    log(a, f'op {op_id} submitted ({len(items)} images, {a.model}, seed {a.seed}, '
           f'reconstruct_images, disable_recaption)')
    log(a, f'prompt: {prompt}')
    poll(a, op_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['submit', 'poll', 'fetch'])
    ap.add_argument('target', nargs='?', help='operation id (poll) or world id (fetch)')
    ap.add_argument('--images', nargs='+', default=[], help='path[:azimuth_deg] per view, in any order')
    ap.add_argument('--name', required=True); ap.add_argument('--marble-dir', required=True)
    ap.add_argument('--ops', help='ops log stem (default name)')
    ap.add_argument('--spz'); ap.add_argument('--thumb'); ap.add_argument('--interval', type=int, default=60)
    ap.add_argument('--prompt'); ap.add_argument('--prompt-file', help='JSON from scripts/world_prompt.py')
    ap.add_argument('--seed', type=int); ap.add_argument('--model', default='marble-1.1')
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    if a.mode == 'submit' and not a.images:
        sys.exit('submit needs --images')
    {'submit': submit, 'poll': lambda a: poll(a, a.target),
     'fetch': lambda a: fetch(a, a.target)}[a.mode](a)


if __name__ == '__main__':
    main()
