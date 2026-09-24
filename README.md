# ComfyUI-APIMaster

Image and video generation nodes for ComfyUI, backed by [APIMaster](https://apimaster.ai/docs)
or any other OpenAI-compatible gateway.

**No extra Python dependencies.** The nodes use the standard library plus the
`torch` / `numpy` / `Pillow` that ComfyUI already ships, so installing them cannot break
your environment.

| Node | What it does |
| --- | --- |
| `APIMaster Config` | Endpoint + key, passed to the other nodes |
| `APIMaster Text to Image` | `gpt-image-2`, `doubao-seedream-5-0-pro-260628`, `gemini-3.1-flash-image`, Midjourney → `IMAGE` |
| `APIMaster Image to Image` | Reference images (and an optional mask) → `IMAGE` |
| `APIMaster Video` | `sora-2`, `sora-2-pro`, `seedance`, `kling` → an MP4 in your output folder |
| `APIMaster List Models` | Prints what the endpoint actually serves right now |

## Install

### ComfyUI Manager

Search for **APIMaster** in ComfyUI Manager and install. Restart ComfyUI.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/apimaster-ai/ComfyUI-APIMaster
# nothing to pip install
```

Restart ComfyUI.

## Set your key

Set an environment variable **before** starting ComfyUI:

```bash
# macOS / Linux
export APIMASTER_API_KEY=sk-...

# Windows PowerShell
$env:APIMASTER_API_KEY = "sk-..."
```

Then leave the `api_key` field in the Config node empty.

> **Why not just type it into the node?** Because widget values are saved inside the
> workflow JSON, and workflow JSON gets shared — posted in issues, uploaded with
> generated images, published on model sites. Keys leak that way constantly. The node
> prints a warning if you fill the field in anyway.

The key is also read from `~/.apimaster/config.json` if you use the
[`apimaster-cli`](https://github.com/apimaster-ai/apimaster-cli).

## Example workflows

Drag either file from [`example_workflows/`](example_workflows/) onto the ComfyUI canvas:

- `text_to_image.json` — prompt → `gpt-image-2` → preview
- `image_to_video.json` — load an image → `sora-2` → MP4

## Notes that will save you time

**Sync vs async for images.** Sync mode returns the image in one request and is simpler.
For `2k` and especially `4k` it can exceed the gateway's own timeout and come back as a
408 — switch the node's `mode` to `async` and it submits the job and polls instead.

**Aspect ratio in image-to-video.** Always set `aspect_ratio` explicitly. A portrait
reference image with no aspect set is treated as 16:9 by the gateway, and you get a
letterboxed result.

**Resolution per video model.** `sora-2` serves 720p only; `1024p` and `1080p` need
`sora-2-pro`. The node corrects this for you and logs when it does.

**Advanced image parameters change the price.** Quality, background and output-format
options narrow which upstream channels can serve a request, which can route you to a
more expensive one. The nodes send only the fields you actually set.

**Batch sizes.** If the endpoint returns images of different sizes in one batch, the node
raises instead of silently resizing. Set `n=1` or pass an explicit `size`.

## Which models are available

Model catalogs change. Drop an `APIMaster List Models` node into any workflow, set
`filter` to `image` or `video`, and run it — the list is printed to the node and to the
console. Or from a terminal:

```bash
npx @apimaster/cli models --kind image
```

## Use it with another provider

The nodes are not hardcoded to APIMaster. Point `base_url` at any OpenAI-compatible
endpoint that implements `/images/generations`, and use the `model_override` field for a
model id that is not in the dropdown.

## Development

```bash
python -m unittest discover -s tests -v
```

The tests run against a local mock server built into the repo — no key, no network, no
tokens spent.

Four of them exercise the ComfyUI `IMAGE` contract (tensor → data URI → bytes → tensor)
and need `torch`, so they skip on a plain Python install. Run them inside ComfyUI's own
environment for full coverage:

```bash
# from ComfyUI/custom_nodes/ComfyUI-APIMaster
../../.venv/Scripts/python.exe -m unittest discover -s tests    # Windows
../../venv/bin/python -m unittest discover -s tests             # macOS / Linux
```

Verified on ComfyUI with all five nodes appearing in `/object_info`, and both example
workflows validated against that live node registry.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `HTTP 401` | Key has stray whitespace or quotes, or it was never set in the environment ComfyUI runs in |
| `HTTP 404` | `base_url` is missing the `/v1` suffix |
| `HTTP 400 ... use the Images API` | A chat model id was selected for an image node |
| `HTTP 402` | Out of balance |
| `HTTP 408` | Sync generation timed out — switch to `async` |
| Nodes missing after install | Restart ComfyUI; check the console for an import error |

## License

MIT
