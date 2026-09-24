"""ComfyUI nodes for APIMaster image and video models."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Tuple

from .client import (
    DEFAULT_BASE_URL,
    APIMasterClient,
    APIMasterError,
    bytes_to_tensor,
    resolve_api_key,
    stack_tensors,
    tensor_to_data_uri,
)

CATEGORY = "APIMaster"

SIZES = [
    "auto",
    "1:1",
    "3:2",
    "2:3",
    "4:3",
    "3:4",
    "5:4",
    "4:5",
    "16:9",
    "9:16",
    "2:1",
    "1:2",
    "3:1",
    "1:3",
    "21:9",
    "9:21",
]

# Verified against GET /v1/models on 2026-09-22. Ids are case-sensitive.
IMAGE_MODELS = ["gpt-image-2", "doubao-seedream-5-0-pro-260628", "gemini-3.1-flash-image", "midjourney-v8.2", "midjourney-niji-7"]
VIDEO_MODELS = ["sora-2", "sora-2-pro", "seedance-2.5", "seedance-2.0", "kling-v3-motion-control", "kling-v3-omni", "MiniMax-H3", "grok-imagine-video-1.5"]


def _progress(label: str):
    def report(status, attempt):
        if attempt % 5 == 0:
            print(f"[APIMaster] {label}: {status or 'pending'} ({attempt * 4 + 12}s)")

    return report


class APIMasterConfig:
    """Endpoint + key, shared by the other nodes.

    Leave `api_key` empty and set APIMASTER_API_KEY in the environment instead — a key
    typed into this field is saved inside the workflow JSON, which people share.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("APIMASTER_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "APIMaster endpoint and credentials. Prefer the APIMASTER_API_KEY environment variable over typing a key here."

    def build(self, base_url: str, api_key: str = ""):
        key = resolve_api_key(api_key)
        if not key:
            raise APIMasterError(
                None,
                "No API key found. Set APIMASTER_API_KEY before starting ComfyUI, "
                "or paste a key into this node.",
            )
        if api_key.strip():
            print(
                "[APIMaster] Warning: the key is stored in this workflow's JSON. "
                "Clear the field and use APIMASTER_API_KEY before sharing the workflow."
            )
        return ({"api_key": key, "base_url": base_url.rstrip("/")},)


class APIMasterListModels:
    """Print what the endpoint currently serves. Useful when a model id stops working."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("APIMASTER_CONFIG",),
                "filter": ("STRING", {"default": "image"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("models",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, config: Dict[str, Any], filter: str):
        client = APIMasterClient(config["api_key"], config["base_url"])
        ids = client.list_models()
        if filter.strip():
            needle = filter.strip().lower()
            ids = [i for i in ids if needle in i.lower()]
        text = "\n".join(ids) if ids else "(no models matched)"
        print(f"[APIMaster] {len(ids)} models matching '{filter}':\n{text}")
        return {"ui": {"text": [text]}, "result": (text,)}


class APIMasterTextToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("APIMASTER_CONFIG",),
                "prompt": ("STRING", {"default": "a corgi astronaut on the moon, cinematic", "multiline": True}),
                "model": (IMAGE_MODELS, {"default": "gpt-image-2"}),
                "size": (SIZES, {"default": "1:1"}),
                "resolution": (["1k", "2k", "4k"], {"default": "1k"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "mode": (["sync", "async"], {"default": "sync"}),
            },
            "optional": {
                "model_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "urls")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Text to image through APIMaster. Use async mode for 2k/4k jobs."

    def generate(
        self,
        config,
        prompt,
        model,
        size,
        resolution,
        batch_size,
        mode,
        model_override="",
    ):
        client = APIMasterClient(config["api_key"], config["base_url"])
        chosen = model_override.strip() or model
        payload: Dict[str, Any] = {"model": chosen, "prompt": prompt}
        if size != "auto":
            payload["size"] = size
        payload["resolution"] = resolution
        if batch_size > 1:
            payload["n"] = batch_size

        started = time.time()
        if mode == "async":
            urls = client.generate_image_async(payload, on_progress=_progress("image"))
        else:
            urls = client.generate_image_sync(payload, resolution=resolution)
        tensors = [bytes_to_tensor(client.download(url)) for url in urls]
        print(f"[APIMaster] {chosen} produced {len(urls)} image(s) in {time.time() - started:.1f}s")
        return (stack_tensors(tensors), "\n".join(urls))


class APIMasterImageToImage:
    """Image-to-image and inpainting. Reference images are uploaded inline as data URIs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("APIMASTER_CONFIG",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "replace the background with a desert sunset", "multiline": True}),
                "model": (IMAGE_MODELS, {"default": "gpt-image-2"}),
                "size": (SIZES, {"default": "auto"}),
                "resolution": (["1k", "2k", "4k"], {"default": "1k"}),
                "mode": (["sync", "async"], {"default": "sync"}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "mask_url": ("STRING", {"default": ""}),
                "model_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "urls")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        config,
        image,
        prompt,
        model,
        size,
        resolution,
        mode,
        image_2=None,
        mask_url="",
        model_override="",
    ):
        client = APIMasterClient(config["api_key"], config["base_url"])
        chosen = model_override.strip() or model

        references: List[str] = [tensor_to_data_uri(image, i) for i in range(min(len(image), 8))]
        if image_2 is not None:
            references += [tensor_to_data_uri(image_2, i) for i in range(min(len(image_2), 8))]
        if len(references) > 16:
            raise APIMasterError(None, "At most 16 reference images are accepted.")

        payload: Dict[str, Any] = {"model": chosen, "prompt": prompt, "image_urls": references}
        if size != "auto":
            payload["size"] = size
        payload["resolution"] = resolution
        if mask_url.strip():
            payload["mask_url"] = mask_url.strip()

        if mode == "async":
            urls = client.generate_image_async(payload, on_progress=_progress("image2image"))
        else:
            urls = client.generate_image_sync(payload, resolution=resolution)
        tensors = [bytes_to_tensor(client.download(url)) for url in urls]
        return (stack_tensors(tensors), "\n".join(urls))


def _output_dir() -> str:
    try:
        import folder_paths  # provided by ComfyUI at runtime

        return folder_paths.get_output_directory()
    except Exception:  # running outside ComfyUI, e.g. in tests
        path = os.path.join(os.getcwd(), "output")
        os.makedirs(path, exist_ok=True)
        return path


class APIMasterVideo:
    """Text-to-video and image-to-video. Saves an MP4 into ComfyUI's output folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("APIMASTER_CONFIG",),
                "prompt": ("STRING", {"default": "a waterfall forming a rainbow, cinematic", "multiline": True}),
                "model": (VIDEO_MODELS, {"default": "sora-2"}),
                "duration": ("INT", {"default": 4, "min": 4, "max": 20, "step": 4}),
                "resolution": (["720p", "1024p", "1080p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "filename_prefix": ("STRING", {"default": "apimaster"}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "model_override": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "task_id")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Submits a video job, polls it, and writes the MP4 to ComfyUI's output folder. "
        "For image-to-video always set aspect_ratio explicitly."
    )

    def generate(
        self,
        config,
        prompt,
        model,
        duration,
        resolution,
        aspect_ratio,
        filename_prefix,
        reference_image=None,
        model_override="",
    ):
        client = APIMasterClient(config["api_key"], config["base_url"])
        chosen = model_override.strip() or model

        if chosen == "sora-2" and resolution != "720p":
            print("[APIMaster] sora-2 only serves 720p — falling back. Use sora-2-pro for 1024p/1080p.")
            resolution = "720p"

        payload: Dict[str, Any] = {
            "model": chosen,
            "prompt": prompt,
            "duration": int(duration),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
        }
        if reference_image is not None:
            payload["image_urls"] = [tensor_to_data_uri(reference_image, 0, fmt="JPEG")]

        started = time.time()
        task_id = client.submit_video(payload)
        print(f"[APIMaster] video task {task_id} submitted, typically 1-3 minutes")
        result = client.wait_for_video(task_id, on_progress=_progress("video"))
        content_url = result.get("url") or f"{client.base_url}/videos/{task_id}/content"
        data = client.download(content_url, timeout=600)

        directory = _output_dir()
        filename = f"{filename_prefix}_{chosen}_{task_id[-8:]}.mp4"
        path = os.path.join(directory, filename)
        with open(path, "wb") as handle:
            handle.write(data)
        print(f"[APIMaster] saved {path} ({len(data) / 1e6:.1f} MB) in {time.time() - started:.0f}s")

        return {
            "ui": {"text": [f"{filename}  ({len(data) / 1e6:.1f} MB)"]},
            "result": (path, task_id),
        }


NODE_CLASS_MAPPINGS = {
    "APIMasterConfig": APIMasterConfig,
    "APIMasterListModels": APIMasterListModels,
    "APIMasterTextToImage": APIMasterTextToImage,
    "APIMasterImageToImage": APIMasterImageToImage,
    "APIMasterVideo": APIMasterVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "APIMasterConfig": "APIMaster Config",
    "APIMasterListModels": "APIMaster List Models",
    "APIMasterTextToImage": "APIMaster Text to Image",
    "APIMasterImageToImage": "APIMaster Image to Image",
    "APIMasterVideo": "APIMaster Video",
}
