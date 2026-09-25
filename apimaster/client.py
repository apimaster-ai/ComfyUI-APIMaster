"""HTTP client for APIMaster-style OpenAI-compatible endpoints.

Deliberately built on the standard library only. ComfyUI installs are fragile and a
custom node that drags in its own pinned `requests`/`httpx` is a common way to break
someone's environment.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BASE_URL = "https://apimaster.ai/v1"
USER_AGENT = "ComfyUI-APIMaster/0.1.0"

# Documented client-side read timeouts per resolution tier. Generation is slow and the
# default urllib timeout would abort a perfectly healthy 4k job.
SYNC_TIMEOUT = {"1k": 200, "2k": 320, "4k": 620}


class APIMasterError(RuntimeError):
    """Raised with a message that is useful inside the ComfyUI error toast."""

    HINTS = {
        400: "Bad request. Check the model id and parameters; image models must not be called on /chat/completions.",
        401: "Unauthorized. The API key is wrong, expired, or has stray whitespace.",
        402: "Insufficient balance. Top up the account.",
        403: "Forbidden. This key may not be allowed to use this model.",
        404: "Not found. Check the base URL — it should end with /v1.",
        408: "Generation timed out. Lower the resolution, or switch the node to async mode.",
        429: "Rate limited. Retry in a moment.",
        500: "Upstream error. Retry.",
        502: "Upstream error. Retry.",
        503: "Upstream unavailable. Retry.",
    }

    def __init__(self, status: Optional[int], detail: str = ""):
        hint = self.HINTS.get(status or 0, "Request failed.")
        prefix = f"HTTP {status}: " if status else ""
        super().__init__(f"{prefix}{hint} {detail}".strip())
        self.status = status


def uses_default_host(base_url: str) -> bool:
    """True for https://apimaster.ai (or a subdomain), the only place a stored key may go."""
    parsed = urllib.parse.urlparse(base_url.strip())
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (host == "apimaster.ai" or host.endswith(".apimaster.ai"))


def resolve_api_key(explicit: str = "", base_url: str = DEFAULT_BASE_URL) -> str:
    """Key lookup order: node field, environment, then ~/.apimaster/config.json.

    Workflow JSON gets shared, posted in issues and uploaded to civitai. Leaving the key
    field empty and using the environment is the safe path, so it is the documented one.

    A stored key (environment or config file) is only released for https://apimaster.ai.
    base_url is a node input, so anyone who hands you a workflow chooses it; without this
    check, a workflow pointing at their server would collect your key.
    """
    if explicit and explicit.strip():
        return explicit.strip()
    if not uses_default_host(base_url):
        return ""
    value = os.environ.get("APIMASTER_API_KEY")
    if value:
        return value.strip()
    config_path = os.path.join(os.path.expanduser("~"), ".apimaster", "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return str(json.load(handle).get("key", "")).strip()
    except (OSError, ValueError):
        return ""


def _request(
    url: str,
    api_key: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method=method)
    request.add_header("Authorization", f"Bearer {api_key}")
    request.add_header("Content-Type", "application/json")
    request.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise APIMasterError(exc.code, detail) from None
    except urllib.error.URLError as exc:
        raise APIMasterError(None, f"Cannot reach {url}: {exc.reason}") from None
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return raw


class APIMasterClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL):
        if not api_key:
            raise APIMasterError(
                None,
                "No API key. Set APIMASTER_API_KEY in your environment, or paste a key "
                "into the APIMaster Config node.",
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    # -- models ---------------------------------------------------------------

    def list_models(self) -> List[str]:
        data = _request(f"{self.base_url}/models", self.api_key, timeout=30)
        return [m.get("id") for m in (data or {}).get("data", []) if m.get("id")]

    # -- images ---------------------------------------------------------------

    def generate_image_sync(self, payload: Dict[str, Any], resolution: str = "1k") -> List[str]:
        timeout = SYNC_TIMEOUT.get(resolution, 200)
        data = _request(
            f"{self.base_url}/images/generations",
            self.api_key,
            method="POST",
            payload=payload,
            timeout=timeout,
        )
        urls = [item.get("url") for item in (data or {}).get("data", []) if item.get("url")]
        if not urls:
            raise APIMasterError(None, f"No image URL in response: {json.dumps(data)[:300]}")
        return urls

    def generate_image_async(
        self, payload: Dict[str, Any], on_progress=None
    ) -> List[str]:
        submit = _request(
            f"{self.base_url}/images/generations/async",
            self.api_key,
            method="POST",
            payload=payload,
            timeout=60,
        )
        items = (submit or {}).get("data") or []
        task_id = items[0].get("task_id") if items else None
        if not task_id:
            raise APIMasterError(None, f"No task_id in response: {json.dumps(submit)[:300]}")
        result = self._poll_task(task_id, payload.get("model", ""), on_progress=on_progress)
        images = (result.get("result") or {}).get("images") or []
        urls: List[str] = []
        for image in images:
            url = image.get("url")
            if isinstance(url, list):
                urls.extend(url)
            elif url:
                urls.append(url)
        if not urls:
            raise APIMasterError(None, f"Task finished without images: {json.dumps(result)[:300]}")
        return urls

    def _poll_task(self, task_id: str, model: str, on_progress=None) -> Dict[str, Any]:
        # First poll is delayed: the task is never ready sooner and an immediate poll
        # just burns a request.
        time.sleep(12)
        query = urllib.parse.urlencode({"model": model}) if model else ""
        url = f"{self.base_url}/tasks/{urllib.parse.quote(task_id)}" + (f"?{query}" if query else "")
        for attempt in range(200):
            data = _request(url, self.api_key, timeout=30) or {}
            payload = data.get("data", data)
            status = payload.get("status")
            if on_progress:
                on_progress(status, attempt)
            if status == "completed":
                return payload
            if status in ("failed", "error", "cancelled"):
                raise APIMasterError(None, f"Task {status}: {json.dumps(data)[:300]}")
            # pending / processing / in_progress all mean "keep waiting"
            time.sleep(4)
        raise APIMasterError(None, "Gave up polling the image task after ~13 minutes.")

    # -- video ----------------------------------------------------------------

    def submit_video(self, payload: Dict[str, Any]) -> str:
        data = _request(
            f"{self.base_url}/videos/generations",
            self.api_key,
            method="POST",
            payload=payload,
            timeout=60,
        ) or {}
        items = data.get("data") or []
        task_id = (items[0].get("task_id") if items else None) or data.get("id")
        if not task_id:
            raise APIMasterError(None, f"No task id in response: {json.dumps(data)[:300]}")
        return task_id

    def wait_for_video(self, task_id: str, on_progress=None) -> Dict[str, Any]:
        time.sleep(15)
        for attempt in range(240):
            data = _request(
                f"{self.base_url}/videos/{urllib.parse.quote(task_id)}", self.api_key, timeout=30
            ) or {}
            status = data.get("status")
            if on_progress:
                on_progress(status, attempt)
            if status == "completed":
                return data
            if status in ("failed", "error", "cancelled"):
                raise APIMasterError(None, f"Task {status}: {json.dumps(data)[:300]}")
            time.sleep(4)
        raise APIMasterError(None, "Gave up polling the video task after ~16 minutes.")

    def download(self, url: str, timeout: int = 300) -> bytes:
        request = urllib.request.Request(url)
        # Only our own domain needs the bearer token; upstream CDNs reject unknown headers.
        if urllib.parse.urlparse(url).netloc in urllib.parse.urlparse(self.base_url).netloc:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        request.add_header("User-Agent", USER_AGENT)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise APIMasterError(exc.code, f"Download failed for {url}") from None
        except urllib.error.URLError as exc:
            raise APIMasterError(None, f"Download failed: {exc.reason}") from None


# -- ComfyUI tensor <-> image helpers -----------------------------------------


def tensor_to_data_uri(image_tensor, index: int = 0, fmt: str = "PNG") -> str:
    """Convert one image out of a ComfyUI IMAGE batch into a data URI."""
    import numpy as np
    from PIL import Image

    array = image_tensor[index].cpu().numpy() if hasattr(image_tensor[index], "cpu") else image_tensor[index]
    array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(array)
    buffer = io.BytesIO()
    pil.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def bytes_to_tensor(payload: bytes):
    """Decode downloaded image bytes into a ComfyUI IMAGE tensor of shape [1, H, W, 3]."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    pil = Image.open(io.BytesIO(payload))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    array = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def stack_tensors(tensors: List[Any]):
    """Concatenate single images into one batch, padding is not attempted on purpose.

    Mixed sizes in one batch are a silent source of confusion, so we return the first
    size and tell the user instead of resizing behind their back.
    """
    import torch

    if not tensors:
        raise APIMasterError(None, "No images were produced.")
    first_shape = tensors[0].shape
    same = [t for t in tensors if t.shape == first_shape]
    if len(same) != len(tensors):
        raise APIMasterError(
            None,
            "The endpoint returned images of different sizes in one batch. "
            "Set n=1, or pass an explicit size so every image matches.",
        )
    return torch.cat(same, dim=0)
