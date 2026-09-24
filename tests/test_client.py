"""Tests for the APIMaster client. Run with: python -m unittest discover tests

They exercise the HTTP paths against a local mock, so no key and no network are needed.
Tensor helpers that need torch are skipped when torch is not installed.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apimaster.client import (  # noqa: E402
    APIMasterClient,
    APIMasterError,
    resolve_api_key,
)
from tests import mock_server  # noqa: E402


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = mock_server.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def client(self):
        return APIMasterClient("test-key-1234567890", self.base)

    def test_requires_a_key(self):
        with self.assertRaises(APIMasterError) as ctx:
            APIMasterClient("", self.base)
        self.assertIn("APIMASTER_API_KEY", str(ctx.exception))

    def test_lists_models(self):
        ids = self.client().list_models()
        self.assertIn("gpt-image-2", ids)
        self.assertIn("sora-2", ids)

    def test_rejects_a_bad_key_with_a_useful_message(self):
        bad = APIMasterClient("x", self.base)
        with self.assertRaises(APIMasterError) as ctx:
            bad.list_models()
        self.assertEqual(ctx.exception.status, 401)
        self.assertIn("stray whitespace", str(ctx.exception))

    def test_sync_image_returns_urls(self):
        urls = self.client().generate_image_sync({"model": "gpt-image-2", "prompt": "a cat", "n": 2})
        self.assertEqual(len(urls), 2)
        self.assertTrue(urls[0].endswith(".png"))

    def test_missing_prompt_is_a_400_with_the_upstream_detail(self):
        with self.assertRaises(APIMasterError) as ctx:
            self.client().generate_image_sync({"model": "gpt-image-2"})
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("prompt is required", str(ctx.exception))

    def test_downloads_an_image(self):
        client = self.client()
        url = client.generate_image_sync({"model": "gpt-image-2", "prompt": "a cat"})[0]
        payload = client.download(url)
        self.assertTrue(payload.startswith(b"\x89PNG"))

    def test_video_submit_and_poll(self):
        client = self.client()
        task_id = client.submit_video({"model": "sora-2", "prompt": "a waterfall", "duration": 4})
        self.assertEqual(task_id, "task_mock")
        # The client sleeps before its first poll; keep the test honest but quick by
        # checking the polling contract rather than the wall-clock behaviour.
        original_sleep = mock_server.time.sleep
        try:
            import apimaster.client as client_module

            client_module.time.sleep = lambda _s: None
            result = client.wait_for_video(task_id)
        finally:
            mock_server.time.sleep = original_sleep
        self.assertEqual(result["status"], "completed")
        self.assertIn("/content", result["url"])
        self.assertTrue(client.download(result["url"]).startswith(b"\x00\x00\x00\x18ftyp"))

    def test_async_image_polls_until_complete(self):
        client = self.client()
        import apimaster.client as client_module

        client_module.time.sleep = lambda _s: None
        urls = client.generate_image_async({"model": "gpt-image-2", "prompt": "a cat"})
        self.assertEqual(len(urls), 1)
        self.assertTrue(urls[0].endswith("a.png"))


class KeyResolutionTest(unittest.TestCase):
    def test_explicit_key_wins(self):
        os.environ["APIMASTER_API_KEY"] = "from-env"
        try:
            self.assertEqual(resolve_api_key("  explicit  "), "explicit")
        finally:
            del os.environ["APIMASTER_API_KEY"]

    def test_environment_is_used_when_the_field_is_empty(self):
        os.environ["APIMASTER_API_KEY"] = "from-env"
        try:
            self.assertEqual(resolve_api_key(""), "from-env")
        finally:
            del os.environ["APIMASTER_API_KEY"]


class TensorHelpersTest(unittest.TestCase):
    """Covers the ComfyUI IMAGE contract: a float32 tensor of shape [B, H, W, 3] in 0..1."""

    def test_numpy_batch_converts_to_a_data_uri(self):
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"numpy unavailable ({type(exc).__name__})")
        from apimaster.client import tensor_to_data_uri

        batch = np.zeros((1, 4, 4, 3), dtype="float32")
        batch[0, 0, 0] = [1.0, 0.0, 0.0]
        uri = tensor_to_data_uri(batch, 0)
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        self.assertGreater(len(uri), 80)

    def _torch(self):
        # Not just ImportError: a broken Windows torch install raises OSError on a
        # missing DLL, which would turn a skip into a red CI run.
        try:
            import torch

            return torch
        except Exception as exc:  # pragma: no cover - only outside a working ComfyUI env
            self.skipTest(f"torch unavailable ({type(exc).__name__}); run inside ComfyUI's venv")

    def test_downloaded_bytes_become_a_comfyui_image_tensor(self):
        torch = self._torch()
        from apimaster.client import bytes_to_tensor
        from tests.mock_server import TINY_PNG

        tensor = bytes_to_tensor(TINY_PNG)
        self.assertEqual(tensor.dim(), 4)
        self.assertEqual(tensor.shape[0], 1)          # batch
        self.assertEqual(tensor.shape[3], 3)          # RGB, alpha dropped
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_tensor_survives_a_round_trip_through_the_api_format(self):
        """tensor -> data URI (what we upload) -> bytes -> tensor (what we return)."""
        torch = self._torch()
        import base64

        from apimaster.client import bytes_to_tensor, tensor_to_data_uri

        original = torch.zeros((1, 8, 6, 3), dtype=torch.float32)
        original[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
        original[0, 7, 5] = torch.tensor([0.0, 0.0, 1.0])

        uri = tensor_to_data_uri(original, 0)
        payload = base64.b64decode(uri.split(",", 1)[1])
        restored = bytes_to_tensor(payload)

        self.assertEqual(restored.shape, original.shape)
        # PNG is lossless, so the corner pixels must come back exactly.
        self.assertTrue(torch.allclose(restored[0, 0, 0], original[0, 0, 0], atol=1 / 255))
        self.assertTrue(torch.allclose(restored[0, 7, 5], original[0, 7, 5], atol=1 / 255))

    def test_mismatched_sizes_raise_instead_of_silently_resizing(self):
        torch = self._torch()
        from apimaster.client import APIMasterError, stack_tensors

        a = torch.zeros((1, 4, 4, 3))
        b = torch.zeros((1, 8, 8, 3))
        with self.assertRaises(APIMasterError) as ctx:
            stack_tensors([a, b])
        self.assertIn("different sizes", str(ctx.exception))

    def test_same_sizes_stack_into_one_batch(self):
        torch = self._torch()
        from apimaster.client import stack_tensors

        batch = stack_tensors([torch.zeros((1, 4, 4, 3)), torch.ones((1, 4, 4, 3))])
        self.assertEqual(batch.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
