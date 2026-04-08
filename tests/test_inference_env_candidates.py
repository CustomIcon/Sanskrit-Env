import importlib
import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def load_inference_with_env(env_overrides):
    with patch.dict(os.environ, env_overrides, clear=False):
        sys.modules.pop("inference", None)
        try:
            yield importlib.import_module("inference")
        finally:
            sys.modules.pop("inference", None)


class InferenceEnvCandidateTests(unittest.TestCase):
    def test_space_target_prefers_space_before_local(self):
        with load_inference_with_env(
            {
                "SANSKRIT_ENV_TARGET": "space",
                "HF_SPACE_URL": "https://huggingface.co/spaces/Adityahars/Sanskrit-env",
                "SANSKRIT_ENV_URL": "http://localhost:7860",
                "ENV_URL": "",
                "OPENENV_BASE_URL": "",
                "SPACE_URL": "",
                "SPACE_HOST": "",
                "HF_SPACE_ID": "",
                "SPACE_ID": "",
                "DEFAULT_HF_SPACE_URL": "",
            }
        ) as module:
            candidates = module._build_env_url_candidates()

            self.assertEqual(candidates[0], "https://adityahars-sanskrit-env.hf.space")
            self.assertIn("http://localhost:7860", candidates)

    def test_local_target_prefers_local_before_space(self):
        with load_inference_with_env(
            {
                "SANSKRIT_ENV_TARGET": "local",
                "HF_SPACE_URL": "https://demo-space.hf.space",
                "SANSKRIT_ENV_URL": "http://localhost:7860",
                "ENV_URL": "",
                "OPENENV_BASE_URL": "",
                "SPACE_URL": "",
                "SPACE_HOST": "",
                "HF_SPACE_ID": "",
                "SPACE_ID": "",
                "DEFAULT_HF_SPACE_URL": "",
            }
        ) as module:
            candidates = module._build_env_url_candidates()

            self.assertEqual(candidates[0], "http://localhost:7860")
            self.assertIn("https://demo-space.hf.space", candidates)

    def test_space_page_url_and_space_id_are_normalized(self):
        with load_inference_with_env({"DEFAULT_HF_SPACE_URL": ""}) as module:
            self.assertEqual(
                module._normalize_env_url("https://huggingface.co/spaces/Adityahars/Sanskrit-env"),
                "https://adityahars-sanskrit-env.hf.space",
            )
            self.assertEqual(
                module._hf_space_url_from_id("Adityahars/Sanskrit-env"),
                "https://adityahars-sanskrit-env.hf.space",
            )


if __name__ == "__main__":
    unittest.main()