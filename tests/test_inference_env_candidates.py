import asyncio
import importlib
import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch


@contextmanager
def load_inference_with_env(env_overrides):
    with patch.dict(os.environ, env_overrides, clear=False):
        sys.modules.pop("inference", None)
        try:
            yield importlib.import_module("inference")
        finally:
            sys.modules.pop("inference", None)


class InferenceConfigTests(unittest.TestCase):
    def test_defaults_match_submission_contract(self):
        with load_inference_with_env(
            {
                "API_BASE_URL": "",
                "MODEL_NAME": "",
                "LOCAL_IMAGE_NAME": "",
                "HF_TOKEN": "",
            }
        ) as module:
            self.assertEqual(module.API_BASE_URL, "https://router.huggingface.co/v1")
            self.assertEqual(module.MODEL_NAME, "Qwen/Qwen2.5-72B-Instruct")
            self.assertEqual(module.LOCAL_IMAGE_NAME, "")

    def test_create_env_uses_space_by_default(self):
        with load_inference_with_env({"LOCAL_IMAGE_NAME": ""}) as module:
            mock_env = AsyncMock()
            mock_env.connect = AsyncMock(return_value=mock_env)

            with patch.object(module, "SanskritEnv") as mock_client_class:
                mock_client_class.return_value = mock_env

                env = asyncio.run(module.create_env())

                mock_client_class.assert_called_once_with(base_url=module.SPACE_BASE_URL)
                mock_env.connect.assert_awaited_once()
                self.assertIs(env, mock_env)

    def test_create_env_uses_local_image_when_configured(self):
        with load_inference_with_env({"LOCAL_IMAGE_NAME": "sanskrit-env:local"}) as module:
            created_env = object()

            with patch.object(
                module.SanskritEnv,
                "from_docker_image",
                new=AsyncMock(return_value=created_env),
            ) as mock_from_docker_image:
                env = asyncio.run(module.create_env())

                mock_from_docker_image.assert_awaited_once_with("sanskrit-env:local")
                self.assertIs(env, created_env)

    def test_task_plan_has_configured_episodes_for_requested_task(self):
        with load_inference_with_env({}) as module:
            task_plan = module.build_task_plan("glossary_anchoring")

            self.assertEqual(len(task_plan), module.EPISODES_PER_TASK)
            self.assertTrue(all(task_id == "glossary_anchoring" for task_id in task_plan))

    def test_inference_does_not_expose_report_score_remap(self):
        with load_inference_with_env({}) as module:
            self.assertFalse(hasattr(module, "remap_report_score"))

    def test_task_label_uses_human_readable_name_and_difficulty(self):
        with load_inference_with_env({}) as module:
            self.assertEqual(module.build_task_label("glossary_anchoring"), "glossary anchoring (easy)")
            self.assertEqual(module.build_task_label("samasa_classification"), "samasa classification (medium)")


if __name__ == "__main__":
    unittest.main()
