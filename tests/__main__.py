import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.test_attention import test_attention_cache_smoke
from tests.test_engine import test_engine_smoke
from tests.test_generate import test_model_runner_via_engine, test_static_batch_matches_single
from tests.test_kv_cache import test_kv_cache_smoke
from tests.test_backends import test_load_gpt_backend
from tests.test_scheduler import test_scheduler_smoke
from tests.test_server import test_completions_smoke, test_completions_validation, test_health


class SmokeTests(unittest.TestCase):
    def test_kv_cache(self):
        test_kv_cache_smoke()

    def test_attention(self):
        test_attention_cache_smoke()

    def test_scheduler(self):
        test_scheduler_smoke()

    def test_engine(self):
        test_engine_smoke()

    def test_static_batch(self):
        test_static_batch_matches_single()

    def test_model_runner(self):
        test_model_runner_via_engine()

    def test_backends(self):
        test_load_gpt_backend()

    def test_server_health(self):
        test_health()

    def test_server_validation(self):
        test_completions_validation()

    def test_server_completions(self):
        test_completions_smoke()


if __name__ == "__main__":
    unittest.main()
