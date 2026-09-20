"""Verify paid GPU resource ceilings through the installed SDK, without cloud calls."""

import ast
import unittest
from pathlib import Path

from modal._resources import convert_fn_config_to_resources_config


ROOT = Path(__file__).resolve().parents[1]


class ModalResourceCapsTests(unittest.TestCase):
    def test_gpu_decorators_serialize_request_and_limit_without_changing_timeouts(self):
        expected = {
            "modal_clean_video.py": (32768, 3600, 1),
            "modal_motion.py": (32768, 1500, 1),
            "modal_multiperson.py": (65536, 3600, 2),
            "modal_lhm.py": (65536, 1800, 1),
        }
        for filename, (memory, timeout, count) in expected.items():
            tree = ast.parse((ROOT / "worker" / filename).read_text())
            found = 0
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    kwargs = {item.arg: item.value for item in decorator.keywords}
                    if "gpu" not in kwargs:
                        continue
                    found += 1
                    with self.subTest(worker=filename, function=node.name):
                        config = {
                            key: ast.literal_eval(kwargs[key]) for key in ("cpu", "memory", "gpu")
                        }
                        self.assertEqual(ast.literal_eval(kwargs["timeout"]), timeout)
                        self.assertEqual(config["cpu"], (4, 4))
                        self.assertEqual(config["memory"], (memory, memory))
                        resources = convert_fn_config_to_resources_config(**config)
                        self.assertEqual(resources.milli_cpu, 4000)
                        self.assertEqual(resources.milli_cpu_max, 4000)
                        self.assertEqual(resources.memory_mb, memory)
                        self.assertEqual(resources.memory_mb_max, memory)
            self.assertEqual(found, count, filename)


if __name__ == "__main__":
    unittest.main()
