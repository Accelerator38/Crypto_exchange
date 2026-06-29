"""Entrypoint safety tests for Windows process spawning."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]


def _parse_script(name: str) -> ast.Module:
    return ast.parse((ROOT / name).read_text(encoding="utf-8"))


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if len(test.comparators) != 1:
            continue
        rhs = test.comparators[0]
        if isinstance(rhs, ast.Constant) and rhs.value == "__main__":
            return True
    return False


def _has_top_level_start_call(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Expr) and _contains_start_production(node):
            return True
    return False


def _contains_start_production(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name) and func.id == "start_production":
                return True
    return False


class TestEntrypoints(unittest.TestCase):
    def test_unified_entrypoint_defers_launch_to_main_guard(self):
        tree = _parse_script("Start_panteon.py")
        self.assertTrue(
            any(isinstance(node, ast.FunctionDef) and node.name == "main"
                for node in tree.body)
        )
        self.assertTrue(_has_main_guard(tree))
        self.assertFalse(_has_top_level_start_call(tree))

    def test_runtime_project_root_uses_unified_launcher_marker(self):
        path = ROOT / "src" / "panteon_runtime" / "project_paths.py"
        spec = importlib.util.spec_from_file_location("project_paths_under_test", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with patch.object(module.Path, "cwd", return_value=ROOT.parent):
            found = module._find_project_root(path)

        self.assertEqual(found, ROOT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
