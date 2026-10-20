"""Static checks that do not need torch: every relative import resolves to a real
module and to a name that module actually defines.

Run: python scripts/check_imports.py
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "asr"


def defined_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def main():
    modules = {p.stem: p for p in PKG.glob("*.py") if p.stem != "__init__"}
    trees = {name: ast.parse(path.read_text(encoding="utf-8")) for name, path in modules.items()}
    exported = {name: defined_names(tree) for name, tree in trees.items()}
    exported["__init__"] = set()

    problems = []
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            target = node.module or ""
            if target not in exported:
                problems.append(f"{name}.py:{node.lineno}: no module '{target}'")
                continue
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in exported[target] and root not in {"annotations"}:
                    problems.append(f"{name}.py:{node.lineno}: '{target}' has no name '{root}'")

    if problems:
        print("FAIL")
        for problem in problems:
            print(" ", problem)
        return 1
    print(f"OK - {len(modules)} modules, all relative imports resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
