"""Static checks over the whole service.

Deliberately AST-only: it never imports the package. torch and transformers
are not installed on a development machine here, and installing them to run a
syntax check is not a trade worth making.

What it checks:
  - every file parses
  - no module-level import of torch, transformers or PIL, which would make
    the service unimportable without the heavy stack and would drag model
    loading into process start
  - no local re-import shadows a module-level name (the bug class that has
    cost this codebase several UnboundLocalErrors in production)
  - the payload contract still matches the chatbot's
"""
from __future__ import annotations

import ast
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEAVY = {"torch", "transformers", "PIL", "numpy"}

# Keys the chatbot's search reads off a point. Losing one here means the bot
# still finds the product and can no longer say what colour it is.
REQUIRED_PAYLOAD_KEYS = {
    "source", "ai_tenant_id", "page_id", "sku", "product_sku", "product_id",
    "name", "title", "description", "image_url", "price", "currency",
    "type", "in_stock", "qty", "updated_at", "url", "raw_product",
}


def python_files():
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in {"__pycache__", ".git", ".venv", "venv", "node_modules"}]
        for name in sorted(names):
            if name.endswith(".py"):
                yield os.path.join(base, name)


def rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def check_heavy_imports(tree, path, problems):
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split(".")[0]]
        for name in names:
            if name in HEAVY:
                problems.append(
                    f"{rel(path)}:{node.lineno}: module-level import of {name!r} -- "
                    f"move it inside the function that needs it"
                )


def check_shadowed_imports(tree, path, problems):
    """A local `import X` binds X for the whole function.

    Using the module-level X earlier in that same function is then an
    UnboundLocalError, thrown only on the path that reaches it.
    """
    module_names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_names.update(a.asname or a.name for a in node.names)

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        local_imports = {}
        for node in ast.walk(func):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if node is func:
                    continue
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    if bound in module_names:
                        local_imports.setdefault(bound, node.lineno)

        for name, import_line in local_imports.items():
            for node in ast.walk(func):
                if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        and node.id == name and node.lineno < import_line):
                    problems.append(
                        f"{rel(path)}:{node.lineno}: {name!r} is used before the "
                        f"local import on line {import_line} shadows it"
                    )
                    break


def check_payload_contract(problems):
    """The payload keys the mirrored indexer writes."""
    path = os.path.join(ROOT, "imagesearch", "indexer.py")
    tree = ast.parse(io.open(path, encoding="utf-8").read(), path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_product_payload":
            keys = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    keys.update(k.value for k in sub.keys
                                if isinstance(k, ast.Constant) and isinstance(k.value, str))
            missing = REQUIRED_PAYLOAD_KEYS - keys
            if missing:
                problems.append(
                    f"{rel(path)}: build_product_payload no longer writes "
                    f"{sorted(missing)} -- it has drifted from the chatbot's"
                )
            return

    problems.append(f"{rel(path)}: build_product_payload is gone")


def check_vector_size(problems):
    """Both files that name the dimension have to agree, and with the
    collections already in Qdrant."""
    sizes = {}
    for name in ("imagesearch/config.py", "imagesearch/indexer.py"):
        path = os.path.join(ROOT, name)
        tree = ast.parse(io.open(path, encoding="utf-8").read(), path)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == "VECTOR_SIZE" for t in node.targets)):
                if isinstance(node.value, ast.Constant):
                    sizes[name] = int(node.value.value)
                elif isinstance(node.value, ast.Call):
                    # int(os.environ.get("VECTOR_SIZE", "768")) -- the
                    # default is the last argument of the inner call.
                    for arg in node.value.args:
                        if isinstance(arg, ast.Call) and arg.args:
                            last = arg.args[-1]
                            if isinstance(last, ast.Constant):
                                sizes[name] = int(last.value)

    if len(set(sizes.values())) > 1:
        problems.append(f"VECTOR_SIZE disagrees between files: {sizes}")
    elif set(sizes.values()) != {768}:
        problems.append(
            f"VECTOR_SIZE is {sizes} -- the shared collections are 768-dimensional"
        )


def main() -> int:
    problems = []
    files = list(python_files())

    for path in files:
        source = io.open(path, encoding="utf-8").read()
        try:
            tree = ast.parse(source, path)
        except SyntaxError as e:
            problems.append(f"{rel(path)}:{e.lineno}: {e.msg}")
            continue

        if rel(path).startswith("tools/"):
            continue
        check_heavy_imports(tree, path, problems)
        check_shadowed_imports(tree, path, problems)

    check_payload_contract(problems)
    check_vector_size(problems)

    print(f"scanned {len(files)} python files under {ROOT}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  " + p)
        return 1

    print("OK: all files parse, no heavy imports at module level, "
          "no shadowed imports, payload contract intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
