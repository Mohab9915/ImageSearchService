"""Has the mirrored indexer drifted from the chatbot's?

Both services write into the same Qdrant collections, so the functions that
decide a point's id, its payload and which pictures get indexed have to stay
byte-identical. This diffs them against the chatbot's copy.

It is skipped, not failed, when the chatbot repository is not beside this one
-- on a build box it will not be. Run it locally before deploying a change to
either side.

    python tools/test_mirror_in_step.py [path/to/SocialMediaChatbot]
"""
from __future__ import annotations

import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHATBOT = os.path.join(os.path.dirname(HERE), "SocialMediaChatbot")

# Change one of these on either side and the two services stop agreeing about
# what is in a collection.
PINNED = [
    "build_product_payload",
    "_point_id",
    "_product_sku",
    "product_images",
    "colour_images",
    "_extract_price",
    "_extract_stock",
    "translated",
    "ensure_collection",
    "tenant_product_collection_name",
]

FAILURES = []


def functions(path):
    tree = ast.parse(io.open(path, encoding="utf-8").read(), path)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.dump(node, include_attributes=False)
    return out


def main() -> int:
    chatbot_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHATBOT
    theirs_path = os.path.join(chatbot_root, "helpers", "product_image_indexer.py")
    ours_path = os.path.join(HERE, "imagesearch", "indexer.py")

    if not os.path.exists(theirs_path):
        print(f"SKIP: no chatbot checkout at {chatbot_root}")
        print("      pass its path as an argument to compare")
        return 0

    theirs = functions(theirs_path)
    ours = functions(ours_path)

    print(f"comparing {len(PINNED)} pinned functions")
    for name in PINNED:
        if name not in theirs:
            print(f"  FAIL  {name}: gone from the chatbot's indexer")
            FAILURES.append(name)
        elif name not in ours:
            print(f"  FAIL  {name}: missing from the mirror")
            FAILURES.append(name)
        elif theirs[name] != ours[name]:
            print(f"  FAIL  {name}: the two copies differ")
            FAILURES.append(name)
        else:
            print(f"  PASS  {name}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} function(s) have drifted: {FAILURES}")
        print("Re-copy helpers/product_image_indexer.py and repoint its two imports.")
        return 1
    print("ALL PASS -- the mirror is in step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
