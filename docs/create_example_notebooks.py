"""Stage example notebooks into the docs source tree and (re)generate examples.rst.

Notebooks live in ``LDAQ/examples/`` (outside the Sphinx ``source/`` root).
Sphinx requires sources under ``source/``, so we materialise each notebook
into ``source/examples/`` as a symlink (preferred) or a copy (fallback).
This replaces the previous ``nbsphinx_link`` ``.nblink`` shims and lets the
docs build against current ``nbsphinx`` / ``sphinx`` / ``docutils``.
"""

import glob
import json
import os
import re
import shutil

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLES_SRC = os.path.join(THIS_DIR, "..", "examples")
EXAMPLES_DST = os.path.join(THIS_DIR, "source", "examples")
EXAMPLES_RST = os.path.join(THIS_DIR, "source", "examples.rst")

RST_HEADER = """
Examples
========

.. toctree::
   :maxdepth: 3
"""


def has_title(notebook_path: str) -> bool:
    """Return True iff the notebook has a top-level ``# Heading`` markdown cell."""
    with open(notebook_path) as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        if re.search(r"^\s*#\s+\S", text, flags=re.MULTILINE):
            return True
    return False


def stage_notebook(src: str, dst: str) -> None:
    """Place a notebook at ``dst`` by symlinking ``src``; fall back to copy."""
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def main() -> None:
    print("Staging example notebooks...")
    os.makedirs(EXAMPLES_DST, exist_ok=True)

    # Clean stale staged artifacts (notebooks + legacy .nblink shims).
    for name in os.listdir(EXAMPLES_DST):
        if name.endswith((".ipynb", ".nblink")):
            os.remove(os.path.join(EXAMPLES_DST, name))

    notebooks = sorted(glob.glob(os.path.join(EXAMPLES_SRC, "*.ipynb")))
    with open(EXAMPLES_RST, "w") as f:
        f.write(RST_HEADER)
        for nb in notebooks:
            stem = os.path.splitext(os.path.basename(nb))[0]
            if not has_title(nb):
                # nbsphinx needs a title for the TOC entry; skip silently.
                print(f"  skipped {stem}.ipynb (no title cell)")
                continue
            stage_notebook(nb, os.path.join(EXAMPLES_DST, stem + ".ipynb"))
            f.write("\n   examples/" + stem)
            print(f"  staged {stem}.ipynb")


if __name__ == "__main__":
    main()
