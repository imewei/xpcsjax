"""Guards HEAVY_NODES parity between the Makefile and .github/workflows/ci.yml.

Makefile's HEAVY_NODES list is the single source of truth for which
individually-flaky/heavy test nodes get deselected from the parallel pass and
run serially instead. ci.yml has no equivalent single source -- it repeats the
same three node ids literally across four run blocks (test-ubuntu Test/Retry,
test-other Test/Retry), each pair (parallel --deselect + serial run) relying
on a "MUST mirror Makefile's HEAVY_NODES exactly" comment rather than any
enforced check. A drift here is silent: a node dropped from one copy but not
another quietly reintroduces it to the parallel xdist pass on that one path,
undoing the OOM/flake mitigation the deselect exists for -- unlike a shard
count mismatch (which pytest-split rejects loudly), nothing here errors on
its own.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAKEFILE = ROOT / "Makefile"
CI_YML = ROOT / ".github" / "workflows" / "ci.yml"

# One run block (parallel pass + serial pass) per Test/Retry step; ci.yml
# currently has four such blocks (test-ubuntu Test, test-ubuntu Retry,
# test-other Test, test-other Retry), each referencing every heavy node
# twice (once in --deselect, once as a bare serial-run argument).
EXPECTED_OCCURRENCES_PER_NODE = 8


def _makefile_heavy_nodes() -> set[str]:
    text = MAKEFILE.read_text()
    match = re.search(r"HEAVY_NODES := \\\n(.*?)\nHEAVY_FILES", text, re.DOTALL)
    assert match, "Makefile's `HEAVY_NODES := \\` block not found -- did its format change?"
    nodes = {
        line.strip().rstrip("\\").strip() for line in match.group(1).splitlines() if line.strip()
    }
    assert nodes, "Parsed zero HEAVY_NODES from the Makefile -- parsing regex is likely broken."
    return nodes


def _ci_yml_node_occurrences() -> list[str]:
    text = CI_YML.read_text()
    return re.findall(r'"([^"]*::[^"]*)"', text)


def test_ci_yml_deselects_exactly_the_makefile_heavy_nodes():
    makefile_nodes = _makefile_heavy_nodes()
    ci_nodes = set(_ci_yml_node_occurrences())
    assert ci_nodes == makefile_nodes, (
        "ci.yml's quoted test-node ids don't match Makefile's HEAVY_NODES.\n"
        f"  Only in Makefile: {makefile_nodes - ci_nodes}\n"
        f"  Only in ci.yml:   {ci_nodes - makefile_nodes}\n"
        "Update both together -- see ci.yml's 'MUST mirror Makefile's "
        "HEAVY_NODES exactly' comments."
    )


def test_ci_yml_references_each_heavy_node_symmetrically():
    makefile_nodes = _makefile_heavy_nodes()
    occurrences = _ci_yml_node_occurrences()
    counts = {node: occurrences.count(node) for node in makefile_nodes}
    assert all(count == EXPECTED_OCCURRENCES_PER_NODE for count in counts.values()), (
        f"Expected every HEAVY_NODES entry to appear exactly "
        f"{EXPECTED_OCCURRENCES_PER_NODE} times in ci.yml (once per --deselect "
        "and once per serial-run reference, across all four Test/Retry run "
        f"blocks); got: {counts}. An asymmetric count means one of ci.yml's "
        "four run blocks is missing a node in either its --deselect list or "
        "its serial pass."
    )
