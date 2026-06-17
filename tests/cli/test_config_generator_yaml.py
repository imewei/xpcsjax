"""Regression tests for config_generator scalar substitution (audit C2).

`yaml.dump(scalar, default_flow_style=True)` appends a `\n...` document-end
marker; the old `.strip()` left the `...` embedded mid-line, producing
unparseable YAML whenever an override flag was used.
"""

from __future__ import annotations

import yaml

from xpcsjax.cli.config_generator import generate_config


def test_overrides_produce_parseable_yaml(tmp_path):
    out = tmp_path / "cfg.yaml"
    generate_config(
        "two_component",
        out,
        data_path="/data/x.hdf",
        q=0.05,
        dt=0.2,
        time_length=500,
    )
    content = out.read_text(encoding="utf-8")

    # No embedded YAML document-end marker spliced mid-template.
    assert "\n..." not in content

    # Re-parses cleanly as a single document.
    loaded = yaml.safe_load(content)
    ap = loaded["analyzer_parameters"]
    assert ap["scattering"]["wavevector_q"] == 0.05
    assert ap["dt"] == 0.2
    assert ap["end_frame"] == 500
    assert loaded["experimental_data"]["file_path"] == "/data/x.hdf"
