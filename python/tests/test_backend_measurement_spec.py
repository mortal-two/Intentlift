import json

import pytest

from intentir.backend.measurement_spec import (
    DEFAULT_MEASUREMENT_SPEC,
    MEASUREMENT_PROTOCOL,
    load_measurement_spec,
)


def test_measurement_spec_is_complete():
    spec = load_measurement_spec()
    assert spec["protocol"] == MEASUREMENT_PROTOCOL
    assert set(spec["data_regimes"]) == {"warm", "streaming", "cold-data"}
    assert spec["comparison_rule"]["required_equal_fields"]


def test_measurement_spec_rejects_unknown_root_field(tmp_path):
    value = load_measurement_spec()
    value["unexpected"] = True
    path = tmp_path / "measurement.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unknown root fields"):
        load_measurement_spec(path)


def test_default_measurement_spec_path_exists():
    assert DEFAULT_MEASUREMENT_SPEC.is_file()
