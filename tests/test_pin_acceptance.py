"""§5's six acceptance tests for the dataset pin, numbered as the order is.

Each one is here because it would catch a real regression, not to restate a
happy path. The pin read/write, the v3 frame and the ledger field are one
feature; these tests are the contract that they stay one.

(``tests/test_readonly.py`` already covers ``pin.json`` under the read-only
root, and ``tests/test_registry.py`` pins the belt at 15. Neither is repeated
here.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_scout import frame, workspace
from data_scout.tools import findings_append

DATASET_ID = "demo-dataset"

PIN = {
    "pin_schema": 1,
    "dataset_id": DATASET_ID,
    "dataset_commit": "84ba1903af58166a9e92dc7687e6893863e732e9",
    "dataset_manifest_hash": "cab6e809cb982eba178a0f3a631e16d6.dir",
    "dataset_files": 2,
    "dataset_bytes": 15,
    "tag": DATASET_ID,
    "dvc_version": "3.67.1",
    "written_at": "2026-11-01T11:53:57Z",
}


def _payload(root: Path) -> Path:
    """A two-file payload under ``data/``, 15 bytes total."""
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "a.txt").write_bytes(b"12345")
    (root / "data" / "b.txt").write_bytes(b"1234567890")
    return root


def _write_pin(root: Path, **overrides) -> Path:
    data = dict(PIN)
    data.update(overrides)
    path = root / workspace.PIN_FILENAME
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# 1. absent pin --------------------------------------------------------------


def test_1_absent_pin_is_unpinned_not_an_error(roots):
    """No pin.json: false + nulls in the frame *and* the ledger params, no raise."""
    root = _payload(roots.dataset())

    measurement = frame.measure_frame(DATASET_ID, root=root)
    assert measurement.pin is None
    assert measurement.identity.files == 2

    rendered = frame.render_frame(measurement)
    assert "dataset_pinned: false" in rendered
    for key in ("dataset_commit", "dataset_manifest_hash", "dataset_files", "dataset_bytes"):
        assert f"{key}: null" in rendered

    params = frame.identity_finding(
        measurement, dataset_id=DATASET_ID, session=1, promoted=False,
        retire_reason="one-off: test",
    )["params"]
    assert params["dataset_pinned"] is False
    assert params["dataset_commit"] is None
    assert params["dataset_manifest_hash"] is None
    assert params["dataset_files"] is None
    assert params["dataset_bytes"] is None


# 2. round trip --------------------------------------------------------------


def test_2_the_pin_round_trips_frame_frontmatter_and_ledger_params(roots):
    """``read_frame`` must return the pin — this catches a missing `known` key,
    which would silently drop it into ``Frame.extra``."""
    root = _payload(roots.dataset())
    _write_pin(root)

    measurement = frame.measure_frame(DATASET_ID, root=root)
    assert measurement.pin is not None
    assert measurement.pin.dataset_commit == PIN["dataset_commit"]

    path = workspace.frame_path(DATASET_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frame.render_frame(measurement), encoding="utf-8")

    read_back = workspace.read_frame(DATASET_ID)
    assert read_back.pin is not None
    assert read_back.pin.pinned is True
    assert read_back.pin.dataset_commit == PIN["dataset_commit"]
    assert read_back.pin.dataset_manifest_hash == PIN["dataset_manifest_hash"]
    assert read_back.pin.dataset_files == PIN["dataset_files"]
    assert read_back.pin.dataset_bytes == PIN["dataset_bytes"]

    params = frame.identity_finding(
        measurement, dataset_id=DATASET_ID, session=1, promoted=False,
        retire_reason="one-off: test",
    )["params"]
    assert params["dataset_commit"] == PIN["dataset_commit"]
    assert params["dataset_manifest_hash"] == PIN["dataset_manifest_hash"]
    assert params["dataset_files"] == PIN["dataset_files"]
    assert params["dataset_bytes"] == PIN["dataset_bytes"]


# 3. an old line still validates --------------------------------------------


def test_3_a_line_from_before_the_field_re_serializes_with_an_explicit_null():
    """Absent ≠ null. ``to_json`` *adds* the null; that asymmetry is documented
    behaviour, asserted here so nobody 'fixes' it into a migration."""
    from data_scout.ledger import Finding

    old = {
        "dataset_id": DATASET_ID,
        "session": 1,
        "unit": "file",
        "tool": "file_census",
        "tool_version": "1",
        "params": {"path": "."},
        "value": 0.9,
        "n": 2,
        "promoted": False,
        "retire_reason": "one-off: written before the pin field existed",
    }
    raw = json.dumps(old, sort_keys=True)
    assert "dataset_commit" not in raw

    finding = Finding.from_json(raw)
    assert finding.dataset_commit is None
    assert json.loads(finding.to_json())["dataset_commit"] is None


# 4. a corrupt pin fails loudly ---------------------------------------------


@pytest.mark.parametrize(
    "contents",
    [
        "{not json",
        json.dumps(dict(PIN, pin_schema=99)),
        json.dumps(dict(PIN, dataset_id="some-other-dataset")),
    ],
    ids=["malformed-json", "unknown-schema", "dataset-id-mismatch"],
)
def test_4_a_corrupt_pin_raises_and_never_degrades_to_null(roots, contents):
    root = _payload(roots.dataset())
    pin = root / workspace.PIN_FILENAME
    pin.write_text(contents, encoding="utf-8")

    with pytest.raises(workspace.PinUnreadable) as exc:
        workspace.read_pin(DATASET_ID)
    assert str(pin) in str(exc.value)

    # findings_append propagates: it must not write a plausible-looking null.
    record = {
        "unit": "file",
        "tool": "file_census",
        "tool_version": "1",
        "params": {"path": "."},
        "value": 0.9,
        "n": 2,
        "promoted": False,
        "retire_reason": "one-off: the pin is corrupt",
    }
    with pytest.raises(workspace.PinUnreadable):
        findings_append(record)
    assert not workspace.ledger_path(DATASET_ID).exists()


# 5. a stale pin -------------------------------------------------------------


def test_5_a_pin_that_disagrees_with_the_measurement_fails_before_writing(roots):
    root = _payload(roots.dataset())
    _write_pin(root, dataset_files=8, dataset_bytes=934_843_620)

    with pytest.raises(frame.PinStale) as exc:
        frame.measure_frame(DATASET_ID, root=root)

    message = str(exc.value)
    assert "8 files / 934843620 B" in message
    assert "2 files / 15 B" in message
    assert not workspace.frame_path(DATASET_ID).exists(), "no frame may be written"


# 6. findings_append fills the pin ------------------------------------------


def test_6_findings_append_fills_the_pin_and_a_supplied_commit_wins(roots):
    root = _payload(roots.dataset())
    _write_pin(root)

    base = {
        "unit": "file",
        "tool": "file_census",
        "tool_version": "1",
        "params": {"path": "."},
        "value": 0.9,
        "n": 2,
        "promoted": False,
        "retire_reason": "one-off: test",
    }
    findings_append(base)
    first = json.loads(workspace.ledger_path(DATASET_ID).read_text(encoding="utf-8").splitlines()[0])
    assert first["dataset_commit"] == PIN["dataset_commit"]

    own = "f" * 40
    findings_append(base | {"dataset_commit": own})
    second = json.loads(workspace.ledger_path(DATASET_ID).read_text(encoding="utf-8").splitlines()[1])
    assert second["dataset_commit"] == own
