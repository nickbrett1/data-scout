"""The frame schema: a payload identity that does not drift on VCS scratch.

The defect these tests pin: ``frame.md`` used to record one figure over the
whole checkout, so DVC rewriting ``.dvc/tmp/lock`` (9 -> 8 bytes, 3 h 44 m
after the frame was measured) moved the frame's identity while the payload was
provably untouched. The acceptance test below writes exactly that scratch file
and asserts the identity is unchanged — and, as a control, that the rule being
replaced *does* move.

It fails on the pre-fix code because there is no ``data_scout.frame`` and no
payload identity to be stable: the only figure such a frame had was the
checkout's, and that is the number the control shows drifting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data_scout import frame, workspace

DATASET_ID = "demo-dataset"

V1_FRAME = f"""---
dataset_id: {DATASET_ID}
unit: files
tier_ceiling: 1
created: 2026-10-26
---

# {DATASET_ID}

A phase-1 record: one whole-tree figure, no schema version.
"""


def _naive_v1_identity(root: Path) -> dict[str, int | str]:
    """The rule this module replaces: every file under the root, no exclusions.

    Reproduced here deliberately, so the test shows *what* moves rather than
    asserting that something does.
    """
    records = sorted(
        (path.relative_to(root).as_posix(), path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    digest = hashlib.sha256()
    for relative, size in records:
        digest.update(f"{relative}\t{size}\n".encode())
    return {
        "files": len(records),
        "bytes": sum(size for _relative, size in records),
        "manifest": digest.hexdigest(),
    }


@pytest.fixture
def dataset_with_payload(roots):
    """A dataset with a real payload and DVC's scratch directory."""
    root = roots.dataset()
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir()
    (root / "data" / "book.pdf").write_bytes(b"%PDF-1.7 payload")
    (root / "README.md").write_text("checkout plumbing\n", encoding="utf-8")
    (root / ".dvc" / "tmp").mkdir(parents=True)
    (root / ".dvc" / "config").write_text("[core]\n", encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# 1. the acceptance test: a DVC lockfile rewrite must not move the identity
# --------------------------------------------------------------------------


def test_the_frame_identity_is_unchanged_when_dvc_rewrites_its_lockfile(
    dataset_with_payload,
):
    root = dataset_with_payload
    lock = root / ".dvc" / "tmp" / "lock"
    lock.write_text("123456789", encoding="utf-8")  # 9 bytes, as measured

    before = frame.measure_frame(DATASET_ID, root=root)
    naive_before = _naive_v1_identity(root)

    # DVC rewrote its own lockfile: 9 -> 8 bytes, mtime well after the frame.
    lock.write_text("12345678", encoding="utf-8")

    after = frame.measure_frame(DATASET_ID, root=root)
    naive_after = _naive_v1_identity(root)

    # The claim about the data is unchanged, manifest included.
    assert after.identity == before.identity
    assert after.identity.manifest == before.identity.manifest
    assert after.identity.files == 1
    assert after.identity.bytes == len(b"%PDF-1.7 payload")

    # The checkout figure is stable too, because it was measured under the same
    # declared exclusions — it is not silently a different rule. (The tally of
    # what was excluded moves with the lockfile; it is a diagnostic, not part of
    # the identity claim.)
    assert after.checkout.manifest == before.checkout.manifest
    assert after.checkout.files == before.checkout.files
    assert after.checkout.bytes == before.checkout.bytes

    # The exclusion is named on the frame, and the excluded scratch is counted.
    assert set(after.exclusions) == {"git-internals", "dvc-internals"}
    assert after.checkout.excluded_files == 2  # .dvc/config + .dvc/tmp/lock

    # Control: the whole-tree rule this replaces *does* move with the lockfile.
    assert naive_after["bytes"] == naive_before["bytes"] - 1
    assert naive_after["manifest"] != naive_before["manifest"]


def test_the_exclusion_set_is_not_a_caller_option(dataset_with_payload):
    """A per-call exclusion rule is the defect, not the fix."""
    with pytest.raises(TypeError):
        frame.measure_frame(DATASET_ID, root=dataset_with_payload, exclusions=())


def test_the_schema_version_is_emitted_by_the_measurer(dataset_with_payload):
    measurement = frame.measure_frame(DATASET_ID, root=dataset_with_payload)
    assert measurement.schema_version == frame.FRAME_SCHEMA_VERSION == 2
    # And it travels onto the ledger line without the caller supplying it.
    record = frame.identity_finding(
        measurement, dataset_id=DATASET_ID, session=1, promoted=False, retire_reason="test"
    )
    assert record["tool_version"] == "2"
    assert record["params"]["frame_schema"] == 2


def test_the_written_frame_records_what_was_included_and_excluded(dataset_with_payload):
    measurement = frame.measure_frame(DATASET_ID, root=dataset_with_payload)
    path = frame.write_frame(DATASET_ID, measurement, lenses=["text", "structure"])
    text = path.read_text(encoding="utf-8")

    assert "frame_schema: 2" in text
    assert "unit: files" in text
    assert "  - git-internals" in text
    assert "  - dvc-internals" in text
    assert f"identity_files: {measurement.identity.files}" in text
    assert measurement.identity.manifest in text

    read_back = workspace.read_frame(DATASET_ID)
    assert read_back.schema_version == 2
    assert read_back.exclusions == ("git-internals", "dvc-internals")
    assert read_back.identity.scope == measurement.identity.scope
    assert read_back.identity.files == measurement.identity.files
    assert read_back.identity.bytes == measurement.identity.bytes
    assert read_back.identity.manifest == measurement.identity.manifest
    assert read_back.checkout.manifest == measurement.checkout.manifest


def test_writing_a_frame_will_not_replace_an_existing_record(dataset_with_payload):
    measurement = frame.measure_frame(DATASET_ID, root=dataset_with_payload)
    frame.write_frame(DATASET_ID, measurement)
    with pytest.raises(frame.FrameMeasureError, match="already exists"):
        frame.write_frame(DATASET_ID, measurement)


def test_replacing_a_frame_requires_a_reason_and_records_it(dataset_with_payload):
    """``overwrite=True`` alone destroys the old text with no trail."""
    measurement = frame.measure_frame(DATASET_ID, root=dataset_with_payload)
    frame.write_frame(DATASET_ID, measurement)

    with pytest.raises(frame.FrameMeasureError, match="overwrite_reason"):
        frame.write_frame(DATASET_ID, measurement, overwrite=True)
    with pytest.raises(frame.FrameMeasureError, match="non-empty overwrite_reason"):
        frame.write_frame(DATASET_ID, measurement, overwrite=True, overwrite_reason="   ")
    with pytest.raises(frame.FrameMeasureError, match="one line"):
        frame.write_frame(
            DATASET_ID, measurement, overwrite=True, overwrite_reason="v2\ninjected: field"
        )

    reason = "re-measured under v2: the v1 figure included .dvc scratch"
    path = frame.write_frame(
        DATASET_ID, measurement, overwrite=True, overwrite_reason=reason
    )
    text = path.read_text(encoding="utf-8")
    assert f"supersedes_reason: {reason}" in text
    assert workspace.read_frame(DATASET_ID).extra["supersedes_reason"] == reason


def test_an_overwrite_reason_without_an_overwrite_is_refused(dataset_with_payload):
    """A reason on a fresh frame would document a destruction that did not happen."""
    measurement = frame.measure_frame(DATASET_ID, root=dataset_with_payload)
    with pytest.raises(frame.FrameMeasureError, match="without overwrite=True"):
        frame.write_frame(
            DATASET_ID, measurement, overwrite_reason="nothing was replaced"
        )


# --------------------------------------------------------------------------
# 2. v1 and v2 are not comparable — no delta, a reason
# --------------------------------------------------------------------------


def _write_frame_text(text: str) -> None:
    path = workspace.frame_path(DATASET_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_v1_frame_and_a_v2_frame_are_not_comparable(dataset_with_payload):
    _write_frame_text(V1_FRAME)
    v1 = workspace.read_frame(DATASET_ID)
    assert v1.schema_version == 1
    assert v1.exclusions == ()

    measurement = frame.measure_frame(DATASET_ID, root=dataset_with_payload)
    _write_frame_text(frame.render_frame(measurement))
    v2 = workspace.read_frame(DATASET_ID)
    assert v2.schema_version == 2

    result = frame.compare_frames(v1, v2)
    assert result.comparable is False
    assert result.delta is None
    assert "different frame rules" in result.reason
    assert "v1" in result.reason and "v2" in result.reason


def test_two_v2_frames_are_comparable_and_report_only_the_identity_delta(
    dataset_with_payload,
):
    root = dataset_with_payload
    first = frame.measure_frame(DATASET_ID, root=root)
    (root / "data" / "extra.txt").write_text("x", encoding="utf-8")
    second = frame.measure_frame(DATASET_ID, root=root)

    _write_frame_text(frame.render_frame(first))
    left = workspace.read_frame(DATASET_ID)
    _write_frame_text(frame.render_frame(second))
    right = workspace.read_frame(DATASET_ID)

    result = frame.compare_frames(left, right)
    assert result.comparable is True
    assert result.reason is None
    assert result.delta["bytes"] == 1
    assert result.delta["files"] == 1
    assert result.delta["manifest_same"] is False


def test_a_different_exclusion_set_is_not_comparable(
    dataset_with_payload, monkeypatch
):
    """Same schema version but a different rule is still not a delta."""
    first = frame.measure_frame(DATASET_ID, root=dataset_with_payload)
    _write_frame_text(frame.render_frame(first))

    monkeypatch.setattr(frame, "FRAME_EXCLUSIONS", frame.FRAME_EXCLUSIONS[:1])
    monkeypatch.setattr(frame, "EXCLUSION_NAMES", ("git-internals",))
    second = frame.measure_frame(DATASET_ID, root=dataset_with_payload)
    assert second.exclusions == ("git-internals",)

    _write_frame_text(frame.render_frame(first))
    left = workspace.read_frame(DATASET_ID)
    _write_frame_text(frame.render_frame(second))
    right = workspace.read_frame(DATASET_ID)

    result = frame.compare_frames(left, right)
    assert result.comparable is False
    assert result.delta is None
    assert "different exclusions" in result.reason


def test_a_frame_from_an_unknown_schema_fails_loudly(dataset_workspace):
    _write_frame_text(
        V1_FRAME.replace("unit: files", "frame_schema: 99\nunit: files")
    )
    with pytest.raises(frame.FrameSchemaError, match="schema v99"):
        workspace.read_frame(DATASET_ID)


# --------------------------------------------------------------------------
# 3. the ledger-side comparability check the render gate uses
# --------------------------------------------------------------------------


def _frame_line(schema: int, size: int) -> str:
    return json.dumps(
        {
            "tool": "frame",
            "value": size,
            "n": 8,
            "params": {"frame_schema": schema},
        }
    )


def test_frame_lines_under_different_schemas_are_reported_as_incomparable():
    problems = frame.comparability_problems(
        [_frame_line(1, 934_881_648)], [_frame_line(2, 934_843_620)]
    )
    assert len(problems) == 1
    assert "not comparable" in problems[0]
    assert "different frame rules" in problems[0]


def test_frame_lines_under_the_same_schema_claim_no_incomparability():
    assert frame.comparability_problems([_frame_line(2, 1)], [_frame_line(2, 1)]) == []


def test_a_ledger_without_a_frame_line_makes_no_comparability_claim():
    assert frame.comparability_problems(['{"tool": "file_census"}'], []) == []


def test_ledger_frame_schemas_reports_the_set_with_history_visible():
    """A v1 line beside a v2 line is the normal append-only outcome, not an error."""
    lines = [
        _frame_line(1, 934_881_648),
        _frame_line(2, 934_843_620),
        '{"tool": "file_census"}',
        # No params.frame_schema is v1, the phase-1 shape.
        json.dumps({"tool": "frame", "value": 1, "n": 1}),
        # Unreadable version is unknown (-1), not dropped and not v1.
        json.dumps({"tool": "frame", "value": 1, "n": 1, "params": {"frame_schema": "?"}}),
    ]
    assert frame.ledger_frame_schemas(lines) == [-1, 1, 2]


def test_ledger_frame_schemas_is_empty_when_no_frame_line_is_present():
    """Empty is "no line", which is not "schema zero"."""
    assert frame.ledger_frame_schemas(['{"tool": "file_census"}']) == []
