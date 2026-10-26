"""``findings.jsonl`` is append-only and every line is decided. Tested, not advised."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from data_scout.ledger import Finding, Ledger, diff_lines, normalized_line
from data_scout.tools import findings_append


def record(**overrides):
    payload = {
        "corpus_id": "demo-corpus",
        "session": 1,
        "unit": "file",
        "tool": "candidate_density",
        "tool_version": "1",
        "params": {"per_chars": 1000},
        "value": 0.41,
        "n": 1293,
        "sample": None,
        "promoted": True,
        "script": "data_scout.tools.candidate_density",
    }
    payload.update(overrides)
    return payload


def test_a_decided_promoted_line_round_trips():
    finding = Finding.model_validate(record())
    line = finding.to_json()
    assert json.loads(line)["value"] == 0.41
    assert Finding.from_json(line) == finding


def test_promoted_without_script_is_refused():
    with pytest.raises(ValidationError, match="requires a script"):
        Finding.model_validate(record(script=None))


def test_unpromoted_without_a_retire_reason_is_refused():
    with pytest.raises(ValidationError, match="retire_reason"):
        Finding.model_validate(record(promoted=False, script=None))


def test_unpromoted_with_a_reason_is_a_deliberate_one_off():
    finding = Finding.model_validate(
        record(
            promoted=False,
            script=None,
            retire_reason="one-off: PDF page geometry is not corpus-agnostic",
        )
    )
    assert finding.promoted is False


def test_script_must_be_a_module_path_not_a_file_path():
    with pytest.raises(ValidationError, match="file path"):
        Finding.model_validate(record(script="src/data_scout/tools/quotes.py"))


def test_script_must_resolve_to_real_code():
    with pytest.raises(ValidationError, match="does not resolve"):
        Finding.model_validate(record(script="data_scout.tools.imaginary_tool"))


def test_a_nan_is_a_measurement_that_did_not_happen():
    with pytest.raises(ValidationError, match="not finite"):
        Finding.model_validate(record(value=float("nan")))


def test_params_must_be_json_serializable():
    with pytest.raises(ValidationError, match="JSON-serializable"):
        Finding.model_validate(record(params={"path": Path("/tmp/x")}))


def test_append_only_correction_appends(tmp_path):
    ledger = Ledger(tmp_path / "findings.jsonl")
    ledger.append(Finding.model_validate(record(value=0.41)))
    ledger.append(Finding.model_validate(record(value=0.39, params={"per_chars": 1200})))
    findings = ledger.findings()
    assert [finding.value for finding in findings] == [0.41, 0.39]
    assert len(ledger.lines()) == 2


def test_there_is_no_update_and_no_delete():
    """The absence is the design: a correction is a new line, never an edit."""
    for forbidden in ("update", "delete", "replace", "truncate", "rewrite", "edit", "pop"):
        assert not hasattr(Ledger, forbidden), forbidden


def test_verify_catches_a_hand_edited_line(tmp_path):
    path = tmp_path / "findings.jsonl"
    ledger = Ledger(path)
    ledger.append(Finding.model_validate(record()))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"corpus_id":"demo-corpus"}\n')
    problems = ledger.verify()
    assert len(problems) == 1
    assert problems[0].startswith("line 2")


def test_an_absent_ledger_is_an_empty_ledger(tmp_path):
    assert Ledger(tmp_path / "nothing.jsonl").lines() == []


def test_findings_append_fills_the_environment_stamp(corpus_workspace):
    result = findings_append(
        {
            "unit": "file",
            "tool": "file_census",
            "tool_version": "1",
            "value": 0.9,
            "n": 10,
            "promoted": False,
            "retire_reason": "one-off: the corpus was two files",
        }
    )
    assert result["value"] == 1
    line = json.loads(Path(result["ledger"]).read_text(encoding="utf-8").strip())
    assert line["corpus_id"] == "demo-corpus"
    assert line["session"] == 1
    assert line["ts"].endswith("Z")


def test_findings_append_without_a_session_fails_loudly(corpus_workspace, monkeypatch):
    monkeypatch.delenv("DATA_SCOUT_SESSION")
    payload = record()
    payload.pop("session")
    with pytest.raises(ValueError, match="session"):
        findings_append(payload)


def test_the_diff_is_modulo_ts_and_nothing_else():
    left = [record() | {"ts": "2026-10-26T00:00:00Z", "value": 0.41}]
    right = [record() | {"ts": "2026-10-27T11:11:11Z", "value": 0.41}]
    assert diff_lines(left, right) == []
    right[0]["value"] = 0.42
    assert diff_lines(left, right)


def test_a_refetched_observation_does_not_fail_the_diff():
    """The line records the re-fetch; that must not be able to fail the gate."""
    left = [record(params={"per_chars": 1000, "observation_source": "fetched"})]
    right = [record(params={"per_chars": 1000, "observation_source": "cache"})]
    assert diff_lines(left, right) == []
    assert "observation_source" not in normalized_line(left[0])
