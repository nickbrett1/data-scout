"""The MCP façade: the belt, generated from the registry, on the same port.

Design §2.4 asks for the library first and *"the MCP façade generated from the
registry afterwards"*. This is that generation step, and it is deliberately
mechanical: the tool names are ``registry.names()``, so the façade cannot drift
from the belt — a test asserts the two sets are equal, and a sixteenth tool
would fail it before it ever reached a client.

Two rules the façade inherits rather than re-decides:

* **An MCP result never becomes a record** (design §2.4). Nothing here appends to
  ``findings.jsonl`` on your behalf. ``findings_append`` is exposed because a
  *cell* (or a human) decides what a measurement means; the façade merely carries
  the call through the same validator the ledger uses.
* **Fail loudly, never infer.** A missing extra, a missing binary or an
  unconfigured gateway raises ``ToolUnavailable`` with the install command in the
  message, and the MCP error carries that text verbatim. A tool that returns
  "no clusters" because BERTopic is absent is worse than one that refuses.

Paths given to a path-taking tool must resolve under the corpora root or the
workspace root. The corpora are mounted read-only and this server never writes,
but a network-facing tool should not be a way to read ``/etc`` from a client.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from data_scout import __version__, workspace
from data_scout.tools import (
    candidate_density,
    cluster_topics,
    embed_sample,
    exif_census,
    file_census,
    findings_append,
    image_inventory,
    measurement_register,
    near_dup_docs,
    near_dup_images,
    partition_doc,
    pdf_forensics,
    sample,
    structure_mix,
    text_quality,
)
from data_scout.tools._common import ToolUnavailable

MCP_PATH = "/mcp"


def _deliberate(fn):
    """Turn a refusal into an MCP *tool error* that carries its own message.

    The SDK's default is to reduce any exception to "Error executing tool <name>"
    and keep the text on the server — correct for a crash, useless for a refusal.
    A missing extra, a missing binary or a path outside the roots is a message the
    caller must read and act on, so those are re-raised as ``ToolError`` with the
    text verbatim.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ToolUnavailable, ValidationError, ValueError, KeyError, OSError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def _guard(value: str, *, what: str = "path") -> Path:
    """Refuse a path that resolves outside both roots, naming both roots."""
    resolved = Path(value).expanduser().resolve()
    allowed = (workspace.corpora_root().resolve(), workspace.workspace_root().resolve())
    for root in allowed:
        if resolved == root or root in resolved.parents:
            return resolved
    raise ValueError(
        f"{what} {value!r} is outside both roots. The corpora root is "
        f"{allowed[0]} and the workspace root is {allowed[1]}; pass a path under one "
        "of them. This server does not read the rest of the filesystem."
    )


def build_server() -> MCPServer:
    """The belt as an MCP server. One tool per registry name, no more."""
    server = MCPServer(
        name="data-scout",
        title="data-scout",
        version=__version__,
        instructions=(
            "Instruments for profiling an unfamiliar corpus. Each tool returns "
            "{value, n, unit_notes} plus named detail fields; read the named "
            "fields rather than the prose. Nothing here writes a finding: a call "
            "measures, and a separate findings_append records what a human or a "
            "cell decided it means."
        ),
    )

    @server.tool(name="file_census")
    @_deliberate
    def file_census_tool(path: str, max_files: int | None = None) -> dict[str, Any]:
        """Mime/extension mix, size distribution and zero-byte count for a tree.

        Tier 0: shells to `file`. The headline number is the share of files in
        the largest mime bucket.
        """
        return file_census(_guard(path), max_files=max_files)

    @server.tool(name="pdf_forensics")
    @_deliberate
    def pdf_forensics_tool(path: str, max_files: int | None = None) -> dict[str, Any]:
        """Font count and text layer per PDF, via pdfinfo/pdffonts/pdfimages.

        The decisive cheap answer: `documents_without_text_layer` > 0 means the
        PDFs are scans and every text measurement downstream is fiction.
        """
        return pdf_forensics(_guard(path), max_files=max_files)

    @server.tool(name="exif_census")
    @_deliberate
    def exif_census_tool(path: str, max_files: int | None = None) -> dict[str, Any]:
        """Camera/software/orientation histograms, via exiftool.

        `photographic_share` separates photographs from scans, screenshots and
        generated art.
        """
        return exif_census(_guard(path), max_files=max_files)

    @server.tool(name="partition_doc")
    @_deliberate
    def partition_doc_tool(path: str) -> dict[str, Any]:
        """Convert one document to typed parts through the Docling sidecar.

        Network client, not a local model (design §2.4.1). Requires DOCLING_URL
        and fails loudly without it.
        """
        return partition_doc(_guard(path))

    @server.tool(name="structure_mix")
    @_deliberate
    def structure_mix_tool(parts: list[dict[str, Any]], total_parts: int | None = None) -> dict[str, Any]:
        """Narrative/picture/table shares over a partition_doc part list.

        `narrative_share` is prose; a plate book comes back near zero. Pass
        `total_parts` when the part list you were handed is a capped sample.
        """
        return structure_mix(parts, total_parts=total_parts)

    @server.tool(name="candidate_density")
    @_deliberate
    def candidate_density_tool(
        source: str,
        per_chars: int = 1000,
        min_line_chars: int = 25,
        max_examples: int = 10,
    ) -> dict[str, Any]:
        """Quotable-candidate lines per 1000 characters, and why lines were rejected.

        `source` is a file under a root, or the text itself. The filter is ported
        from reader's quotes.py, not imported from it.
        """
        return candidate_density(
            _maybe_guard(source),
            per_chars=per_chars,
            min_line_chars=min_line_chars,
            max_examples=max_examples,
        )

    @server.tool(name="text_quality")
    @_deliberate
    def text_quality_tool(source: str, include_textdescriptives: bool = True) -> dict[str, Any]:
        """Dictionary hit rate, entropy, language, TTR, MTLD and Flesch reading ease.

        Dictionary hit rate is the value: text that is not mostly dictionary words
        is not text a reader can read.
        """
        return text_quality(_maybe_guard(source), include_textdescriptives=include_textdescriptives)

    @server.tool(name="embed_sample")
    @_deliberate
    def embed_sample_tool(texts: list[str], model: str | None = None) -> dict[str, Any]:
        """Embed texts through the LiteLLM gateway; the value is the dimension.

        Requires LITELLM_BASE_URL, LITELLM_API_KEY and an explicit model id.
        Observations are cached on (text, model, tool_version).
        """
        return embed_sample(texts, model=model)

    @server.tool(name="cluster_topics")
    @_deliberate
    def cluster_topics_tool(
        vectors: list[list[float]],
        texts: list[str] | None = None,
        min_topic_size: int = 10,
    ) -> dict[str, Any]:
        """BERTopic over embeddings; the value is the outlier share.

        Needs the optional extra: `pip install -e ".[semantic]"`. Refuses rather
        than substituting k-means, which would produce a different number under
        the same name.
        """
        return cluster_topics(vectors, texts=texts, min_topic_size=min_topic_size)

    @server.tool(name="image_inventory")
    @_deliberate
    def image_inventory_tool(path: str, model: str | None = None, threshold: float = 0.25) -> dict[str, Any]:
        """Zero-shot probe of what a corpus's images are, in one shared space.

        Needs a multimodal embedder (DATA_SCOUT_VISION_EMBED_MODEL) and the
        gateway; fails loudly without both.
        """
        return image_inventory(_guard(path), model=model, threshold=threshold)

    @server.tool(name="near_dup_docs")
    @_deliberate
    def near_dup_docs_tool(texts: list[str], threshold: float = 0.9) -> dict[str, Any]:
        """MinHash/LSH duplicate clusters; the value is the duplicate share.

        One work in three manifestations is the case this exists for.
        """
        return near_dup_docs(texts, threshold=threshold)

    @server.tool(name="near_dup_images")
    @_deliberate
    def near_dup_images_tool(path: str, max_distance: int = 8) -> dict[str, Any]:
        """Perceptual-hash duplicate clusters over an image directory.

        The value is the duplicate share; undecodable images are counted, not
        dropped.
        """
        return near_dup_images(_guard(path), max_distance=max_distance)

    @server.tool(name="sample")
    @_deliberate
    def sample_tool(
        items: list[str],
        n: int,
        seed: int = 0,
        strata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """A seeded, recorded draw — the anti-theatre instrument.

        Every finding line carries `n` and `sample`; this is how the sample was
        chosen and it must be reproducible from the recorded seed.
        """
        return sample(
            items,
            n=n,
            seed=seed,
            stratify_by=(strata.get if strata else None),
        )

    @server.tool(name="findings_append")
    @_deliberate
    def findings_append_tool(record: dict[str, Any]) -> dict[str, Any]:
        """Append one decided line to findings.jsonl, after validating it.

        A line is either `promoted: true` with a module-path `script`, or
        `promoted: false` with a `retire_reason`. Anything else is refused: an
        undecided line is a session that did not close.
        """
        return findings_append(record)

    @server.tool(name="measurement_register")
    @_deliberate
    def measurement_register_tool(
        name: str | None = None,
        group: str | None = None,
        validated_only: bool = False,
    ) -> dict[str, Any]:
        """The registry: what each instrument is, and which corpus trusted its number.

        `validated_on` is corpus ids where the number has been read and trusted —
        not "works". An empty list means built and untrusted, which is the honest
        state of a new belt.
        """
        return measurement_register(name, group=group, validated_only=validated_only)

    return server


def _maybe_guard(value: str) -> str:
    """Guard a string that is *either* a path or the text itself."""
    if "\n" in value or len(value) > 4096:
        return value
    candidate = Path(value).expanduser()
    try:
        return str(_guard(value))
    except ValueError:
        if candidate.exists():
            raise
        # Not a path and not under a root: it is text (or a relative name that
        # does not exist), and the instrument will fail loudly on its own terms.
        return value

