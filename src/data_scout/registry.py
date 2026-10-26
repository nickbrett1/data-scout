"""The measurement registry — the source of truth for the instrument belt.

Everything downstream reads this: the ledger's ``script`` resolution, the
``measurement_register`` tool, and the MCP façade that design §2.4 says is
*generated* from the registry rather than hand-written. So the registry is
built registry-first, not library-first — it is what makes "built but never
trusted" distinguishable from "never built".
"""

from __future__ import annotations

import importlib.util
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Purity = Literal["pure", "read-only-sql", "network"]
Group = Literal["forensics", "structure", "content", "semantic", "redundancy", "sampling", "ledger"]

#: design §10 caps the belt at ~15 and says prune any tool no corpus has used
#: twice. Fifteen instruments are in design §2.2, so the cap is full on day one
#: and there is no headroom: do not add a sixteenth without pruning one.
MAX_BELT_SIZE = 15


class ToolSpec(BaseModel):
    """One instrument, as the ledger and any consumer needs to see it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    group: Group
    version: str
    unit: str = Field(description="what the number is a number of")
    tier: int = Field(ge=0, le=3, description="cost ladder: knowable before running")
    purity: Purity
    deps: tuple[str, ...] = ()
    validated_on: tuple[str, ...] = Field(
        default=(),
        description=(
            "corpus ids where the number has been READ AND TRUSTED — not "
            "'works', not 'has been checked'"
        ),
    )
    summary: str

    @property
    def module(self) -> str:
        """The importable module path — one module per tool.

        ``script`` in a finding line is a module path, not a file path, and it
        is the same string as this tool's module so that
        ``data_scout.tools.<name>`` resolves both as a module and (re-exported
        from the package) as the function.
        """
        return f"data_scout.tools.{self.name}"

    def resolvable(self) -> bool:
        """True when the module named by :attr:`module` is importable."""
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    def as_record(self) -> dict:
        """A JSON-friendly view, including the computed ``resolvable`` flag."""
        return {
            **self.model_dump(mode="json"),
            "module": self.module,
            "resolvable": self.resolvable(),
        }


def _tool(**kwargs) -> ToolSpec:  # tiny helper so the table stays readable
    return ToolSpec(**kwargs)


BELT: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        # ---- forensics (tier 0, exhaustive, free) -------------------------
        _tool(
            name="file_census",
            group="forensics",
            version="1",
            unit="files",
            tier=0,
            purity="pure",
            deps=("file",),
            summary="magic/size/mtime histogram over a tree",
        ),
        _tool(
            name="pdf_forensics",
            group="forensics",
            version="1",
            unit="documents (PDFs)",
            tier=0,
            purity="pure",
            deps=("poppler-utils",),
            summary="pdfinfo + pdffonts + pdfimages -list in one call; zero fonts means no text layer",
        ),
        _tool(
            name="exif_census",
            group="forensics",
            version="1",
            unit="images",
            tier=0,
            purity="pure",
            deps=("exiftool",),
            summary="exiftool over a tree, aggregated into make/model/software/date histograms",
        ),
        # ---- structure ----------------------------------------------------
        _tool(
            name="partition_doc",
            group="structure",
            version="1",
            unit="parts",
            tier=1,
            purity="network",
            deps=("httpx", "docling-serve"),
            summary="Docling layout-aware parts and reading order, as a library client of a sidecar",
        ),
        _tool(
            name="structure_mix",
            group="structure",
            version="1",
            unit="parts",
            tier=1,
            purity="pure",
            deps=(),
            summary="per-document share of each part type — a prose book and a plate book separate immediately",
        ),
        # ---- content ------------------------------------------------------
        _tool(
            name="candidate_density",
            group="content",
            version="1",
            unit="characters",
            tier=1,
            purity="pure",
            deps=("wordfreq",),
            summary="reader's quotes.py filter used as a measuring instrument, per 1000 chars",
        ),
        _tool(
            name="text_quality",
            group="content",
            version="1",
            unit="characters",
            tier=1,
            purity="pure",
            deps=("wordfreq", "textdescriptives"),
            summary="char entropy, non-alpha ratio, dictionary hit rate, language ID, readability",
        ),
        # ---- semantic (tier 2, optional extra, must fail loudly) ----------
        _tool(
            name="embed_sample",
            group="semantic",
            version="1",
            unit="texts",
            tier=2,
            purity="network",
            deps=("httpx", "numpy"),
            summary="stratified sample -> embeddings via the LiteLLM gateway",
        ),
        _tool(
            name="cluster_topics",
            group="semantic",
            version="1",
            unit="documents",
            tier=2,
            purity="pure",
            deps=("bertopic", "umap-learn", "hdbscan", "scikit-learn"),
            summary="UMAP -> HDBSCAN -> c-TF-IDF labels; the outlier share is the number that matters",
        ),
        _tool(
            name="image_inventory",
            group="semantic",
            version="1",
            unit="images",
            tier=2,
            purity="network",
            deps=("pillow", "numpy", "httpx"),
            summary="zero-shot tag counts by cosine against a short probe list in the shared image space",
        ),
        # ---- redundancy ---------------------------------------------------
        _tool(
            name="near_dup_docs",
            group="redundancy",
            version="1",
            unit="documents",
            tier=1,
            purity="pure",
            deps=("datasketch",),
            summary="MinHash/LSH; catches one work in epub+mobi+pdf and boilerplate chunks",
        ),
        _tool(
            name="near_dup_images",
            group="redundancy",
            version="1",
            unit="images",
            tier=1,
            purity="pure",
            deps=("pillow", "imagehash"),
            summary="phash clusters — reuses the approach of reader's image dedupe, not its code",
        ),
        # ---- sampling -----------------------------------------------------
        _tool(
            name="sample",
            group="sampling",
            version="1",
            unit="rows",
            tier=0,
            purity="pure",
            deps=(),
            summary="stratified/seeded draw — the rig is a tool, not a habit",
        ),
        # ---- ledger -------------------------------------------------------
        _tool(
            name="findings_append",
            group="ledger",
            version="1",
            unit="records",
            tier=0,
            purity="pure",
            deps=("pydantic",),
            summary="append one decided finding line; no update, no delete, by design",
        ),
        _tool(
            name="measurement_register",
            group="ledger",
            version="1",
            unit="registry entries",
            tier=0,
            purity="pure",
            deps=("pydantic",),
            summary="read this registry: name, version, unit, tier, purity, deps, validated_on",
        ),
    )
}


def names() -> tuple[str, ...]:
    """Every instrument name, in belt order."""
    return tuple(BELT)


def get(name: str) -> ToolSpec:
    try:
        return BELT[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not in the belt. The belt is {len(BELT)} instruments "
            f"by design (design §10); a corpus that needs a sixteenth is a "
            f"recorded gap, not a new tool. Known: {', '.join(names())}"
        ) from None


def tool_version(name: str) -> str:
    """The version string that is part of every cache key and finding line."""
    return get(name).version


def all_tools() -> tuple[ToolSpec, ...]:
    return tuple(BELT.values())


def as_records() -> list[dict]:
    return [spec.as_record() for spec in BELT.values()]


def problems() -> list[str]:
    """Everything wrong with the registry, as strings. Empty means healthy.

    Checked as a test rather than asserted at import: a registry that raises on
    import is a package nobody can import to find out what is wrong.
    """
    found: list[str] = []
    if len(BELT) > MAX_BELT_SIZE:
        found.append(f"belt has {len(BELT)} tools; the cap is {MAX_BELT_SIZE}")
    for spec in BELT.values():
        if spec.name != spec.module.rsplit(".", 1)[-1]:
            found.append(f"{spec.name}: module path does not end in the tool name")
        if not spec.resolvable():
            found.append(f"{spec.name}: module {spec.module} is not importable")
        if spec.purity == "network" and spec.tier < 1:
            found.append(f"{spec.name}: network purity at tier 0 cannot be free by construction")
    return found
