"""The brief — what to run on this corpus, and what would change my mind.

A report is evidence: this instrument said this number. A *brief* is a decision:
given those numbers, what kind of thing is this, what should be built on top of
it, and what is the next measurement that could overturn the answer. Both are
needed and they are not the same document — a table of readings, however crisp,
leaves the reader to invent the thesis, and the thesis is the only part they
actually wanted.

So this module holds the judgement in one place, as data:

* :func:`read` turns ledger lines into a :class:`Corpus` — a handful of facts,
  each carrying the line it came from, so a brief can be *checked* rather than
  believed. Anything it could not find stays ``None`` and is reported as an open
  axis, never as a zero.
* :func:`axes` answers the questions that actually separate one system from
  another (is there a schema? is the content text? does the parser see it?), each
  with the reading *and* what it implies.
* :func:`verdict` picks one of five systems — relational, document intelligence,
  vision over page images, OCR-then-documents, or a routed corpus — and says what
  not to build, plus the two or three measurements that would change its mind.

The thresholds are stated, not buried: :data:`PROSE_WORDS_PER_PAGE` and
:data:`PLATE_IMAGES_PER_PAGE` are the line between "text you can read" and
"pictures with captions". They are heuristics with a citation, and the brief
prints them so a reader can disagree with a number instead of a mood.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from data_scout import render, workspace
from data_scout.ledger import Ledger

Kind = Literal["prose", "plate", "sparse", "mixed", "unknown"]

#: Above this, a page is mostly words and a text pipeline has something to read.
PROSE_WORDS_PER_PAGE = 150.0

#: Above this, a page is mostly pictures and the text layer is a caption, not
#: content. Below :data:`SPARSE_WORDS_PER_PAGE` there is barely a caption either.
PLATE_IMAGES_PER_PAGE = 5.0
SPARSE_WORDS_PER_PAGE = 30.0

#: A corpus is "tabular" when most parsed table parts carry text — that is the
#: only shape for which a relational store beats a document store.
TABLE_SHARE_FOR_SQL = 0.5


@dataclass(frozen=True)
class Fact:
    """One number from the ledger, phrased as a reading, with its line."""

    name: str
    value: Any
    reading: str
    line: int


@dataclass(frozen=True)
class DocProfile:
    """What one document's numbers say it is: words a page, pictures a page."""

    document: str
    pages: int | None = None
    words: int | None = None
    images_per_page: float | None = None
    lines: tuple[int, ...] = ()

    @property
    def words_per_page(self) -> float | None:
        if self.words is None or not self.pages:
            return None
        return self.words / self.pages

    @property
    def kind(self) -> Kind:
        """Which pipeline this document belongs in."""
        if self.images_per_page is not None and self.images_per_page >= PLATE_IMAGES_PER_PAGE:
            return "plate"
        words_per_page = self.words_per_page
        if words_per_page is None:
            return "unknown"
        if words_per_page >= PROSE_WORDS_PER_PAGE:
            return "prose"
        if words_per_page < SPARSE_WORDS_PER_PAGE:
            return "sparse"
        return "mixed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "pages": self.pages,
            "words": self.words,
            "words_per_page": None if self.words_per_page is None else round(self.words_per_page, 1),
            "images_per_page": self.images_per_page,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Corpus:
    """The handful of facts a decision needs, each traceable to a ledger line."""

    facts: tuple[Fact, ...] = ()
    documents: tuple[DocProfile, ...] = ()
    tables_with_text: int | None = None
    table_parts: int | None = None
    pages_lost: int | None = None
    pages_measured: int | None = None

    def fact(self, name: str) -> Fact | None:
        for fact in self.facts:
            if fact.name == name:
                return fact
        return None

    def by_kind(self, kind: Kind) -> tuple[DocProfile, ...]:
        return tuple(document for document in self.documents if document.kind == kind)

    @property
    def table_share(self) -> float | None:
        if not self.table_parts or self.tables_with_text is None:
            return None
        return self.tables_with_text / self.table_parts


@dataclass(frozen=True)
class Axis:
    """One decision axis: the question, the reading, and what it implies."""

    name: str
    question: str
    reading: str
    implies: str
    lines: tuple[int, ...] = ()
    open: bool = False


@dataclass(frozen=True)
class Verdict:
    """One recommendation, with its costs and its refutation conditions."""

    system: str
    thesis: str
    moves: tuple[str, ...]
    not_to_do: tuple[str, ...]
    falsifiers: tuple[str, ...]


# --------------------------------------------------------------------------
# reading the ledger
# --------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _document_of(params: dict[str, Any]) -> str | None:
    for key in ("document", "file"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def read(findings: Sequence[Any]) -> Corpus:
    """Turn ledger findings into the facts a decision rests on.

    Two extraction rules, deliberately narrow: a document profile comes from any
    line naming a document and its page count, and a corpus fact comes from a
    line whose unit says what it is a number of. Everything else is ignored here
    — the brief is not a second report, and a fact it invents is a fact nobody
    can check.
    """
    facts: list[Fact] = []
    profiles: dict[str, DocProfile] = {}
    tables_with_text: int | None = None
    table_parts: int | None = None
    pages_lost: int | None = None
    pages_measured: int | None = None

    for line_number, finding in enumerate(findings, start=1):
        params = dict(getattr(finding, "params", None) or {})
        unit = str(getattr(finding, "unit", "") or "")
        value = getattr(finding, "value", None)
        count = getattr(finding, "n", None)

        # The dataset frame: how many files, how many bytes, what it is called.
        if getattr(finding, "tool", "") == "frame" and params.get("identity_files"):
            files = int(params["identity_files"])
            size = _number(value)
            if size is not None:
                facts.append(
                    Fact(
                        name="bytes",
                        value=size,
                        reading=f"{size / 1e6:.0f} MB across {files} files",
                        line=line_number,
                    )
                )

        # Per-document shape: any line naming a document with a page count.
        document = _document_of(params)
        pages = _number(params.get("pages"))
        if document and pages:
            profile = profiles.get(document, DocProfile(document=document))
            update: dict[str, Any] = {"pages": int(pages), "lines": (*profile.lines, line_number)}
            words = _number(params.get("docling_words") or params.get("words"))
            if words is not None:
                update["words"] = int(words)
            if "images per page" in unit:
                images = _number(value)
                if images is not None:
                    update["images_per_page"] = round(images, 2)
                    facts.append(
                        Fact(
                            name=f"images_per_page:{document}",
                            value=images,
                            reading=f"{document}: {images:.1f} images a page",
                            line=line_number,
                        )
                    )
            elif words is not None and pages:
                per_page = words / pages
                facts.append(
                    Fact(
                        name=f"words_per_page:{document}",
                        value=per_page,
                        reading=f"{document}: {per_page:.0f} words a page over {int(pages)} pages",
                        line=line_number,
                    )
                )
            profiles[document] = replace(profile, **update)

        # Table parts: the only evidence that a relational store could pay.
        if "table parts" in unit:
            if isinstance(count, int | float):
                table_parts = int(count)
            if isinstance(getattr(finding, "value", None), int | float):
                tables_with_text = int(finding.value)

        # Parser loss: pages where the layout parser returned nothing.
        if unit.startswith("pages (of"):
            lost = _number(value)
            measured = _number(count)
            if lost is not None:
                pages_lost = int(lost)
            if measured is not None:
                pages_measured = int(measured)

    # The ledger is append-only, so a re-run repeats a fact. The newest line is
    # the one to cite; older copies would make the brief look like it had two
    # answers where it has one.
    newest: dict[str, Fact] = {}
    for fact in facts:
        newest[fact.name] = fact

    corpus = Corpus(
        facts=tuple(newest.values()),
        documents=tuple(profiles.values()),
        tables_with_text=tables_with_text,
        table_parts=table_parts,
        pages_lost=pages_lost,
        pages_measured=pages_measured,
    )
    return corpus


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------


def axes(corpus: Corpus) -> tuple[Axis, ...]:
    """The questions that separate one system from another, answered or open."""
    prose = corpus.by_kind("prose")
    plates = corpus.by_kind("plate")
    sparse = corpus.by_kind("sparse")

    table_share = corpus.table_share
    if table_share is None:
        shape = Axis(
            name="Shape",
            question="Is there a schema to query, or a document to read?",
            reading="no finding measured table parts, so the question is unanswered",
            implies="measure the structure mix before committing to any store",
            open=True,
        )
    elif table_share >= TABLE_SHARE_FOR_SQL:
        shape = Axis(
            name="Shape",
            question="Is there a schema to query, or a document to read?",
            reading=f"{corpus.tables_with_text} of {corpus.table_parts} parsed table parts carry text "
            f"({table_share:.0%})",
            implies="a relational store is the right home: this is rows, not prose",
        )
    else:
        shape = Axis(
            name="Shape",
            question="Is there a schema to query, or a document to read?",
            reading=f"{corpus.tables_with_text} of {corpus.table_parts} parsed table parts carry text "
            f"({table_share:.0%}) — the tables are furniture, not data",
            implies="do not build a warehouse or a chart pipeline; there is nothing to aggregate",
            lines=(),
        )

    if prose or plates:
        parts = []
        if prose:
            parts.append(f"{len(prose)} prose")
        if plates:
            parts.append(f"{len(plates)} image-plate")
        if sparse:
            parts.append(f"{len(sparse)} sparse-text")
        unmeasured = len(corpus.documents) - len(prose) - len(plates) - len(sparse) - len(
            corpus.by_kind("mixed")
        )
        if unmeasured:
            parts.append(f"{unmeasured} not yet readable as either")
        modality = Axis(
            name="Modality",
            question="Is the content words, pictures, or both?",
            reading=", ".join(parts) + f" — of {len(corpus.documents)} profiled documents",
            implies=(
                "one system cannot serve both: route each document to the pipeline its numbers name"
                if prose and plates
                else "one pipeline is enough"
            ),
            lines=tuple(sorted({line for document in corpus.documents for line in document.lines})),
        )
    else:
        modality = Axis(
            name="Modality",
            question="Is the content words, pictures, or both?",
            reading="no document carries both a page count and a word or image count",
            implies="nothing can be routed yet: measure a page and a word count per document",
            open=True,
        )

    lost = corpus.pages_lost
    measured = corpus.pages_measured
    if lost is not None and measured:
        fidelity = Axis(
            name="Parser fidelity",
            question="Does the layout parser see everything a reader would?",
            reading=f"{lost} of {measured} pages came back empty from the parser while a raw text "
            f"extractor had text ({lost / measured:.1%})",
            implies="the pipeline needs a whole-page fallback, or it silently drops pages",
        )
    else:
        fidelity = Axis(
            name="Parser fidelity",
            question="Does the layout parser see everything a reader would?",
            reading="no page-level parser loss was measured",
            implies="measure it on the documents you intend to parse before trusting the parse",
            open=True,
        )

    return (shape, modality, fidelity)


def verdict(corpus: Corpus) -> Verdict:
    """One recommendation from the facts, or the single measurement to take first."""
    prose = corpus.by_kind("prose")
    plates = corpus.by_kind("plate")
    share = corpus.table_share

    if share is not None and share >= TABLE_SHARE_FOR_SQL:
        return Verdict(
            system="Relational store (SQL) with a thin BI layer",
            thesis=(
                f"Most parsed table parts carry text ({share:.0%}): this corpus is rows. "
                "Load it into a table per document family, keep the prose as a notes column, "
                "and let the questions be SQL."
            ),
            moves=(
                "Score the tables for header regularity before designing a schema — a table with "
                "merged cells is a paragraph drawn with lines.",
                "Load one document family end to end and try three real questions against it.",
                "Keep the raw part text beside every row so a wrong parse is visible, not hidden.",
            ),
            not_to_do=(
                "Do not start with a document embedding pipeline: retrieval over prose answers "
                "different questions than the ones a row-shaped corpus invites.",
            ),
            falsifiers=(
                "If the table headers turn out to be irregular across documents, the schema is a "
                "document store with a table extractor, not a warehouse.",
                "If one family dominates and the others are noise, load only that family.",
            ),
        )

    if plates and not prose:
        return Verdict(
            system="Vision over page images (describe, then embed)",
            thesis=(
                f"The profiled documents are picture books — up to "
                f"{max((d.images_per_page or 0) for d in plates):.0f} images a page — so the text "
                "layer is a caption, not content. Render pages to images and let a vision model "
                "say what is on them; retrieve over those descriptions."
            ),
            moves=(
                "Render a 20-page sample at 200 DPI and caption it; read the captions yourself "
                "before paying for the rest.",
                "Decide the unit: page, spread, or single image. A spread is often the artwork.",
                "Embed the captions, not the pixels, for retrieval — captions are searchable text "
                "and pixels are not.",
            ),
            not_to_do=(
                "Do not build a text pipeline on the extractable text layer: it is captions and "
                "page furniture, and summarising it describes the book's layout.",
            ),
            falsifiers=(
                "If captions already carry the meaning, a cheap text pipeline over them wins — "
                "check that before rendering a thousand pages.",
                "If the plates are decorative (repeated motifs, no per-page meaning), the corpus is "
                "smaller than it looks.",
            ),
        )

    if prose and plates:
        plate_word = "document" if len(plates) == 1 else "documents"
        unknown = len(corpus.documents) - len(prose) - len(plates) - len(
            corpus.by_kind("sparse")
        ) - len(corpus.by_kind("mixed"))
        unmeasured = (
            f" {unknown} more {'is' if unknown == 1 else 'are'} profiled but has no word count yet, "
            "so it is not yet routed."
            if unknown
            else ""
        )
        return Verdict(
            system="Document intelligence, routed per document",
            thesis=(
                f"This is two corpora in one directory. {len(prose)} of "
                f"{len(corpus.documents)} profiled documents read as prose (at least "
                f"{PROSE_WORDS_PER_PAGE:.0f} words a page): parse, chunk, embed, then retrieve and "
                f"summarise. {len(plates)} {plate_word} "
                f"{'is an image-plate book' if len(plates) == 1 else 'are image-plate books'} where "
                "a page's words are a caption, and a text pipeline will confidently describe "
                f"nothing.{unmeasured} Route on the numbers, not on the file extension."
            ),
            moves=(
                "Profile every document with words and images per page — the routing decision is "
                "that table and nothing else.",
                "For the prose books: parse with the layout parser, chunk by section, embed, then "
                "ask the three questions you actually have (themes, quotes, what a chapter says).",
                "For the plate books: render a 20-page sample and caption it; a browser view of the "
                "pages is a legitimate first deliverable and costs nothing.",
            ),
            not_to_do=(
                "Do not run one pipeline over the whole directory and average the result: the "
                "average of a novel and an artbook describes neither.",
                "Do not build a warehouse. Nothing here is rows.",
                "Do not trust a parse that returns an empty page: add a whole-page text fallback "
                "where the parser and a raw extractor disagree.",
            ),
            falsifiers=(
                "If most 'prose' documents turn out to be scanned images, the text pipeline becomes "
                "an OCR job — measure the text layer before parsing.",
                "If the plate books are the ones you care about, this is a vision project with a "
                "text appendix, and the order of work flips.",
                "If a document's text is unreachable (drm, permissions), it is a specimen for "
                "inspection only until that is solved.",
            ),
        )

    if prose:
        return Verdict(
            system="Document intelligence (parse, chunk, embed, retrieve)",
            thesis=(
                f"Every profiled document reads as prose (at least {PROSE_WORDS_PER_PAGE:.0f} words a "
                "page), so the work is reading: parse with structure, chunk by section, embed, then "
                "retrieve, summarise and mine quotes and themes."
            ),
            moves=(
                "Parse one document end to end and check the section structure by eye.",
                "Chunk by section and embed; ask three real questions and grade the answers.",
                "Mine quotes and themes with citations back to a page, or the output is not checkable.",
            ),
            not_to_do=("Do not build a warehouse, and do not caption pages you can already read.",),
            falsifiers=(
                "If the chunked retrieval cannot answer a question the document plainly answers, "
                "the parse is wrong, not the model.",
                "If the corpus is only a few hundred pages, an agent reading it whole beats a "
                "vector index — measure the size before building one.",
            ),
        )

    return Verdict(
        system="Undecided — measure before building",
        thesis=(
            "The ledger does not yet contain the facts that separate one system from another: how "
            "many documents there are, how many words and images a page, whether any table carries "
            "text. Anything chosen now is a guess wearing a recommendation's clothes."
        ),
        moves=(
            "Profile each document: pages, words, images. Two numbers a document decide more than "
            "any model choice.",
            "Measure the structure mix on one document: how much of it is prose, how much is a picture.",
            "Only then pick a pipeline — the brief will pick one for you.",
        ),
        not_to_do=("Do not start with the model. Start with the page.",),
        falsifiers=("Any of the three measurements above changes the recommendation.",),
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _escape(text: str) -> str:
    return render._escape(text)


def _lines_html(lines: Iterable[int]) -> str:
    """The ledger lines a reading came from, as a quiet citation."""
    numbers = tuple(lines)
    if not numbers:
        return ""
    joined = ", ".join(str(number) for number in numbers)
    plural = "lines" if len(numbers) > 1 else "line"
    return f'<span class="brief-cite">{plural} {joined}</span>'


def _fmt(value: float | None, template: str) -> str:
    return "—" if value is None else template.format(value)


def _documents_html(corpus: Corpus) -> str:
    if not corpus.documents:
        return ""
    rows = [
        '<table class="brief-table"><thead><tr>'
        "<th>document</th><th>pages</th><th>words / page</th><th>images / page</th><th>reads as</th>"
        "</tr></thead><tbody>"
    ]
    for document in corpus.documents:
        name = document.document.split("/")[-1]
        rows.append(
            "<tr>"
            f'<td class="brief-doc">{_escape(name)}</td>'
            f"<td>{_fmt(None if document.pages is None else float(document.pages), '{:.0f}')}</td>"
            f"<td>{_fmt(document.words_per_page, '{:.0f}')}</td>"
            f"<td>{_fmt(document.images_per_page, '{:.1f}')}</td>"
            f'<td><span class="brief-kind k-{document.kind}">{document.kind}</span></td>'
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def body(corpus: Corpus) -> str:
    """The brief's HTML body: verdict, axes, documents, plan, refutations."""
    said = verdict(corpus)
    parts = [
        '<section class="brief-verdict">',
        '<p class="brief-label">What to run on this corpus</p>',
        f'<h1>{_escape(said.system)}</h1>',
        f"<p>{_escape(said.thesis)}</p>",
        "</section>",
        '<section class="brief-axes"><h2>What the numbers force</h2>',
    ]
    for axis in axes(corpus):
        state = "open" if axis.open else "settled"
        parts.append(
            f'<div class="brief-axis a-{state}">'
            f'<p class="brief-question">{_escape(axis.question)}</p>'
            f'<p class="brief-reading">{_escape(axis.reading)} '
            f"{_lines_html(axis.lines)}</p>"
            f'<p class="brief-implies">{_escape(axis.implies)}</p>'
            "</div>"
        )
    parts.append("</section>")

    documents = _documents_html(corpus)
    if documents:
        parts.append(
            '<section class="brief-docs"><h2>Per document</h2>'
            '<p class="brief-note">Words a page and pictures a page are the routing decision: '
            f"prose is at least {PROSE_WORDS_PER_PAGE:.0f} words a page, a plate book at least "
            f"{PLATE_IMAGES_PER_PAGE:.0f} images a page. A dash is a measurement nobody took.</p>"
            f"{documents}</section>"
        )

    parts.append('<section class="brief-do"><h2>First moves</h2><ol>')
    parts.extend(f"<li>{_escape(move)}</li>" for move in said.moves)
    parts.append("</ol></section>")

    parts.append('<section class="brief-not"><h2>Do not</h2><ul>')
    parts.extend(f"<li>{_escape(item)}</li>" for item in said.not_to_do)
    parts.append("</ul></section>")

    parts.append('<section class="brief-open"><h2>What would change this</h2><ul>')
    parts.extend(f"<li>{_escape(item)}</li>" for item in said.falsifiers)
    parts.append("</ul></section>")

    parts.append(
        '<section class="report-notes"><h2>Where this comes from</h2>'
        "<p>Every reading above is a line in <code>findings.jsonl</code>; the citations name the "
        "line. Nothing here is a measurement of its own — the evidence is the report.</p></section>"
    )
    return "".join(parts)


def render_brief(corpus: Corpus, *, dataset_id: str | None = None) -> str:
    """The whole brief, as one styled document."""
    title = f"brief — {dataset_id}" if dataset_id else "brief"
    meta = f"{dataset_id} · what to run on it" if dataset_id else "what to run on a corpus"
    return render.html_document(body(corpus), title=title, meta=meta)


def write_brief(dataset_id: str, ledger: Ledger) -> Any:
    """Read the ledger for a dataset, write ``brief-<dataset>.html``, return its path."""
    corpus = read(ledger.findings())
    target = workspace.artifact_path(dataset_id, f"brief-{dataset_id}.html")
    target.write_text(render_brief(corpus, dataset_id=dataset_id), encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """``brief-render <dataset-id>`` — the decision, from the ledger as it stands."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="brief-render", description="What to run on a dataset.")
    parser.add_argument("dataset", nargs="?", help="dataset id; defaults to DATA_SCOUT_DATASET_ID")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    dataset_id = args.dataset or workspace.current_dataset_id()
    ledger_path = workspace.ledger_path(dataset_id)
    if not ledger_path.is_file():
        print(f"brief-render: no ledger at {ledger_path}", file=sys.stderr)
        return 2
    written = write_brief(dataset_id, Ledger(ledger_path))
    if not args.quiet:
        corpus = read(Ledger(ledger_path).findings())
        print(f"brief-render: {verdict(corpus).system}")
        print(f"brief-render: wrote {written}")
    return 0
