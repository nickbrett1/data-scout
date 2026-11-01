"""The brief — what to run on this corpus, and what would change my mind.

A report is evidence: this instrument said this number. A *brief* is a decision,
and it is built in two separable halves:

1. **facts** — a closed set of named readings, each citing the ledger lines it came
   from. This module's :func:`read` is the only place a ledger line becomes a
   fact, and it never guesses: a fact nobody measured is absent, not zero.
2. **evaluation** — :mod:`data_scout.profiles` scores every profile in the
   catalogue against those facts and returns pass / fail / **unknown** per
   criterion. Unknown never disqualifies; it names the next measurement.

This module renders the result: the winner, the whole matrix (including the
profiles the corpus disqualifies), the per-document routing table, and the plan.
All profiles are shown because a single label hides the refusals, and the refusals
are half the answer.

The evidence — the numbers themselves — stays where it belongs, in the report.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any, Literal

from data_scout import profiles, render, workspace
from data_scout.ledger import Ledger

Kind = Literal["prose", "plate", "sparse", "mixed", "unknown"]

#: Above this, a page is mostly words and a text pipeline has something to read.
PROSE_WORDS_PER_PAGE = 150.0

#: Above this, a page is mostly pictures and the text layer is a caption, not
#: content. Below :data:`SPARSE_WORDS_PER_PAGE` there is barely a caption either.
PLATE_IMAGES_PER_PAGE = 5.0
SPARSE_WORDS_PER_PAGE = 30.0


@dataclass(frozen=True)
class Fact:
    """One reading from the ledger, with the lines that support it.

    ``observed`` is the sentence a reader sees — the number formatted for a page,
    because "0.0" under a heading means nothing to anybody. ``value`` is what a
    criterion tests.
    """

    name: str
    value: float
    observed: str
    lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class DocProfile:
    """What one document's numbers say it is: words a page, pictures a page."""

    document: str
    pages: int | None = None
    words: int | None = None
    images_per_page: float | None = None
    narrative_share: float | None = None
    lines: tuple[int, ...] = ()

    @property
    def words_per_page(self) -> float | None:
        if self.words is None or not self.pages:
            return None
        return self.words / self.pages

    def value(self, fact: str) -> float | None:
        """The per-document fact a criterion asked for, or None if unmeasured."""
        if fact == "words_per_page":
            return self.words_per_page
        if fact == "images_per_page":
            return self.images_per_page
        if fact == "narrative_share":
            return self.narrative_share
        return None

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

    @property
    def name(self) -> str:
        return self.document.split("/")[-1]


#: The per-document facts a profile may test. Declared, so evaluation never has to
#: guess whether ``images_per_page`` means "the corpus" or "each document".
PER_DOCUMENT_FACTS = frozenset({"words_per_page", "images_per_page", "narrative_share"})


@dataclass(frozen=True)
class Corpus:
    """The facts a decision rests on, and the documents they were read from."""

    facts: tuple[Fact, ...] = ()
    documents: tuple[DocProfile, ...] = ()

    def fact(self, name: str) -> Fact | None:
        for fact in self.facts:
            if fact.name == name:
                return fact
        return None

    def per_document_facts(self) -> frozenset[str]:
        return PER_DOCUMENT_FACTS

    def by_kind(self, kind: Kind) -> tuple[DocProfile, ...]:
        return tuple(document for document in self.documents if document.kind == kind)


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


def _count(number: float) -> str:
    """A count for a page: 1,160 pages, not 1160.0 pages."""
    return f"{number:,.0f}"


def read(findings: Sequence[Any]) -> Corpus:
    """Turn ledger findings into the facts a decision rests on.

    Every fact here has one documented source pattern, and a fact whose source is
    not in the ledger is simply absent. That is the whole design: the brief may not
    invent a reading, and a missing reading is what makes a profile *unknown*
    rather than disqualified.
    """
    facts: dict[str, Fact] = {}
    profiles_by_document: dict[str, DocProfile] = {}
    table_parts: int | None = None
    tables_with_text: int | None = None
    pages_lost: int | None = None
    pages_measured: int | None = None
    table_line: int | None = None
    loss_line: int | None = None

    def put(fact: Fact) -> None:
        # The ledger is append-only, so a re-run repeats a fact. The newest line
        # wins; older copies would make the brief look like it had two answers.
        facts[fact.name] = fact

    for line_number, finding in enumerate(findings, start=1):
        params = dict(getattr(finding, "params", None) or {})
        unit = str(getattr(finding, "unit", "") or "")
        value = getattr(finding, "value", None)
        count = getattr(finding, "n", None)
        tool = str(getattr(finding, "tool", "") or "")

        # The dataset frame: how many payload files, and how many bytes.
        if tool == "frame" and params.get("identity_files"):
            files = int(params["identity_files"])
            size = _number(value)
            if size is not None:
                put(
                    Fact(
                        "documents",
                        float(files),
                        f"{files} documents",
                        (line_number,),
                    )
                )
                put(
                    Fact(
                        "bytes",
                        size,
                        f"{size / 1e6:.0f} MB across {files} documents",
                        (line_number,),
                    )
                )

        # Per-document shape: any line naming a document with a page count.
        document = _document_of(params)
        page_count = _number(params.get("pages"))
        if document and page_count:
            profile = profiles_by_document.get(document, DocProfile(document=document))
            update: dict[str, Any] = {
                "pages": int(page_count),
                "lines": (*profile.lines, line_number),
            }
            words = _number(params.get("docling_words") or params.get("words"))
            if words is not None:
                update["words"] = int(words)
            images = _number(value) if "images per page" in unit else None
            if images is not None:
                update["images_per_page"] = round(images, 2)
            share = _number(value) if tool == "structure_mix" else None
            if share is not None:
                update["narrative_share"] = share
            changed = any(
                getattr(profile, field) != value
                for field, value in update.items()
                if field != "lines"
            )
            if not changed:
                # An append-only ledger repeats a measurement when a tool is re-run.
                # The newest line is the citation; older identical lines are not
                # a second answer, and a document measured once must not be cited twice.
                update["lines"] = (line_number,)
            profiles_by_document[document] = DocProfile(**{**profile.__dict__, **update})

        # Table parts: the only evidence that a relational store could pay.
        if "table parts" in unit:
            table_line = line_number
            if isinstance(count, int | float):
                table_parts = int(count)
            if isinstance(value, int | float):
                tables_with_text = int(value)

        # Parser loss: pages where the layout parser returned nothing.
        if unit.startswith("pages (of"):
            loss_line = line_number
            lost = _number(value)
            measured = _number(count)
            if lost is not None:
                pages_lost = int(lost)
            if measured is not None:
                pages_measured = int(measured)

        # Text layer: pdf_forensics counts documents *without* one. Lines that
        # carry an `attribute` (fonts, encryption) or a `probe` (extraction test)
        # are different questions about the same unit and are excluded here.
        if (
            tool == "pdf_forensics"
            and unit.startswith("documents (PDF")
            and "attribute" not in params
            and "probe" not in params
        ):
            missing = _number(value)
            total = _number(count)
            if missing is not None and total:
                have = total - missing
                put(
                    Fact(
                        "text_layer_share",
                        have / total,
                        f"{_count(have)} of {_count(total)} PDFs carry a text layer",
                        (line_number,),
                    )
                )

        # Near-duplicate share: the answer to "is this eight books or three?".
        if tool == "near_dup_docs":
            share = _number(value)
            if share is not None:
                put(
                    Fact(
                        "dup_share",
                        share,
                        f"{share:.0%} of documents are near duplicates",
                        (line_number,),
                    )
                )

    documents = tuple(profiles_by_document.values())

    # --- derived facts: arithmetic over the readings, never a new observation ---
    # A derived fact cites the lines it was derived from, so every number on the
    # page can be traced back to a reading: an arithmetic result is still an
    # answer to "where did this come from?".
    document_lines = tuple(sorted({line for document in documents for line in document.lines}))

    pages = [document.pages for document in documents if document.pages]
    if pages:
        put(
            Fact(
                "pages",
                float(sum(pages)),
                f"{_count(sum(pages))} pages profiled",
                document_lines,
            )
        )

    words = [document.words for document in documents if document.words]
    if words:
        put(
            Fact(
                "corpus_words",
                float(sum(words)),
                f"{_count(sum(words))} words",
                document_lines,
            )
        )

    per_page = [document.words_per_page for document in documents if document.words_per_page]
    if per_page:
        lines = tuple(sorted({line for document in documents for line in document.lines}))
        put(
            Fact(
                "words_per_page_median",
                float(median(per_page)),
                f"median {median(per_page):.0f} words a page",
                lines,
            )
        )

    picture_counts = [
        document.images_per_page for document in documents if document.images_per_page is not None
    ]
    if picture_counts:
        densest = max(documents, key=lambda document: document.images_per_page or 0)
        put(
            Fact(
                "images_per_page_max",
                float(max(picture_counts)),
                f"up to {max(picture_counts):.1f} images a page ({densest.name})",
                densest.lines,
            )
        )

    if documents:
        plates = [document for document in documents if document.kind == "plate"]
        put(
            Fact(
                "plate_share",
                len(plates) / len(documents),
                f"{len(plates)} of {len(documents)} profiled documents "
                f"{'is a plate book' if len(plates) == 1 else 'are plate books'}",
                tuple(sorted({line for document in documents for line in document.lines})),
            )
        )

    shares = [
        document.narrative_share
        for document in documents
        if document.narrative_share is not None
    ]
    if shares:
        put(
            Fact(
                "narrative_share_median",
                float(median(shares)),
                f"median narrative share {median(shares):.2f}",
                document_lines,
            )
        )

    if table_parts and tables_with_text is not None:
        put(
            Fact(
                "table_share",
                tables_with_text / table_parts,
                f"{tables_with_text} of {table_parts} parsed table parts carry text",
                () if table_line is None else (table_line,),
            )
        )

    if pages_lost is not None and pages_measured:
        put(
            Fact(
                "parser_loss_share",
                pages_lost / pages_measured,
                f"{pages_lost} of {_count(pages_measured)} pages came back empty from the parser "
                f"({pages_lost / pages_measured:.1%})",
                () if loss_line is None else (loss_line,),
            )
        )

    return Corpus(facts=tuple(facts.values()), documents=documents)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _escape(text: str) -> str:
    return render._escape(text)


def _cite(lines: Iterable[int]) -> str:
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
        rows.append(
            "<tr>"
            f'<td class="brief-doc">{_escape(document.name)}</td>'
            f"<td>{_fmt(None if document.pages is None else float(document.pages), '{:.0f}')}</td>"
            f"<td>{_fmt(document.words_per_page, '{:.0f}')}</td>"
            f"<td>{_fmt(document.images_per_page, '{:.1f}')}</td>"
            f'<td><span class="brief-kind k-{document.kind}">{document.kind}</span></td>'
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "".join(rows)


_MARKS = {"pass": "\u2713", "fail": "\u2717", "unknown": "?"}


def _cell_html(cell: profiles.Cell) -> str:
    observed = cell.observed or "not measured"
    detail = ""
    if cell.state == "unknown" and cell.tool:
        detail = f'<span class="cell-task">run <code>{_escape(cell.tool)}</code></span>'
    elif cell.documents:
        detail = f'<span class="cell-task">{_escape(", ".join(cell.documents))}</span>'
    return (
        f'<li class="cell c-{cell.state}">'
        f'<span class="mark">{_MARKS[cell.state]}</span>'
        f'<code class="cell-test">{_escape(cell.criterion.describe())}</code>'
        f'<span class="cell-observed">{_escape(observed)}</span>'
        f"{_cite(cell.lines)}{detail}"
        "</li>"
    )


def _assessment_html(assessment: profiles.Assessment) -> str:
    profile = assessment.profile
    # A profile the corpus fits is the news and opens; a disqualified one is one
    # line of information and stays shut.
    opened = " open" if assessment.outcome in ("fits", "partial") else ""
    favour_cells = "".join(_cell_html(cell) for cell in assessment.favours)
    favours = (
        f'<p class="profile-label">Favours — {assessment.measured_favours} of '
        f"{len(assessment.favours)} measured</p>"
        f'<ul class="cells">{favour_cells}</ul>'
        if assessment.favours
        else ""
    )
    score = (
        f' · score {assessment.score:.2f}'
        if assessment.score is not None and assessment.outcome == "fits"
        else ""
    )
    return (
        f'<details class="prof o-{assessment.outcome}" id="p-{_escape(profile.id)}"{opened}>'
        f'<summary><span class="prof-top"><span class="prof-title">{_escape(profile.title)}</span>'
        f'<span class="badge b-{assessment.outcome}">{_escape(assessment.headline)}</span>'
        '<span class="f-caret"></span></span></summary>'
        f'<div class="prof-body"><p class="prof-summary">{_escape(profile.summary)}</p>'
        f'<ul class="cells">{"".join(_cell_html(cell) for cell in assessment.gates)}</ul>'
        f"{favours}"
        f'<p class="prof-meta">{score.lstrip(" ·")}'
        f'<a href="/profiles#profile-{_escape(profile.id)}">definition</a> · '
        f'catalog v{assessment.catalog_version}</p>'
        "</div></details>"
    )


def _verdict_html(corpus: Corpus, assessments: Sequence[profiles.Assessment]) -> str:
    winner = profiles.recommendation(assessments)
    if winner is None:
        blocked = [
            cell
            for assessment in assessments
            if assessment.outcome == "unknown"
            for cell in assessment.unknown
        ]
        detail = (
            "Nothing measured disqualifies a profile outright; what is missing is "
            + ", ".join(sorted({f"<code>{_escape(cell.criterion.fact)}</code>" for cell in blocked}))
            if blocked
            else "Every profile in the catalogue is disqualified by a measured fact."
        )
        return (
            '<section class="brief-verdict"><p class="brief-label">What to run on this corpus</p>'
            "<h1>Undecided — measure before building</h1>"
            f'<p>{detail}</p></section>'
        )

    facts_by_name = {fact.name: fact for fact in corpus.facts}
    gate_readings = " ".join(
        f"{_escape(facts_by_name[cell.criterion.fact].observed)}."
        for cell in winner.gates
        if cell.criterion.fact in facts_by_name
    )
    routing = _routing_note(corpus, winner)
    return (
        '<section class="brief-verdict"><p class="brief-label">What to run on this corpus</p>'
        f"<h1>{_escape(winner.profile.title)}</h1>"
        f"<p>{_escape(winner.profile.summary)}</p>"
        f'<p class="verdict-evidence">{gate_readings}</p>'
        f"{routing}"
        f'<p class="profile-meta">fits {winner.passed} of {winner.profile.gate_total()} gates · '
        f"catalog v{winner.catalog_version}</p></section>"
    )


def _routing_note(corpus: Corpus, winner: profiles.Assessment) -> str:
    """What to do with the documents the winner does not cover."""
    outside = tuple(
        document
        for document in corpus.documents
        if document.kind == "unknown" or document.kind == "plate"
    )
    if not outside or winner.profile.id == "visual-search":
        return ""
    plates = tuple(document for document in outside if document.kind == "plate")
    unknown = tuple(document for document in outside if document.kind == "unknown")
    pieces = []
    if plates:
        names = ", ".join(_escape(document.name) for document in plates)
        pieces.append(
            f"{len(plates)} of {len(corpus.documents)} documents "
            f"{'is an image-plate book' if len(plates) == 1 else 'are image-plate books'} "
            f"({names}): route {'it' if len(plates) == 1 else 'them'} to a vision pass, "
            "because a text pipeline will describe the layout instead of the content."
        )
    if unknown:
        pieces.append(
            f"{len(unknown)} document{'s' if len(unknown) > 1 else ''} "
            f"{'has' if len(unknown) == 1 else 'have'} no word count yet, so "
            f"{'it is' if len(unknown) == 1 else 'they are'} not yet routed."
        )
    return f'<p class="verdict-routing">{_escape(" ".join(pieces))}</p>'


def _next_measurements(assessments: Sequence[profiles.Assessment]) -> str:
    """Gate criteria nobody has measured, as an agenda with a tool on each."""
    wanted: dict[str, str | None] = {}
    for assessment in assessments:
        if assessment.outcome == "disqualified":
            continue
        for cell in assessment.unknown:
            wanted.setdefault(cell.criterion.fact, cell.tool)
    if not wanted:
        return ""
    items = "".join(
        f'<li><code>{_escape(fact)}</code>'
        + (f' <span class="brief-cite">{_escape(tool)}</span>' if tool else "")
        + "</li>"
        for fact, tool in sorted(wanted.items())
    )
    return (
        '<section class="brief-open"><h2>What is still unmeasured</h2>'
        '<p class="brief-note">These are gate criteria on profiles the corpus has not been '
        "disqualified from. An unknown never rules a profile out — it is the next session's "
        "question, and it comes with the instrument that answers it.</p>"
        f'<ul class="tasks">{items}</ul></section>'
    )


def body(corpus: Corpus, assessments: Sequence[profiles.Assessment] | None = None) -> str:
    """The brief: verdict, the whole matrix, the documents, the plan."""
    scored = tuple(assessments) if assessments is not None else profiles.evaluate(corpus)
    winner = profiles.recommendation(scored)
    parts = [_verdict_html(corpus, scored)]

    parts.append('<section class="matrix"><h2>Every profile, scored</h2>')
    parts.append(
        '<p class="brief-note">All of them, including the ones this corpus rules out: '
        '<a href="/profiles">what each profile is</a>. '
        "<strong>✓</strong> measured and true · <strong>✗</strong> measured and false · "
        "<strong>?</strong> nobody measured it, which never disqualifies.</p>"
    )
    parts.extend(_assessment_html(assessment) for assessment in scored)
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

    parts.append(_next_measurements(scored))

    if winner is not None:
        parts.append('<section class="brief-do"><h2>First moves</h2><ol>')
        parts.extend(f"<li>{_escape(move)}</li>" for move in winner.profile.moves)
        parts.append("</ol></section>")

        parts.append('<section class="brief-not"><h2>Do not</h2><ul>')
        parts.extend(f"<li>{_escape(item)}</li>" for item in winner.profile.not_to_do)
        parts.append("</ul></section>")

        parts.append('<section class="brief-open"><h2>What would change this</h2><ul>')
        parts.extend(f"<li>{_escape(item)}</li>" for item in winner.profile.falsifiers)
        parts.append("</ul></section>")

    parts.append(
        '<section class="report-notes"><h2>Where this comes from</h2>'
        "<p>Every reading above is a line in <code>findings.jsonl</code>, and the citations name "
        "the line; the profiles are the catalogue at <a href=\"/profiles\">/profiles</a>. Nothing "
        "here is a measurement of its own — the evidence is the report.</p></section>"
    )
    return "".join(parts)


def render_brief(
    corpus: Corpus,
    *,
    dataset_id: str | None = None,
    assessments: Sequence[profiles.Assessment] | None = None,
) -> str:
    """The whole brief, as one styled document."""
    title = f"brief — {dataset_id}" if dataset_id else "brief"
    meta = f"{dataset_id} · what to run on it · catalog v{profiles.CATALOGUE_VERSION}"
    return render.html_document(body(corpus, assessments), title=title, meta=meta)


def write_brief(dataset_id: str, ledger: Ledger) -> Any:
    """Read the ledger for a dataset, write ``brief-<dataset>.html``, return its path."""
    corpus = read(ledger.findings())
    target = workspace.artifact_path(dataset_id, f"brief-{dataset_id}.html")
    target.write_text(render_brief(corpus, dataset_id=dataset_id), encoding="utf-8")
    return target


def index_record(dataset_id: str) -> dict[str, Any] | None:
    """The machine-readable half of the index's call to action.

    The same claim the button makes, for a script: which profile leads, how it
    scored, and what every other profile came out as, so "what did the catalogue
    say about this dataset" needs no HTML parsing.
    """
    path = workspace.ledger_path(dataset_id)
    if not path.is_file():
        return None
    try:
        corpus = read(Ledger(path).findings())
    except (OSError, ValueError):
        # The index is the page a reader uses when something is wrong, so a
        # ledger it cannot parse must degrade to "no brief", never to a 500.
        return None
    assessments = profiles.evaluate(corpus)
    winner = profiles.recommendation(assessments)
    return {
        "catalog_version": profiles.CATALOGUE_VERSION,
        "primary_profile": winner.profile.id if winner else None,
        "title": winner.profile.title if winner else None,
        "headline": winner.headline if winner else "undecided",
        "gates_passed": winner.passed if winner else 0,
        "gates_total": winner.profile.gate_total() if winner else 0,
        "outcomes": {assessment.profile.id: assessment.outcome for assessment in assessments},
        "unmeasured": sorted(
            {
                cell.criterion.fact
                for assessment in assessments
                if assessment.outcome != "disqualified"
                for cell in assessment.unknown
            }
        ),
    }


def headline(dataset_id: str) -> str | None:
    """``literature mining (fits 3/3)`` for the index's call to action, or None.

    The index should not say "what to run on this dataset" and then make you tap to
    find out; a decision is one line, and that line fits on a button.
    """
    path = workspace.ledger_path(dataset_id)
    try:
        corpus = read(Ledger(path).findings())
    except (OSError, ValueError):
        return None
    winner = profiles.recommendation(profiles.evaluate(corpus))
    if winner is None:
        return None
    short = winner.profile.title.split(" — ")[0]
    return f"{short} ({winner.headline})"


def main(argv: Sequence[str] | None = None) -> int:
    """``brief-render <dataset-id> [--for <profile>]`` — the decision, from the ledger."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="brief-render", description="What to run on a dataset.")
    parser.add_argument("dataset", nargs="?", help="dataset id; defaults to DATA_SCOUT_DATASET_ID")
    parser.add_argument(
        "--for",
        dest="profile",
        metavar="PROFILE",
        help="score one profile and report only that, by id (e.g. corpus-qa)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    dataset_id = args.dataset or workspace.current_dataset_id()
    ledger_path = workspace.ledger_path(dataset_id)
    if not ledger_path.is_file():
        print(f"brief-render: no ledger at {ledger_path}", file=sys.stderr)
        return 2

    corpus = read(Ledger(ledger_path).findings())
    if args.profile:
        chosen = [profile for profile in profiles.CATALOGUE if profile.id == args.profile]
        if not chosen:
            known = ", ".join(profile.id for profile in profiles.CATALOGUE)
            print(f"brief-render: no profile {args.profile!r}; known: {known}", file=sys.stderr)
            return 2
        assessment = profiles.evaluate(corpus, chosen)[0]
        if not args.quiet:
            print(f"{assessment.profile.id}: {assessment.headline}")
            for cell in (*assessment.gates, *assessment.favours):
                mark = {"pass": "ok  ", "fail": "FAIL", "unknown": "?   "}[cell.state]
                print(f"  {mark} {cell.criterion.describe():28} {cell.observed or 'not measured'}")
        return 0

    written = write_brief(dataset_id, Ledger(ledger_path))
    if not args.quiet:
        winner = profiles.recommendation(profiles.evaluate(corpus))
        print(f"brief-render: {winner.profile.title if winner else 'undecided'}")
        print(f"brief-render: wrote {written}")
    return 0
