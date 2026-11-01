"""The profile catalogue — interpretation as an evaluation, not a label.

The belt answers questions and the ledger records the answers without judging
them. This module is the *next* layer, and it is deliberately a different kind of
thing: a closed, versioned set of use-cases, each a set of declarative tests over
named measured facts.

Why a catalogue rather than a branch:

* a verdict picked from an open set of systems cannot be wrong, so it cannot be
  trusted. A catalogue makes the conclusion checkable — "fits 3 of 3 gates, fails
  the text layer" is an argument a reader can have;
* the refusals become visible. A corpus is as interesting for the six profiles it
  disqualifies as for the one it fits, and a single label shows none of them;
* every criterion names the instrument that answers it, so a *missing*
  measurement is a task with a tool attached rather than an apology.

The rule that keeps it honest is the tri-state: **pass, fail, or unknown — and
unknown never disqualifies.** "We measured it and it is wrong" is a finding;
"nobody measured it" is the next session's agenda. Collapsing the two is how a
brief becomes vague.

Decision recorded in `data-scout-profile-catalogue-design` §7: the catalogue is
**code**, not YAML shipped as data — predicates are small enough to review in the
same PR as the thresholds they encode, and a declarative file that needs a
validator is a programming language with worse error messages. Moving it to data
later changes where `CATALOGUE` is read from, nothing else.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from data_scout import render

if TYPE_CHECKING:  # pragma: no cover - import cycle kept out of runtime
    from data_scout.brief import Corpus

#: The catalogue's own version. Stamped on every brief, so a matrix can be
#: reproduced: the ledger it read and the rules it applied are both citable.
CATALOGUE_VERSION = 1

State = Literal["pass", "fail", "unknown"]
Outcome = Literal["fits", "partial", "disqualified", "unknown"]

_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
}
_TEST = re.compile(r"^\s*(<=|>=|==|!=|<|>|=)\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class Criterion:
    """One declarative test: a named fact, a comparison, and who could answer it.

    ``fact`` is either a corpus fact (``table_share``) or a per-document fact
    (``images_per_page``). Which one it is, is declared by the profile's ``scope``
    — never guessed at evaluation time, because "this corpus has 40 images a page"
    and "every document has 40 images a page" are different claims.
    """

    fact: str
    test: str
    weight: int = 1
    tool: str | None = None

    def parsed(self) -> tuple[str, float]:
        match = _TEST.match(self.test)
        if match is None:
            raise ValueError(f"unreadable criterion test {self.test!r} for fact {self.fact!r}")
        return match.group(1), float(match.group(2))

    def holds(self, value: float) -> bool:
        op, threshold = self.parsed()
        return _OPERATORS[op](value, threshold)

    def describe(self) -> str:
        """How the test reads on a page: ``table_share >= 0.5``."""
        op, threshold = self.parsed()
        shown = f"{threshold:,.0f}" if threshold.is_integer() else f"{threshold:g}"
        return f"{self.fact} {op} {shown}"


@dataclass(frozen=True)
class Profile:
    """One use-case: what it is for, what it needs, and what it costs to be wrong."""

    id: str
    title: str
    use_case: str
    summary: str
    gates: tuple[Criterion, ...]
    favours: tuple[Criterion, ...] = ()
    #: A document-level test that, when any single document satisfies it, turns a
    #: disqualified corpus into a *partial* one: the artbook that is out of scope
    #: for a prose pipeline is still a real job for a vision one.
    partial: tuple[Criterion, ...] = ()
    moves: tuple[str, ...] = ()
    not_to_do: tuple[str, ...] = ()
    falsifiers: tuple[str, ...] = ()

    def gate_total(self) -> int:
        return len(self.gates)


@dataclass(frozen=True)
class Cell:
    """One criterion, evaluated: what was seen, where it came from, what is missing."""

    criterion: Criterion
    state: State
    observed: str | None = None
    lines: tuple[int, ...] = ()
    documents: tuple[str, ...] = ()

    @property
    def tool(self) -> str | None:
        return self.criterion.tool


@dataclass(frozen=True)
class Assessment:
    """One profile scored against one corpus."""

    profile: Profile
    gates: tuple[Cell, ...]
    favours: tuple[Cell, ...] = ()
    outcome: Outcome = "unknown"
    score: float | None = None
    partial_documents: tuple[str, ...] = ()
    catalog_version: int = CATALOGUE_VERSION

    @property
    def passed(self) -> int:
        return sum(1 for cell in self.gates if cell.state == "pass")

    @property
    def failed(self) -> tuple[Cell, ...]:
        return tuple(cell for cell in self.gates if cell.state == "fail")

    @property
    def unknown(self) -> tuple[Cell, ...]:
        return tuple(cell for cell in self.gates if cell.state == "unknown")

    @property
    def measured_favours(self) -> int:
        return sum(1 for cell in self.favours if cell.state != "unknown")

    @property
    def headline(self) -> str:
        """How the outcome reads on a card: ``fits 3/3``, ``unknown — 2 to measure``."""
        if self.outcome == "fits":
            return f"fits {self.passed}/{self.profile.gate_total()}"
        if self.outcome == "partial":
            count = len(self.partial_documents)
            return f"partial — {count} document{'' if count == 1 else 's'}"
        if self.outcome == "disqualified":
            return "disqualified"
        return f"unknown — {len(self.unknown)} to measure"


# --------------------------------------------------------------------------
# the v1 catalogue
# --------------------------------------------------------------------------

CATALOGUE: tuple[Profile, ...] = (
    Profile(
        id="literature-mining",
        title="Literature mining — themes, quotes, and what a chapter says",
        use_case="mining",
        summary=(
            "A corpus of long-form prose read for what it says: themes, named things, quotes worth "
            "keeping, each traceable to a page."
        ),
        gates=(
            Criterion("words_per_page_median", ">= 150", tool="partition_doc"),
            Criterion("text_layer_share", ">= 0.8", tool="pdf_forensics"),
            Criterion("table_share", "< 0.5", tool="structure_mix"),
        ),
        favours=(
            Criterion("corpus_words", ">= 50000", weight=2, tool="partition_doc"),
            Criterion("narrative_share_median", ">= 0.5", weight=2, tool="structure_mix"),
            Criterion("documents", "<= 200", tool="frame"),
        ),
        partial=(
            Criterion("words_per_page", ">= 150", tool="partition_doc"),
            # For a document with no page count (an EPUB): a chapter of real prose.
            Criterion("words_per_chapter", ">= 1500", tool="partition_doc"),
        ),
        moves=(
            "Parse one book end to end and check the section structure by eye before trusting the "
            "chunker.",
            "Chunk by section, embed, then ask three questions the book plainly answers — themes, "
            "a named subject, a specific claim.",
            "Keep a page citation on every quote and theme, or the output is not checkable.",
        ),
        not_to_do=(
            "Do not build a warehouse: prose does not aggregate.",
            "Do not summarise the whole corpus at once — a summary of eight books is a summary of "
            "none of them.",
            "Do not trust a parse that returns an empty page; add a whole-page text fallback where "
            "the parser and a raw extractor disagree.",
        ),
        falsifiers=(
            "If the parser's narrative share comes back low, these are plate books with text "
            "furniture and the vision profile is the real one.",
            "If the corpus is a few hundred pages, an agent reading it whole beats a vector index.",
        ),
    ),
    Profile(
        id="corpus-qa",
        title="Corpus QA — ask questions, get answers with page citations",
        use_case="retrieval",
        summary=(
            "A question-answering corpus: retrieve the passage that answers the question and cite "
            "the page it came from."
        ),
        gates=(
            Criterion("text_layer_share", ">= 0.8", tool="pdf_forensics"),
            Criterion("table_share", "< 0.5", tool="structure_mix"),
        ),
        favours=(
            Criterion("words_per_page_median", ">= 150", weight=2, tool="partition_doc"),
            Criterion("corpus_words", "<= 5000000", weight=1, tool="partition_doc"),
            Criterion("dup_share", "<= 0.2", weight=1, tool="near_dup_docs"),
        ),
        partial=(Criterion("words_per_page", ">= 150", tool="partition_doc"),),
        moves=(
            "Index a sample of about a hundred pages and grade twenty real questions before "
            "indexing the rest.",
            "Score the answers for 'cites the right page' as well as 'sounds right' — on a prose "
            "corpus the second is nearly free and nearly worthless.",
            "Keep the retrieval unit large enough to answer: a paragraph that answers a question "
            "rarely contains the question's words.",
        ),
        not_to_do=(
            "Do not index page images with no text layer: retrieval over pixels is a different "
            "profile.",
            "Do not build a fine-tuned model before measuring what plain retrieval gets wrong.",
        ),
        falsifiers=(
            "If the questions you actually have are about one subject, a lookup table beats an "
            "index.",
            "If retrieval fails on questions the text plainly answers, the chunking is wrong, not "
            "the model.",
        ),
    ),
    Profile(
        id="visual-search",
        title="Vision over page images — describe the plates, then search them",
        use_case="vision",
        summary=(
            "A picture corpus: render pages to images, have a vision model say what is on them, and "
            "retrieve over those descriptions."
        ),
        gates=(
            # A strict majority: half plate books and half prose is two jobs routed
            # per document, which the `partial` rule below reports as such.
            Criterion("plate_share", "> 0.5", tool="pdf_forensics"),
            Criterion("images_per_page_max", ">= 5", tool="pdf_forensics"),
        ),
        favours=(
            Criterion("images_per_page_max", ">= 20", weight=2, tool="pdf_forensics"),
            Criterion("text_layer_share", "<= 0.5", weight=1, tool="pdf_forensics"),
        ),
        partial=(Criterion("images_per_page", ">= 5", tool="pdf_forensics"),),
        moves=(
            "Render a 20-page sample at 200 DPI and caption it; read the captions yourself before "
            "paying for the rest.",
            "Decide the unit — page, spread, or single image — before batch processing, because a "
            "spread is often the artwork.",
            "Embed the captions, not the pixels: captions are searchable text and pixels are not.",
        ),
        not_to_do=(
            "Do not summarise the extractable text layer: on a plate book it is captions and page "
            "furniture, and the summary describes the layout.",
        ),
        falsifiers=(
            "If the existing captions carry the meaning, a cheap text pipeline over them wins.",
            "If the plates are decorative — repeated motifs, no per-page meaning — the corpus is "
            "smaller than it looks.",
        ),
    ),
    Profile(
        id="warehouse",
        title="Structured analytics — load it into SQL and let the questions be queries",
        use_case="analytics",
        summary=(
            "A row-shaped corpus: tables that carry text and numbers, loaded into a table per "
            "document family with a thin BI layer on top."
        ),
        gates=(Criterion("table_share", ">= 0.5", tool="structure_mix"),),
        favours=(
            Criterion("table_share", ">= 0.8", weight=2, tool="structure_mix"),
            Criterion("dup_share", "<= 0.2", weight=1, tool="near_dup_docs"),
        ),
        moves=(
            "Score the tables for header regularity first: a table with merged cells is a "
            "paragraph drawn with lines.",
            "Load one document family end to end and try three real questions against it.",
            "Keep the raw cell text beside every column so a wrong parse is visible, not silent.",
        ),
        not_to_do=(
            "Do not start with a document embedding pipeline: retrieval answers different "
            "questions than a row-shaped corpus invites.",
        ),
        falsifiers=(
            "If headers turn out to be irregular across documents, the schema is a document store "
            "with a table extractor.",
            "If one family dominates and the rest are noise, load only that family.",
        ),
    ),
    Profile(
        id="records-extraction",
        title="Records extraction — documents into rows of JSON",
        use_case="extraction",
        summary=(
            "Forms and semi-structured documents whose value is the fields: extract key/value pairs "
            "into records a system can use."
        ),
        gates=(
            Criterion("table_share", ">= 0.2", tool="structure_mix"),
            Criterion("words_per_page_median", "<= 300", tool="partition_doc"),
        ),
        favours=(
            Criterion("table_share", ">= 0.4", weight=2, tool="structure_mix"),
            Criterion("narrative_share_median", "<= 0.5", weight=1, tool="structure_mix"),
        ),
        moves=(
            "Hand-label fifty documents before automating one: the field list is the deliverable.",
            "Measure per-document field agreement against those labels, not aggregate accuracy.",
            "Route low-confidence extractions to a person rather than guessing.",
        ),
        not_to_do=("Do not extract fields from prose you have not measured as form-like.",),
        falsifiers=(
            "If the same field appears under different headings across documents, this is a "
            "normalisation project, not an extraction one.",
        ),
    ),
    Profile(
        id="searchable-archive",
        title="Searchable archive — make it findable and stop",
        use_case="archive",
        summary=(
            "The floor: a corpus with text or usable metadata, given stable identities and a plain "
            "full-text index. Every corpus passes this, which is why it is shown."
        ),
        gates=(Criterion("documents", ">= 1", tool="frame"),),
        favours=(Criterion("corpus_words", ">= 10000", tool="partition_doc"),),
        moves=(
            "Index the extractable text and the file names; that is most of the findability.",
            "Keep a stable id per document so a link survives a re-import.",
        ),
        not_to_do=("Do not call this analysis: retrieval over an index of names answers names.",),
        falsifiers=(
            "If the questions need meaning rather than words, a full-text index will fail them and "
            "the QA profile is the real one.",
        ),
    ),
    Profile(
        id="pii-inventory",
        title="Compliance inventory — what personal data is in here",
        use_case="compliance",
        summary=(
            "A corpus that may hold personal data: find it, count it per document, and record what "
            "was searched for."
        ),
        gates=(
            Criterion("entity_density", "> 0"),  # nothing in the belt measures this yet
            Criterion("documents", ">= 1", tool="frame"),
        ),
        moves=(
            "Name the entity classes before measuring them, and record the class list with the "
            "number.",
            "Report per document, not per corpus: a corpus average hides the one file that matters.",
        ),
        not_to_do=(
            "Do not report a zero from an instrument that never ran — that is the defect the "
            "tri-state exists to prevent.",
        ),
        falsifiers=(
            "If entity density is measured at zero by an instrument that provably works, this "
            "profile stops being the reason to do anything.",
        ),
    ),
    Profile(
        id="transcription",
        title="Transcription — audio and video into text",
        use_case="transcription",
        summary=(
            "Spoken content: transcribe, diarise, then treat the transcripts as the corpus. "
            "Included so the catalogue can say no out loud."
        ),
        gates=(Criterion("audio_video_share", ">= 0.5", tool="file_census"),),
        moves=(
            "Measure the media mix first: a corpus with one recording and a thousand PDFs is not "
            "this profile.",
            "Transcribe 10 minutes before transcribing ten hours.",
        ),
        not_to_do=("Do not transcribe documents.",),
        falsifiers=("If the media share is small, this is an appendix to another profile.",),
    ),
)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def _document_matches(corpus: Corpus, criterion: Criterion) -> tuple[str, ...]:
    """Documents satisfying a document-level criterion, in first-seen order."""
    matches: list[str] = []
    for document in corpus.documents:
        value = document.value(criterion.fact)
        if value is not None and criterion.holds(value):
            matches.append(document.document)
    return tuple(matches)


def _cell(corpus: Corpus, criterion: Criterion) -> Cell:
    """Evaluate one criterion, and keep the reading it rests on.

    A reading with no ledger line cannot pass or fail: an uncited number is a
    claim about nobody's data, so it is treated as *unknown* — the honest state.
    """
    if criterion.fact in corpus.per_document_facts():
        matches = _document_matches(corpus, criterion)
        measured = [
            document for document in corpus.documents if document.value(criterion.fact) is not None
        ]
        if not measured:
            return Cell(criterion, "unknown")
        share = len(matches)
        # Every document that was read answers the question, whether it matched
        # or not: a fail is a reading too, and it must be as traceable as a pass.
        lines = tuple(sorted({line for document in measured for line in document.lines}))
        return Cell(
            criterion,
            "pass" if share else "fail",
            observed=f"{share} of {len(corpus.documents)} documents",
            lines=lines,
            documents=matches,
        )

    fact = corpus.fact(criterion.fact)
    if fact is None or not isinstance(fact.value, int | float) or not fact.lines:
        return Cell(criterion, "unknown")
    value = float(fact.value)
    return Cell(
        criterion,
        "pass" if criterion.holds(value) else "fail",
        observed=fact.observed,
        lines=fact.lines,
    )


def _score(cells: Sequence[Cell]) -> float | None:
    """Weighted share of the measured favours. None when nothing was measured."""
    measured = [cell for cell in cells if cell.state != "unknown"]
    if not measured:
        return None
    total = sum(cell.criterion.weight for cell in measured)
    won = sum(cell.criterion.weight for cell in measured if cell.state == "pass")
    return round(won / total, 3) if total else None


def evaluate_one(corpus: Corpus, profile: Profile) -> Assessment:
    gates = tuple(_cell(corpus, criterion) for criterion in profile.gates)
    favours = tuple(_cell(corpus, criterion) for criterion in profile.favours)

    if any(cell.state == "fail" for cell in gates):
        partial: tuple[str, ...] = ()
        for criterion in profile.partial:
            found = _document_matches(corpus, criterion)
            if found:
                partial = found
                break
        return Assessment(
            profile=profile,
            gates=gates,
            favours=favours,
            outcome="partial" if partial else "disqualified",
            score=_score(favours),
            partial_documents=partial,
        )

    if any(cell.state == "unknown" for cell in gates):
        return Assessment(
            profile=profile, gates=gates, favours=favours, outcome="unknown", score=None
        )

    return Assessment(
        profile=profile, gates=gates, favours=favours, outcome="fits", score=_score(favours)
    )


#: Facts the catalogue asks for that no instrument in the belt answers. They are
#: named here rather than given a plausible-looking tool name: an "unknown" that
#: points at a tool which does not exist is worse than one that says so.
UNOWNED_FACTS = frozenset({"entity_density"})

#: The smallest minority that is worth its own run. Below this, a document that
#: goes another way is an exception to route, not a corpus to split: one artbook
#: in eight does not make the other seven a different dataset.
SPLIT_MINORITY_SHARE = 0.25


@dataclass(frozen=True)
class DocumentRoute:
    """One document, and the profile whose per-document test it satisfies.

    A profile's ``partial`` criteria are exactly the tests that can be answered
    *for one document* — "is this book word-dense enough to read as prose?" — so
    they are what decides where a single document goes. When no profile's test
    holds, the document is unrouted, and that is a statement about the catalogue,
    not a defect in the document.
    """

    document: str
    profile: Profile | None
    cells: tuple[Cell, ...] = ()

    @property
    def outcome(self) -> str:
        return "unrouted" if self.profile is None else self.profile.id


def _document_cell(document: Any, criterion: Criterion) -> Cell:
    value = document.value(criterion.fact)
    if value is None:
        return Cell(criterion, "unknown", lines=document.lines)
    return Cell(
        criterion,
        "pass" if criterion.holds(float(value)) else "fail",
        observed=f"{float(value):,.0f}" if abs(float(value)) >= 100 else f"{float(value):g}",
        lines=document.lines,
    )


def document_routes(
    corpus: Corpus, catalogue: Sequence[Profile] = CATALOGUE
) -> tuple[DocumentRoute, ...]:
    """Route each document to the first profile whose per-document test it meets.

    Catalogue order is the precedence: the more specific use-case asks first, so a
    word-dense book with pictures is a literature-mining document rather than a
    vision one. A document no profile claims is reported as unrouted rather than
    forced into the nearest fit.
    """
    routes: list[DocumentRoute] = []
    for document in corpus.documents:
        route = DocumentRoute(document=document.name, profile=None)
        for profile in catalogue:
            if not profile.partial:
                continue
            cells = tuple(_document_cell(document, criterion) for criterion in profile.partial)
            # A criterion the format cannot answer is not a failure: an EPUB has no
            # page count, so `words_per_page` is *not applicable* there rather than
            # false. At least one measured test must pass, and none may fail.
            measured = [cell for cell in cells if cell.state != "unknown"]
            if measured and all(cell.state == "pass" for cell in measured):
                route = DocumentRoute(document=document.name, profile=profile, cells=cells)
                break
        routes.append(route)
    return tuple(routes)


def routing_counts(routes: Sequence[DocumentRoute]) -> dict[str, int]:
    """How many documents each profile would serve, in first-seen order."""
    counts: dict[str, int] = {}
    for route in routes:
        counts[route.outcome] = counts.get(route.outcome, 0) + 1
    return counts


def split_advice(routes: Sequence[DocumentRoute]) -> str:
    """One sentence: is this corpus one job, two runs, or one exception?

    The question the routing table cannot answer by itself. Below
    :data:`SPLIT_MINORITY_SHARE` the minority is routed as an exception, because
    splitting a corpus costs a second pipeline, a second index and a join.
    """
    counts = routing_counts(routes)
    if len([key for key in counts if key != "unrouted"]) < 2:
        return "One profile serves every profiled document: no split is warranted."
    total = len(routes)
    leader, leader_count = max(counts.items(), key=lambda item: item[1])
    minority = total - leader_count
    share = minority / total if total else 0.0
    listing = ", ".join(f"{count} {key}" for key, count in counts.items())
    if share >= SPLIT_MINORITY_SHARE:
        return (
            f"Worth splitting: {minority} of {total} documents ({share:.0%}) go a different "
            f"way from the {leader_count} that {'is' if leader_count == 1 else 'are'} {leader} "
            f"— the split is {listing}. Run them as two corpora and join the results, "
            "rather than averaging them here."
        )
    return (
        f"Not worth splitting: {leader_count} of {total} documents are {leader}, and the "
        f"{minority} that {'is' if minority == 1 else 'are'} not "
        f"({share:.0%}) {'is an exception to route' if minority == 1 else 'are exceptions to route'}, "
        "not a second corpus."
    )


def evaluate(corpus: Corpus, catalogue: Sequence[Profile] = CATALOGUE) -> tuple[Assessment, ...]:
    """Score every profile against one corpus, in catalogue order.

    Order is preserved on purpose: the brief shows all of them, and a reader
    comparing two datasets should see the same rows in the same sequence.
    """
    return tuple(evaluate_one(corpus, profile) for profile in catalogue)


def recommendation(assessments: Sequence[Assessment]) -> Assessment | None:
    """The profile to lead with: most gates passed, then best weighted favours.

    Ties are broken towards the profile with more gates, because a profile that
    asks more questions and answers them all is a stronger claim than one that
    asks only whether anything exists.
    """
    fitting = [assessment for assessment in assessments if assessment.outcome == "fits"]
    if not fitting:
        return None
    return max(
        fitting,
        key=lambda assessment: (
            assessment.passed,
            assessment.score or 0.0,
            assessment.profile.gate_total(),
        ),
    )


# --------------------------------------------------------------------------
# rendering the catalogue itself (`/profiles`)
# --------------------------------------------------------------------------


def _escape(text: str) -> str:
    return render._escape(text)


def _criterion_html(criterion: Criterion, *, kind: str) -> str:
    tool = f'<span class="brief-cite">{_escape(criterion.tool)}</span>' if criterion.tool else ""
    weight = (
        f'<span class="brief-cite">weight {criterion.weight}</span>'
        if kind == "favour" and criterion.weight != 1
        else ""
    )
    return (
        f'<li class="crit crit-{kind}"><code>{_escape(criterion.describe())}</code>'
        f"{tool}{weight}</li>"
    )


def profile_card(profile: Profile) -> str:
    """One profile as a reviewable card: what it needs, and what would settle it."""
    gates = "".join(_criterion_html(criterion, kind="gate") for criterion in profile.gates)
    favours = "".join(_criterion_html(criterion, kind="favour") for criterion in profile.favours)
    partial = "".join(_criterion_html(criterion, kind="partial") for criterion in profile.partial)
    parts = [
        f'<section class="card profile" id="profile-{_escape(profile.id)}">',
        f'<h3>{_escape(profile.title)}</h3>',
        f'<p class="profile-summary">{_escape(profile.summary)}</p>',
        '<div class="profile-grid">',
        f'<div><p class="profile-label">Must hold</p><ul class="criteria">{gates}</ul></div>',
    ]
    if favours:
        parts.append(
            f'<div><p class="profile-label">Favours</p><ul class="criteria">{favours}</ul></div>'
        )
    if partial:
        parts.append(
            f'<div><p class="profile-label">Counts as partial when a single document holds</p>'
            f'<ul class="criteria">{partial}</ul></div>'
        )
    parts.append("</div>")

    # The advice belongs to the profile, not to a dataset: what to do first, what
    # not to do, and what would overturn the call. It lives on this page, collapsed,
    # so the brief can stop at the decision.
    advice: list[str] = []
    for label, kind, items in (
        ("First moves", "ol", profile.moves),
        ("Do not", "ul", profile.not_to_do),
        ("What would overturn this", "ul", profile.falsifiers),
    ):
        if items:
            body = "".join(f"<li>{_escape(item)}</li>" for item in items)
            advice.append(f'<p class="profile-label">{label}</p><{kind}>{body}</{kind}>')
    if advice:
        parts.append(
            '<details class="profile-advice">'
            "<summary>How to run it, and what would overturn it</summary>"
            f'{"".join(advice)}</details>'
        )

    parts.append(
        f'<p class="profile-meta"><code>{_escape(profile.id)}</code> · '
        f'{_escape(profile.use_case)} · catalog v{CATALOGUE_VERSION}</p>'
    )
    parts.append("</section>")
    return "".join(parts)


def render_catalogue(catalogue: Sequence[Profile] = CATALOGUE) -> str:
    """The ruler, without a corpus: every profile, its gates, and who answers them."""
    intro = (
        "<p>The interpretation layer. Each profile is a use-case with a consumer, and each line "
        "under it is a test over a <em>named</em> measured fact — with the instrument that would "
        "answer it. A corpus is scored against all of them, and the brief shows every result: "
        "a profile that is out of scope is information, not a failure to report.</p>"
        '<p class="profile-rule"><strong>Pass</strong> — measured and true. '
        "<strong>Fail</strong> — measured and false. <strong>Unknown</strong> — nobody measured it; "
        "this never disqualifies a profile, it names the next measurement.</p>"
    )
    cards = "".join(profile_card(profile) for profile in catalogue)
    body = (
        '<section class="brief-verdict"><p class="brief-label">The profile catalogue</p>'
        f"<h1>What a corpus can be for</h1>"
        f'<p class="profile-meta">catalog v{CATALOGUE_VERSION} · {len(catalogue)} profiles</p></section>'
        f'<section class="profile-intro">{intro}</section>'
        f"{cards}"
    )
    return render.html_document(body, title="profiles", meta="the interpretation layer")
