"""The instrument belt — fifteen tools, one module each, dispatched by the registry.

Import from here, not from the submodules, so a promoted finding's ``script``
(``data_scout.tools.candidate_density``) resolves to the same code a notebook
calls::

    from data_scout.tools import candidate_density, structure_mix

One implementation per instrument (design §2.2). Do not add a sixteenth: the
belt is capped at fifteen and design §10's pruning rule is the only headroom
there is.
"""

from __future__ import annotations

from data_scout.tools.candidate_density import candidate_density
from data_scout.tools.cluster_topics import cluster_topics
from data_scout.tools.embed_sample import embed_sample
from data_scout.tools.exif_census import exif_census
from data_scout.tools.file_census import file_census
from data_scout.tools.findings_append import append_many, findings_append
from data_scout.tools.image_inventory import image_inventory
from data_scout.tools.measurement_register import measurement_register
from data_scout.tools.near_dup_docs import near_dup_docs
from data_scout.tools.near_dup_images import near_dup_images
from data_scout.tools.partition_doc import partition_doc
from data_scout.tools.pdf_forensics import pdf_forensics
from data_scout.tools.sample import sample, write_sample_anchor
from data_scout.tools.structure_mix import structure_mix
from data_scout.tools.text_quality import text_quality

__all__ = [
    "candidate_density",
    "cluster_topics",
    "embed_sample",
    "exif_census",
    "file_census",
    "findings_append",
    "append_many",
    "image_inventory",
    "measurement_register",
    "near_dup_docs",
    "near_dup_images",
    "partition_doc",
    "pdf_forensics",
    "sample",
    "write_sample_anchor",
    "structure_mix",
    "text_quality",
]
