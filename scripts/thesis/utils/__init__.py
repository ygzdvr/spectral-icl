"""Shared thesis utilities.

Sub-modules are imported explicitly by callers (not re-exported here) to avoid
import-order hazards and to keep the surface area of each utility visible to
readers.

Canonical usage::

    from scripts.thesis.utils.run_metadata import RunContext, ThesisRunDir
    from scripts.thesis.utils.plotting import apply_thesis_style, save_both
"""
