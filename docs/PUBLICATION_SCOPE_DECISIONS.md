# Public export scope and provenance decisions

This is a curated, new history-free export created from the author's private research archive.
**The existing private repository and its history are not copied, cloned, or pushed.**

Excluded on purpose:
- Eleven historical `figures/paper_author_supplied/*.pdf` assets from two-layer base because
  third-party/publisher reuse provenance was not sufficiently established.
- Historic six-layer `paper/` draft files and four archived figure files, while preserving all
  scientific code, checkpoint, scores, tests, and report JSONs. The one test requiring the omitted manuscript is explicitly skipped.
- Old 'private staging' release files, GitHub workflow notes, draft Zenodo metadata, and stale
  publication claims; new CITATION and README are independently generated.
- Raw radar recordings, local paths, personal secrets and CI caches.

The base source manifest retains original SHA-256s for the 41 unmodified two-layer source/evidence
assets. `extensions/netrad-ipix-4layer/SHA256SUMS` and
`extensions/ipix-six-layer-frozen/SHA256SUMS` record actual exported file bytes.
For the four-layer optional raw-sweep script, user-specific import paths were replaced with environment variables; its detector and output logic are unchanged.
For the six-layer source, a non-executable author-header comment was corrected and the Python AST
remains identical to the verified source. For the four-layer BSD-2 license files only the copyright
holder header was corrected; conditions are unchanged.

The public initial Git commit must be created from these distributed files with `git init` rather
than using `git clone` or a GitHub import of the author's private repository. This is a deliberate
no-history publication. No Release or DOI is created by this package builder.
