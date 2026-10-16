# Zenodo archiving after explicit final authorization

1. Push this curated **new public repository** as a history-free initial commit.
2. In Zenodo's GitHub integration, press Sync now and enable
   `Hamzeh62/DU-CCEBM-Reproducibility` (the old private repository is distinct).
3. Inspect GitHub CITATION metadata, both software license scopes, and file list.
4. After final confirmation, create a GitHub Release with tag `v1.0.0` from the new public repo.
5. Confirm that Zenodo ingested the version and produced a real DOI. Only then update
   journal Code Availability and citation metadata with the real DOI.

Zenodo uses `.zenodo.json` in preference to `CITATION.cff`. This repository provides
`CITATION.cff` only, deliberately. There is no fictitious DOI or inherited private Git history.

See https://help.zenodo.org/docs/github/enable-repository/ and
https://help.zenodo.org/docs/github/describe-software/citation-file/ .
