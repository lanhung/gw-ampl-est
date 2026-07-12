# Curated legacy snapshot

This directory contains byte-for-byte small-file snapshots collected read-only
during Phase 0. Paths below each source group preserve their remote relative
layout. Large arrays, images, checkpoints and caches are intentionally absent.

- `candidate_qkzhang`: generator/QC sources from `/root/autodl-tmp/qkzhang`.
- `candidate_wjx`: selected code/docs/manifests from the downstream Git project.
- `baseline_tmp`: exact PDF baseline generator/training/report metadata from
  `/root/autodl-tmp/tmp`.

Do not edit these files. SHA256 and source paths are recorded in
`manifests/legacy_file_inventory.csv` and
`manifests/legacy_snapshot_checksums.sha256`.
