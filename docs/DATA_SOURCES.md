# Data sources

## Control repository

- Host: Vultr
- Repository path: /root/work/lensing-4
- This is the only authoritative code and Git repository.

## Remote compute and data host

- SSH host: gpu.chzmark.com
- SSH port: 2338
- Remote base path:
  /root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/

## In-scope legacy datasets

Only the datasets identified by the names or date markers:

- 0222
- 0228

are in scope.

The user reports that these two datasets were generated with different
random seeds. This must be verified from files, metadata and sample
content before relying on the claim.

## Explicit exclusions

- All existing source code under the remote base directory belongs to an
  older, unrelated project.
- That source code must not be copied, imported, modified, executed or
  used as the implementation basis for this project.
- All datasets other than 0222 and 0228 are out of scope unless the user
  explicitly approves them later.

## Data handling policy

- The 0222 and 0228 datasets are immutable inputs.
- Do not rename, modify, delete or overwrite any file in them.
- Do not recursively copy the full waveform datasets to Vultr.
- Initially retrieve only metadata, schemas, array shapes, small sampled
  rows and audit reports.
- All new code and generated results must be written to a separate new
  AutoDL working directory.
