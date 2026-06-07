# Vendored FastWAM Runtime

This package vendors the runtime subset of `yuantianyuan01/FastWAM` for the
`fastwam-libero` product path.

- Source repository: `https://github.com/yuantianyuan01/FastWAM`
- Source commit: `45d8e1458921d83f8ad6cf9ce993d371208dabd0`
- Source license: MIT License, copyright The FastWAM Authors
- Vendored scope: `src/fastwam` runtime/config files only
- Excluded scope: training launchers, project-level scripts, and `third_party`
  simulator/vendor trees

Local edits should stay limited to making the runtime importable without
training dependencies and to adapting asset/config lookup for WAM Harness.
