# Release notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added
* Pass custom external potential to HubbardRealSpace ([#399, #400])
* Restrict number type in `FroehlichPolaron` ([#405])

### Deprecated
* Keyword `mass` in `FroehlichPolaron` renamed to `two_m` ([#405])

### Other changes
* Documentation update ([#398, #407])
* `excitation` accepts floating point type as argument to define the type for the returned value. ([#405])

## Version [v0.18.0] - 2026-08-13

### Added
- `SignCorrelator` is a new observable for calculating mutual sign coherence [(#397)]
- New mechanism for scaling and shifting Hamiltonians by scalar values based on `ScaledOrShiftedHamiltonian` with public entry points given by `*`, `+`, `VectorInterface.scale`, and `VectorInterface.add`.  ([#396])

### Removed
- `ScaledHamiltonian` is no longer available. ([#396])

### Fixed
- Docs: fix some typos ([#393])
- Fix failing benchmark runs for fork PRs: skip PR comments for fork PRs, modernize output syntax [(#394)]
- Fix mystery allocation in allocation testing ([#395])

## Older versions
For changes prior to [v0.18.0] see the [list of releases](https://github.com/RimuQMC/Rimu.jl/releases) on GitHub.

## Instructions

Add all important changes while preparing a PR to the Unreleased section. When preparing a release, move the entries into a new section for the release.

Types of changes (based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/))

* `Added` for new features.
* `Changed` for changes in existing functionality (breaking changes to API).
* `Deprecated` for soon-to-be removed features.
* `Removed` for now removed features.
* `Fixed` for any bug fixes.
* `Security` in case of vulnerabilities
* `Other changes` for changes to internals or documentation.
