[Diff since v0.16.0](https://github.com/RimuQMC/Rimu.jl/compare/v0.16.0...v0.16.1)

- Adds hard-core boson support and variable particle-number support for `FermiFS` and `HardcoreBoseFS` (#376, #385).
- Broadens `deposit!` call signatures for improved interface flexibility (#377).

- Relaxes `replica_stats` type constraints to support heterogeneous `spectral_states` (#389).
- Removes a Julia deprecation warning/error path for `Tuple{Vararg}` method signatures under `--depwarn=yes` (#390).

- CI maintenance: bump `actions/checkout` from 6 to 7 (#384).

**Merged pull requests:**
- Hard core bosons (#376) (@Skuwar1)
- Widen `deposit!` call signature (#377) (@Kryštof Krsek)
- Bump actions/checkout from 6 to 7 (#384) (@dependabot[bot])
- Variable particle number for FermiFS and HarcoreBoseFS (#385) (@joachimbrand)
- Relax type constraint in replica_stats to allow heterogeneous spectral_states (#389) (@Jyotiraj Nath)
- Remove deprecation warning for Tuple{Vararg} type signature (#390) (@joachimbrand)
