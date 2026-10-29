# Module `StochasticStyles`

```@docs
Rimu.StochasticStyles
```

## Available `StochasticStyle`s

```@docs
StyleUnknown
```
```@autodocs
Modules = [Rimu.StochasticStyles]
Pages = ["styles.jl"]
```

## The `StochasticStyle` interface

```@docs
StochasticStyle
step_stats
apply_column!
default_style
```

## Compression strategies

```@docs
CompressionStrategy
NoCompression
StochasticStyles.ThresholdCompression
compress!
```

## Internals: Spawning strategies and convenience functions

The following functions and types are unexported and internal. Use at your own
risk when defining new styles.

```@autodocs
Modules = [Rimu.StochasticStyles]
Pages = ["spawning.jl"]
Order = [:function,:method,:type]
```
```@docs
Rimu.StochasticStyles.diagonal_step_stats_are_available
```

## Index
```@index
Pages   = ["stochasticstyles.md"]
```
