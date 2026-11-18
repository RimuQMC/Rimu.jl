using Documenter
using Rimu

DocMeta.setdocmeta!(
    Rimu,
    :DocTestSetup,
    :(using Rimu; using Rimu.StatsTools; using DataFrames; using Random;
      using LinearAlgebra; using Suppressor; using StaticArrays);
    recursive=true,
)
# Run with fix=true to fix docstrings. The filter compares floats only up to the first
# 4 digits.
doctest(Rimu; doctestfilters=[r"(\d*)\.(\d{4})\d+" => s"\1.\2"])
