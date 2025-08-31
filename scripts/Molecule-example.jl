# # Example 6: Molecular Hamiltonian Calculation

# This is an example calculation of the water molecule (H₂O).

# A runnable script for this example is located
# [here](https://github.com/RimuQMC/Rimu.jl/blob/develop/scripts/Molecule-example.jl).
# Run it with `julia Molecule-example.jl`.

# First, we load the reqired packages. `Rimu` for FCIQMC calculation.
using Rimu

# ## Setting up the model

# We specify the path to the FCIDUMP file describing the H₂O molecule.
fcidump = joinpath(pkgdir(Rimu), "test/examples/h2o.FCIDUMP");

# Next we construct the Hamiltonian by the constructor [`MolecularHamiltonian`](@ref). 
# The Hartree-Fock ground state is generated automatically at the same time. 
h = MolecularHamiltonian(fcidump)
a = starting_address(h);

# ## Running the calculation

# ### Exact Diagonalization calculation
# We first define the problem via [`ExactDiagonalizationProblem`](@ref) interface.
# Since H₂O has a rather large Hilbert space, a iterative solver is recomended. We
# use [`KrylovKitSolver`](@ref). It requires [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).
# To install it, in Julia REPL type ] to enter the Pkg REPL mode and run:
# ```julia-repl
# pkg> add KrylovKit
# ``` 
# Or via the `Pkg` package
# ```julia-repl
# julia> import Pkg; Pkg.add("KrylovKit.jl")
# ``` 
using KrylovKit

p = ExactDiagonalizationProblem(h, algorithm=KrylovKitSolver(true))

# Then it can be solved by executing
# ```julia
# s = solve(p)
# ```


# ### Projector Monte Carlo / FCIQMC

# Set up the FCIQMC parameters.
steps_equilibrate = 1_000
steps_measure = 2_000
target_walkers = 1_000
time_step = 0.001;

# Define the problem via [`ProjectorMonteCarloProblem`](@ref) interface.
p = ProjectorMonteCarloProblem(h;
    time_step,
    last_step = steps_equilibrate + steps_measure,
    target_walkers,
    initiator=true,
)
# Run the calculation.
result = solve(p);

# Store the result into DataFrame. 
df = DataFrame(result);

# To analyse the energy shift, we can use [`shift_estimator`](@ref).
se = shift_estimator(df; skip=steps_equilibrate)