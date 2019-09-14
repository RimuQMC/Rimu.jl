# # Example script
#
# This is an example to show how to use `Rimu`.
# The source file is [`qmcexample.jl`](https://joachimbrand@bitbucket.org/joachimbrand/rimu.jl/src/master/scripts/qmcexample.jl).
# You are seeing
#md # the HTML-output generated by Documenter from a markdown file
#nb # a notebook
# generated with [`Literate.jl`](https://github.com/fredrikekre/Literate.jl).

# To get started, load `Rimu` and a few plotting scripts
using Rimu, DataFrames
include("plotting.jl")
##jl pygui(true)

# Now let's define a model problem: Here, the Bose Hubbard model on a 1D chain
# defined in real space
ham = BoseHubbardReal1D(
    n = 9,
    m = 9,
    u = 6.0,
    t = 1.0,
    AT = BSAdd64)
# We can check the dimension of linear space for the Hamiltonian:
ham(:dim)

# The aim is to calculate the ground state energy, i.e. the smallest eigenvalue
# of the Hamiltonian `ham`. We will do this with three different methods:
#  * Lanczos iterations with the package `KrylovKit.jl`,
#  * deterministic iterations of the FCIQMC algorithm, and
#  * stochastic iterations following the FCIQMC algorithm.
#
# For each of these methods we need an initial state vector, from which the
# ground state is found by iterations. We will use one that consists of
# a single configuration. The
# bit-string address of this configuration we choose as the default value:
aIni = nearUniform(ham)

# ## Exact energy with Lanczos method from KrylovKit
cIni = DVec(Dict(aIni=>1.0),ham(:dim))
capacity(cIni)
length(cIni)
using KrylovKit
println("Finding ground state deterministically with KrylovKit (Lanczos)")
@time allresults = eigsolve(ham, cIni, 1, :SR; issymmetric = true)
exactEnergy = allresults[1][1]

# ## FCIQMC simulations
# set up parameters for simulations
walkernumber = 20_000
steps = 800
dτ = 0.005

# ### Deterministic FCIQMC
svec2 = DVec(Dict(aIni => 2.0), ham(:dim))
StochasticStyle(svec2)

pa = RunTillLastStep(laststep = steps,  dτ = dτ)
τ_strat = ConstantTimeStep()
s_strat = LogUpdateAfterTargetWalkers(targetwalkers = walkernumber)
v2 = copy(svec2)
println("Finding ground state with deterministic version of fciqmc!()")
@time rdf = fciqmc!(v2, pa, ham, s_strat, τ_strat)

plotQMCStats(rdf)

# stochastic with small walker number
svec = DVec(Dict(aIni => 2), ham(:dim))
StochasticStyle(svec)

pas = RunTillLastStep(laststep = steps, dτ = dτ)
vs = copy(svec)
println("Finding ground state with stochastic version of fciqmc!()")
@time rdfs = fciqmc!(vs, pas, ham, s_strat, τ_strat)

ps = plotQMCStats(rdfs, newfig = false)
show(ps)

# plot energies
# deterministic
plotQMCEnergy(rdf,exactEnergy)

norm_ratio = rdf[2:end,:norm] ./ rdf[1:(end-1),:norm]
Ẽ = rdf[1:end-1,:shift] + (1 .- norm_ratio)./ pa.dτ
plot(rdf[2:end,:steps],Ẽ,".r")

plotQMCEnergy(rdfs, newfig=false)

start_blocking = 400
dfshift = blocking(rdfs[start_blocking:end,:shift])
qmcEnergy = dfshift[1,:mean]
qmcEnergyError = dfshift[mtest(dfshift),:std_err]
block_steps = rdfs[start_blocking:end,:steps]
fill_between(block_steps,qmcEnergy-qmcEnergyError,
    qmcEnergy+qmcEnergyError,facecolor="m",alpha=0.3)
plot(block_steps,ones(length(block_steps))*qmcEnergy,"--m")

using DSP
## define kernel for smoothing
w = 3
gausskernel = [exp(-i^2/(2*w^2)) for i = -3w:3w]
gausskernel ./= sum(abs.(gausskernel))
shift = rdfs[:,:shift]
smooth_shift = conv(shift, gausskernel)[3w+1:end-3w]
pe = plot(rdfs.steps, smooth_shift, ".-g")
show(pe)

# plot energy difference from exact for determinstic
Ediff = abs.(Ẽ .- exactEnergy)
figure(); title("log-lin plot of Ediff")
semilogy(rdf[2:end,:steps],Ediff,"xg")
show()
# plot blocking analysis for manual checking
plotBlockingAnalysisDF(dfshift)
title("Blocking analysis for `shift`")
show()
