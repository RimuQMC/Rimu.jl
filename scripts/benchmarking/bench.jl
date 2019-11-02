using Rimu
using Test
using LinearAlgebra
using Rimu.ConsistentRNG
using BenchmarkTools

@testset "fciqmc.jl" begin
ham = BoseHubbardReal1D(
    n = 15,
    m = 9,
    u = 6.0,
    t = 1.0,
    AT = BSAdd64)
ham(:dim)
aIni = nearUniform(ham)
iShift = diagME(ham, aIni)

# standard fciqmc
s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
svec = DVec(Dict(aIni => 20), ham(:dim))
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 1_000
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())

@test sum(rdfs[:,:spawns]) == 580467

# # replica fciqmc
# tup1 = (copy(svec),copy(svec))
# s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
# pb = RunTillLastStep(laststep = 1, shift = iShift)
# seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
# @time rr = fciqmc!(tup1, ham, pb, s)
# pb.laststep = 10
# @time rr = fciqmc!(tup1, ham, pb, s)
#
# @test sum(rr[1][:,:xHy]) ≈ -10456.373910680508

sv = DVec(Dict(aIni => 20.0), 100)
hsv = ham(sv)
v2 = similar(sv,ham(:dim))
@benchmark ham(v2, hsv)

end

@testset "fciqmc with BoseBA" begin
n = 15
m = 9
aIni = BoseBA(n,m)
ham = BoseHubbardReal1D(
    n = n,
    m = m,
    u = 6.0,
    t = 1.0,
    AT = typeof(aIni))
iShift = diagME(ham, aIni)

# standard fciqmc
s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
svec = DVec(Dict(aIni => 20), ham(:dim))
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 1_000
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())

@test sum(rdfs[:,:spawns]) == 535061

sv = DVec(Dict(aIni => 20.0), 100)
hsv = ham(sv)
v2 = similar(sv,ham(:dim))
@benchmark ham(v2, hsv)

n = 200
m = 200
aIni = BoseBA(n,m)
ham = BoseHubbardReal1D(
    n = n,
    m = m,
    u = 6.0,
    t = 1.0,
    AT = typeof(aIni))
iShift = diagME(ham, aIni)

# standard fciqmc
tw = 1_000
s = LogUpdateAfterTargetWalkers(targetwalkers = tw)
svec = DVec(Dict(aIni => 20), 8*tw)
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift, dτ = 0.001)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 1_000
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())
@test sum(rdfs[:,:spawns]) == 769149

bIni = BoseBA(200,200)
svb = DVec(Dict(bIni => 20.0), 8*tw)
hsvb = ham(svb)
v2b = similar(svb,150_000)
@benchmark ham(v2b, hsvb)

end

@testset "fciqmc. wit BStringAdd" begin
ham = BoseHubbardReal1D(
    n = 15,
    m = 9,
    u = 6.0,
    t = 1.0,
    AT = BStringAdd)
ham(:dim)
aIni = nearUniform(ham)
iShift = diagME(ham, aIni)

# standard fciqmc
s = LogUpdateAfterTargetWalkers(targetwalkers = 1_000)
svec = DVec(Dict(aIni => 20), ham(:dim))
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 1_000
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())

@test sum(rdfs[:,:spawns]) == 534068

ham = BoseHubbardReal1D(
    n = 200,
    m = 200,
    u = 6.0,
    t = 1.0,
    AT = BStringAdd)
ham(:dim)
aIni = nearUniform(ham)
iShift = diagME(ham, aIni)

# standard fciqmc
tw = 1_000
s = LogUpdateAfterTargetWalkers(targetwalkers = tw)
svec = DVec(Dict(aIni => 20), 8*tw)
StochasticStyle(svec)
vs = copy(svec)
seedCRNG!(12345) # uses RandomNumbers.Xorshifts.Xoroshiro128Plus()
pa = RunTillLastStep(laststep = 1, shift = iShift, dτ = 0.001)
@time rdfs = fciqmc!(vs, pa, ham, s, EveryTimeStep())
pa.laststep = 1_000
@time rdfs = fciqmc!(vs, pa, rdfs, ham, s, EveryTimeStep())
@test sum(rdfs[:,:spawns]) == 534068

sv = DVec(Dict(aIni => 20.0), 8*tw)
hsv = ham(sv)
v2 = similar(sv,150_000)
@benchmark ham(v2, hsv)
end

using Rimu.Hamiltonians

c1 = BSAdd64(0xf342564ffdd03e)
c2 = BSAdd128(0xf342564ffdf00dfdfdfdfd037a3de)
bs1 = BitAdd{40}(0xf342564ffd)
bs2 = BitAdd{128}(0xf342564ffdf00dfdfdfdfd037a3de)
bs3 = BitAdd{144}(big"0xf342564ffdf00dfdfdfdfdfdfdfdfdfdfdf")
bb1 = BoseBA(bs1)
bb2 = BoseBA(bs2)
bb3 = BoseBA(bs3)
b1 = BoseBA(15,9)
b2 = BoseBA(200,200)

ham = BoseHubbardReal1D(
    n = 15,
    m = 9,
    u = 6.0,
    t = 1.0,
    AT = BStringAdd)
a1 = nearUniform(ham)

ham = BoseHubbardReal1D(
    n = 200,
    m = 200,
    u = 6.0,
    t = 1.0,
    AT = BStringAdd)
a2 = nearUniform(ham)

# BSAdd64
@benchmark Hamiltonians.numberoccupiedsites(c1)
@benchmark Hamiltonians.bosehubbardinteraction(c1)
@benchmark Hamiltonians.hopnextneighbour(c1,4,17,24)

# BSAdd128
@benchmark Hamiltonians.numberoccupiedsites(c2)
@benchmark Hamiltonians.bosehubbardinteraction(c2)
@benchmark Hamiltonians.hopnextneighbour(c2,4,55,74)
# BoseBA
@benchmark Hamiltonians.numberoccupiedsites(bb1)
@benchmark Hamiltonians.bosehubbardinteraction(bb1)
@benchmark Hamiltonians.hopnextneighbour(bb1,4,17,24)

@benchmark Hamiltonians.numberoccupiedsites(bb2)
@benchmark Hamiltonians.bosehubbardinteraction(bb2)
@benchmark Hamiltonians.hopnextneighbour(bb2,4,55,74)
@benchmark Hamiltonians.numberoccupiedsites(bb3)
@benchmark Hamiltonians.bosehubbardinteraction(bb3)
  # still has 1 memory allocation. I wonder why

@benchmark Hamiltonians.numberoccupiedsites(b1)
@benchmark Hamiltonians.bosehubbardinteraction(b1)
@benchmark Hamiltonians.hopnextneighbour(b1,3,9,15)
@benchmark Hamiltonians.hopnextneighbour(b1,4,9,15)
@benchmark Hamiltonians.numberoccupiedsites(b2)
@benchmark Hamiltonians.bosehubbardinteraction(b2)
@benchmark Hamiltonians.hopnextneighbour(b2,4,200,200)

# BStringAdd
@benchmark Hamiltonians.numberoccupiedsites(a1)
@benchmark Hamiltonians.bosehubbardinteraction(a1)
@benchmark Hamiltonians.hopnextneighbour(a1,3,9,15)
@benchmark Hamiltonians.numberoccupiedsites(a2)
@benchmark Hamiltonians.bosehubbardinteraction(a2)
@benchmark Hamiltonians.hopnextneighbour(a2,4,200,200)

nls = Hamiltonians.numberlinkedsites(bb2)
for i in 1:nls
