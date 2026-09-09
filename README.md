# Rimu

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://RimuQMC.github.io/Rimu.jl/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://RimuQMC.github.io/Rimu.jl/dev/)
[![Coverage Status](https://coveralls.io/repos/github/RimuQMC/Rimu.jl/badge.svg)](https://coveralls.io/github/RimuQMC/Rimu.jl)
[![arXiv](https://img.shields.io/badge/arXiv-2601.19505-green.svg)](http://arxiv.org/abs/2601.19505)

*Random Integrators for many-body quantum systems*

The grand aim is to develop a toolbox for many-body quantum systems that can be represented by a Hamiltonian in second quantisation language. Currently supported features include:
### Interacting with quantum many-body models
* **Full configuration interaction quantum Monte Carlo (FCIQMC)**, a flavour of projector quantum Monte Carlo for stochastically solving the time-independent Schrödinger equation.
* **Matrix-free exact diagonalisation** of quantum Hamiltonians (with external package [`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl)).
* **Sparse matrix representation** of quantum Hamiltonians for exact diagonalisation with sparse linear algebra package of your choice (fastest for small systems).

### Representing quantum many-body models
* A composable and efficient type system for representing single- and multi-component **Fock states** of bosons, fermions, and mixtures thereof, to be used as a basis for representing Hamiltonians.
* An **interface for defining many-body Hamiltonians**.
* Pre-defined models include:
  * **Hubbard model** in real space for bosons and fermions and mixtures in 1, 2, and 3 spatial dimensions.
  * Hubbard and related **lattice models in momentum space** for bosons and fermions in one spatial dimension.
  * **Transcorrelated Hamiltonian** for contact interactions in one dimension for fermions, as described in Jeszenski *et al.* [arXiv:1806.11268](http://arxiv.org/abs/1806.11268).

### Statistical analysis of Monte Carlo data
* **Blocking analysis** following Flyvberg & Peterson [JCP (1989)](http://aip.scitation.org/doi/10.1063/1.457480), and automated with hypothesis testing by Jonsson
[PRE (2018)](https://link.aps.org/doi/10.1103/PhysRevE.98.043304).
* **Unbiased estimators** for the ground state energy by re-reweighting following Nightingale & Blöte [PRB (1986)](https://link.aps.org/doi/10.1103/PhysRevB.33.659) and Umrigar *et al.* [JCP (1993)](http://aip.scitation.org/doi/10.1063/1.465195).

The code supports parallelisation with MPI (harnessing [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl)) as well as native Julia threading (experimental). In the future, we may add tools to solve the time-dependent Schrödinger equation and Master equations for open system time evolution.

**Concept:** Joachim Brand and Elke Pahl.

**Contributors:** Joachim Brand, Elke Pahl, Mingrui Yang, Matija Cufar, Chris Bradly.

Discussions, help, and additional contributions are acknowledged by Ali Alavi,
Didier Adrien, Chris Scott (NeSI), Alexander Pletzer (NeSI).

### Installing Rimu

`Rimu` is a registered package and can be installed with the package manager.
Hit the `]` key at the Julia REPL to get into `Pkg` mode and type
```
pkg> add Rimu
```
Alternatively, use
```julia-repl
julia> using Pkg; Pkg.add(name="Rimu")
```
in order to install `Rimu` from a script.

### Usage

The package is now installed and can be imported with
```julia-repl
julia> using Rimu
```

Note that `Rimu` is under active development and breaking changes to the user interface may occur at any time. We encourage potential users of the package to contact the authors for efficient communication.

### Software publication
To learn more about the algorithms and concepts behind `Rimu`, read the Rimu.jl preprint: 
- “Rimu.jl: Random integrators for many-body quantum systems”, M. Čufar, C. J. Bradly, R. Yang, E. Pahl, and J. Brand, [arXiv:2601.19505 (2026)](http://arxiv.org/abs/2601.19505).

If you use `Rimu` for your work, please cite the preprint:
```
@misc{Cufar2026,
  title = {{Rimu.jl}: {{Random}} Integrators for Many-Body Quantum Systems},
  shorttitle = {Rimu.jl},
  author = {{\v C}ufar, Matija and Bradly, C. J. and Yang, Ray and Pahl, Elke and Brand, Joachim},
  year = 2026,
  number = {arXiv:2601.19505},
  eprint = {2601.19505},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2601.19505},
  archiveprefix = {arXiv},
  url = {http://arxiv.org/abs/2601.19505},
}
```

### Other references
The original references for the FCIQMC algorithm are:
- "Fermion Monte Carlo without fixed nodes: A game of life, death, and annihilation in Slater determinant space", G. H. Booth, A. J. W. Thom, A. Alavi, [*J. Chem. Phys.* **131**, 054106 (2009)](https://doi.org/10.1063/1.3193710).
-  "Communications: Survival of the fittest: accelerating convergence in full configuration-interaction quantum Monte Carlo.", D. Cleland,  G. H. Booth, A. Alavi, [*J. Chem. Phys.* **132**, 041103 (2010)](https://doi.org/10.1063/1.3302277).

Scientific papers describing additional features implemented in `Rimu`:
- "Improved walker population control for full configuration interaction quantum Monte Carlo", M. Yang, E. Pahl, J. Brand, [*J. Chem. Phys.* **153**, 170143 (2020)](https://doi.org/10.1063/5.0023088), [arXiv:2008.01927](https://arxiv.org/abs/2008.01927).
- "Stochastic differential equation approach to understanding the population control bias in full configuration interaction quantum Monte Carlo", J. Brand, M. Yang, E. Pahl. [*Phys. Rev. B* **105** 235144 (2022)](https://link.aps.org/doi/10.1103/PhysRevB.105.235144), [arXiv:2103.07800](http://arxiv.org/abs/2103.07800) (2021).
- “Accelerating the convergence of exact diagonalization with the transcorrelated method: Quantum gas in one dimension with contact interactions”, P. Jeszenszki, H. Luo, A. Alavi, and J. Brand. [*Phys. Rev. A* **98** 053627 (2018)](https://link.aps.org/doi/10.1103/PhysRevA.98.053627), [arXiv:1806.11268](http://arxiv.org/abs/1806.11268).


Papers discussing results obtained with `Rimu`:
- “Scale invariance of the polaron energy at the Mott-superfluid critical point”, M. Čufar, R. Alhyder, C. J. Bradly, V. Colussi, G. M. Bruun, J. Brand, and A. Recati, [arXiv:2604.17824](http://arxiv.org/abs/2604.17824) (2026).
- “Bound excited states of Fröhlich polarons in one dimension”, J. Taylor, M. Čufar, D. Mitrouskas, R. Seiringer, E. Pahl, and J. Brand. [*Phys. Rev. B* **112** 184312 (2025)](https://link.aps.org/doi/10.1103/s9p9-jflq), [arXiv:2506.02440](http://arxiv.org/abs/2506.02440).
- “Lattice Bose polarons at strong coupling and quantum criticality”, R. Alhyder, V. Colussi, M. Čufar, J. Brand, A. Recati, and G. M. Bruun. [*SciPost Physics* **19** 002 (2025)](https://scipost.org/SciPostPhys.19.1.002).
- “Effective Theory for Strongly Attractive One-Dimensional Fermions”, T. G. Backert, F. Brauneis, M. Čufar, J. Brand, H.-W. Hammer, and A. G. Volosniev. [*Phys. Rev. Lett.* **135** 040401 (2025)](https://link.aps.org/doi/10.1103/8mnc-x42q), [arXiv:2412.05915](http://arxiv.org/abs/2412.05915).
- "Magnetic impurity in a one-dimensional few-fermion system", L. Rammelmüller, D. Huber, M. Čufar, J. Brand, A. Volosniev. [*SciPost Physics* **14** 006 (2023)](https://scipost.org/10.21468/SciPostPhys.14.1.006).
- "Polaron-Depleton Transition in the Yrast Excitations of a One-Dimensional Bose Gas with a Mobile Impurity", M. Yang, M. Čufar, E. Pahl, J. Brand, [*Condens. Matter* **7**, 15 (2022)](https://www.mdpi.com/2410-3896/7/1/15).

For more information, consult the [documentation](https://RimuQMC.github.io/Rimu.jl/dev/).
