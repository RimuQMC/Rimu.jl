# Code testing

The script `runtest.jl` in the `test/` folder contains tests of the code in `Rimu`. To run the test simply run the script from the Julia REPL or run
```
Rimu$ julia test/runtest.jl
```
from the command line.

More tests should be added over time to test core functionality of the code. To add new tests, directly edit the file `runtest.jl`.

## Automated testing with GitHub Actions

GitHub Actions are set up to run the test script automatically on the GitHub cloud server every time a new commit to the master branch is pushed to the server. The setup for this to happen is configured in the file
`actions.yml` in the `Rimu/.github/workflows` folder.

## Testing of custom types for use with `Rimu`

The module `Rimu.InterfaceTests` contains a number of functions to test the interfaces of the [`AbstractHamiltonian`](@ref) type hierarchy. See [Interface tests](@ref) in the section [Advanced operator usage and custom Hamiltonians](@ref).
