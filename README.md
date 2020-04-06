# Circulation

Computation of circulation from GP data.

Most of the code is in the two local modules `GPFields` and `Circulation`.

## Setup

To install the required packages, run the following from the command line:

```bash
julia --project -e "using Pkg; Pkg.instantiate()"
```

## Running scripts

The [`circulation.jl`](scripts/circulation.jl) script loads parameters from a TOML file.
As an example, see [`tangle.toml`](examples/tangle.toml).

To run the script with the example parameter file:

```bash
export JULIA_NUM_THREADS=4  # optional, to use threads
julia --project scripts/circulation.jl -p examples/tangle.toml
```
