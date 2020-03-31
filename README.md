# Circulation

Computation of circulation from GP data.

Most of the code is in the two local modules `GPFields` and `Circulation`.
A script `circulation.jl` is provided as an example.

## Setup

To install the required packages, run the following from the command line:

```bash
julia --project -e "using Pkg; Pkg.instantiate()"
```

Then, to run the script:

```bash
export JULIA_NUM_THREADS=4  # optional, to use threads
julia --project circulation.jl
```
