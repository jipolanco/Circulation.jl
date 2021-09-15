# Circulation.jl

Computation of velocity circulation statistics from Navier-Stokes and Gross-Pitaevskii data.

## System requirements

This software is known to work on Linux systems.
In particular, it has run on Fedora 34, Ubuntu 21.04, and RHEL 8.
It will likely also work on other operating systems.

The software requires Julia 1.6 or above.
See below for installation details.
We take advantage of a number of Julia packages.
The full list of dependencies is listed in the different `Project.toml` files, while the actual version numbers that have been known to work are listed in `Manifest.toml`.
As detailed below, the Julia package manager allows to easily install the very same versions of the packages listed in the manifest.

This software runs on standard CPUs.
It is possible to take advantage of the availability of multiple shared-memory CPUs for thread-based parallelisation.

## Installation

To use this software, it is necessary to install Julia and the Julia packages needed for the software to run.
Luckily, this is very easy to do thanks to Julia's built-in package manager.
The installation should typically last about 15 minutes on a normal desktop computer.

### 1. Installing Julia

Julia may be installed by downloading the binaries at the [Julia website](https://julialang.org/downloads/).
Please see that link for more details.

### 2. Installing dependencies

Once Julia has been installed, run the following from the root directory of this project to install the dependencies:

```bash
julia --project -e "using Pkg; Pkg.instantiate()"
```

## Running the examples

### 1. Analysing GP (quantum turbulence) data

First, download the sample data available from [Zenodo](https://doi.org/10.5281/zenodo.5510350), and put the `ReaPsi.001.dat` and `ImaPsi.001.dat` under `test_data/GP/`.
These two files are raw binary files containing the real and imaginary parts of a three-dimensional complex wave number field.
This field is an instantaneous numerical solution of the generalised Gross-Pitaevskii (GP) equations at a resolution of $256^3$ collocation points.

To analyse the data, run the `examples/circulation_GP.jl` script as follows, from the root directory of this project:

```bash
export JULIA_NUM_THREADS=4  # optional, to use threads
julia --project examples/circulation_GP.jl
```

This will generate a `circulation_GP.h5` file with the circulation statistics of the field.

## Output files

Histograms and moments are written to a binary HDF5 file.
The path to the output file is specified in the parameter file.
A single HDF5 file contains circulation statistics for the velocity, the
regularised velocity and momentum.

HDF5 files are easy to read in different languages.
In Python, the `h5py` package can be used.
For an example of how to read statistics in Python, see
[`plot_stats.py`](scripts/plot_stats.py).

HDF5 files have a filesystem-like structure, where each "directory" is called
a *group*.
The structure of the output HDF5 files looks something like the following:

```bash
# Simulation parameters
/ParamsGP                       Group
    /L                          Dataset {3}
    /c                          Dataset {SCALAR}
    /dims                       Dataset {3}
    /kappa                      Dataset {SCALAR}
    /nxi                        Dataset {SCALAR}
    /xi                         Dataset {SCALAR}

# Circulation statistics
/Circulation                    Group
    /loop_sizes                 Dataset {51}
    /Momentum                   Group
        /Histogram              Group
            /bin_edges          Dataset {8100}
            /hist               Dataset {51, 8099}
            /total_samples      Dataset {51}
        /Moments                Group
            /M_abs              Dataset {51, 20}
            /M_odd              Dataset {51, 10}
            /p_abs              Dataset {20}
            /p_odd              Dataset {10}
            /total_samples      Dataset {51}

    # These also include Histogram and Moments groups
    /RegVelocity                Group
    /Velocity                   Group
```

(You can use the command-line utility `h5ls` to see the file structure.)

So, for instance, to read `kappa` and the velocity histogram data in Python:

```py
import h5py

with h5py.File('filename.h5', 'r') as ff:
    kappa = ff['/ParamsGP/kappa'][()]                        # scalar
    hist = ff['/Circulation/Velocity/Histogram/hist'][:, :]  # matrix

    # Alternative:
    g = ff['/ParamsGP']  # HDF5 group with parameters
    kappa = g['kappa'][()]

    g = ff['/Circulation/Velocity/Histogram']
    hist = g['hist'][:, :]
```

