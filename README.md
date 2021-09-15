# Circulation.jl

Computation of velocity circulation statistics from Navier--Stokes and Gross--Pitaevskii data.

Most of the code is in the two submodules `GPFields` and `GPStatistics`.

## Setup

To install the required packages, run the following from the command line:

```bash
julia --project -e "using Pkg; Pkg.instantiate()"
```

## Circulation statistics

Circulation statistics are computed by the
[`circulation.jl`](scripts/circulation.jl) script.

### Running the examples

The script can be run from any subdirectory of this project as:

```bash
export JULIA_NUM_THREADS=4  # optional, to use threads
julia --project examples/circulation_tangle.jl
```

If you want to run the script from a different directory, call Julia with
`--project=/path/to/this/project`.
Alternatively, set the `JULIA_PROJECT` environment variable to this path.

### Output file

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

### Running on a cluster

See the example files in [`examples/idris`](examples/idris), which include a
SLURM submission script and a sample parameter file used in the [Jean-Zay
cluster](http://www.idris.fr/jean-zay/).
