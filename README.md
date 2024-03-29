# Circulation.jl

[![DOI](https://zenodo.org/badge/406815671.svg)](https://zenodo.org/badge/latestdoi/406815671)

Computation of velocity circulation statistics from Navier-Stokes (NS) and
Gross-Pitaevskii (GP) data.

## Contents

- [System requirements](#system-requirements)
- [Installation](#installation)
- [Running the examples](#running-the-examples)
  1. [Analysing velocity fields (e.g. from NS simulations)](#1-analysing-velocity-fields-eg-from-ns-simulations)
  2. [Analysing GP (quantum turbulence) data](#2-analysing-gp-quantum-turbulence-data)
  3. [Detecting discrete vortices from GP data](#3-detecting-discrete-vortices-from-gp-data)
- [Requirements on input data](#requirements-on-input-data)
- [Output files](#output-files)
- [References](#references)

## System requirements

### Software requirements

This software is known to work on Linux and MacOS systems.
In particular, it has run on Fedora 34, Ubuntu 21.04, RHEL 8, and MacOS 11.2.3.
It will likely also work on other operating systems.

The software requires Julia 1.6 or above.
See below for installation details.
We take advantage of a number of Julia packages, in particular [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) and [HDF5.jl](https://github.com/JuliaIO/HDF5.jl), which wrap the corresponding C libraries.
The full list of dependencies is listed in the different `Project.toml` files, while the actual version numbers that have been known to work are detailed in the `Manifest.toml` files.

Below we list the main dependencies of this software.
Note that these do not need to be manually installed, as they are automatically installed by the Julia package manager (as detailed in the next sections).

The main Julia dependencies, and the versions on which the software has been tested, are:

- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) v1.4.5
- [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) v0.15.6
- [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl) v1.6.1
- [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl) v0.5.12

Additional dependencies required for plotting, visualisation and generation of synthetic velocity fields:

- [CairoMakie.jl](https://github.com/JuliaPlots/Makie.jl/tree/master/CairoMakie) v0.6.5
- [LaTeXStrings.jl](https://github.com/stevengj/LaTeXStrings.jl) v1.2.1
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) v1.2.12
- [UnicodePlots.jl](https://github.com/JuliaPlots/UnicodePlots.jl) v2.4.2
- [WriteVTK.jl](https://github.com/jipolanco/WriteVTK.jl) v1.10.1

### Hardware requirements

This software runs on standard CPUs.
It is possible to take advantage of the availability of multiple shared-memory CPUs for thread-based parallelisation.

## Installation

To use this software, it is necessary to install Julia and the Julia packages needed for the software to run.
Luckily, this is very easy to do thanks to Julia's built-in package manager.
The installation should take less than 15 minutes on a normal desktop computer.

### 1. Installing Julia

Julia may be installed by downloading the binaries at the [Julia website](https://julialang.org/downloads/).
Please see that link for more details.

### 2. Installing dependencies

Once Julia has been installed, run the following from the root directory of this project to install the dependencies:

```bash
julia --project -e "using Pkg; Pkg.instantiate()"
```

## Running the examples

### 1. Analysing velocity fields (e.g. from NS simulations)

#### Generating sample data

In this example, we compute circulation statistics on a synthetic velocity field generated using the [`scripts/synthetic.jl`](scripts/synthetic.jl) script.
Note that the script requires an extra set of dependencies, specified in [`scripts/Project.toml`](scripts/Project.toml).
To automatically install them, first run

```bash
julia --project=scripts -e "using Pkg; Pkg.instantiate()"
```

Note that this will install the packages listed in the `Project.toml` and `Manifest.toml` files in the [`scripts/`](scripts/) subdirectory.

Then, run the script as follows:

```bash
julia --project=scripts scripts/synthetic.jl
```

This will in particular generate binary files `VI{x,y,z}_d.000.dat` on the root directory, containing the three components of the synthetic velocity field on 64³ grid points.
Also note that the field can be visualised by opening the generated `synthetic.vti` file in [ParaView](https://www.paraview.org/).
To analyse the fields, first move them to `sample_data/NS/`:

```bash
mv -v VI*_d.000.dat sample_data/NS/
```

#### Computing circulation statistics

To analyse the data, run the [`examples/circulation_NS.jl`](examples/circulation_NS.jl) script as follows, from the root directory of this project:

```bash
export JULIA_NUM_THREADS=4  # optional, to use threads
julia --project examples/circulation_NS.jl
```

Note that the script is fully commented and may be easily modified.

On a modern desktop computer, analysing the full 64³ velocity field using 4 threads should take less than 1 second.

This will generate a `circulation_NS.h5` file containing the circulation statistics of the field.
See [Output files](#output-files) below for the typical structure of these files.

#### Plotting the generated data

The generated data may be plotted using the [`examples/plots/circulation.jl`](examples/plots/circulation.jl) script.
That script provides an example of how to access the generated HDF5 files from Julia.

Similarly to above, here we require an extra set of dependencies, in particular [Makie.jl](https://makie.juliaplots.org/) for the plots.
As before, these dependencies can be installed by running:

```bash
julia --project=examples/plots -e "using Pkg; Pkg.instantiate()"
```

Then, the results may be plotted by running

```bash
julia --project=examples/plots examples/plots/circulation.jl circulation_NS.h5
```

Note that these operations may take a while due to Julia's [time to first plot](https://discourse.julialang.org/t/time-to-first-plot-clarification/58534) problem (but things are quickly improving!).

The script will generate a `circulation_NS.svg` file with the figure, which should look like the following:

![Circulation statistics from synthetic velocity field.](docs/circulation_NS.svg)

Note that, on the right, probability distributions are vertically shifted for visualisation purposes.

### 2. Analysing GP (quantum turbulence) data

First, download the sample data available from [Zenodo](https://doi.org/10.5281/zenodo.5510350), and put the `ReaPsi.001.dat` and `ImaPsi.001.dat` under `sample_data/GP/`.
These two files are raw binary files containing the real and imaginary parts of a three-dimensional complex wave number field.
This field is an instantaneous numerical solution of the generalised Gross-Pitaevskii (GP) equations at a resolution of 256³ collocation points.

To analyse the data, run the [`examples/circulation_GP.jl`](examples/circulation_GP.jl) script as follows, from the root directory of this project:

```bash
export JULIA_NUM_THREADS=4  # optional, to use threads
julia --project examples/circulation_GP.jl
```

Note that the script is fully commented and may be easily modified.

On a modern desktop computer, analysing the full 256³ wave function field using 4 threads with `resampling_factor = 4` (see script) should take less than 5 minutes.

This will generate a `circulation_GP.h5` file containing the circulation statistics of the field.
See [Output files](#output-files) below for the typical structure of these files.

Similarly to the Navier-Stokes case, contents of this file may be plotted using the [`examples/plots/circulation.jl`](examples/plots/circulation.jl) script:

![Circulation statistics from sample GP field.](docs/circulation_GP.svg)

The probability distributions on the right clearly illustrate the quantised nature of circulation in quantum fluids, taking values that are multiples of the quantum of circulation κ.

### 3. Detecting discrete vortices from GP data

The [`detect_discrete_vortices_GP.jl`](examples/detect_discrete_vortices_GP.jl) script reads the same GP sample data as above, to detect the positions and orientations of discrete vortices:

```bash
export JULIA_NUM_THREADS=4  # optional, to use threads
julia --project examples/detect_discrete_vortices_GP.jl
```

The script will analyse each two-dimensional cut of the 256³ sample field (for a total of `3 * 256 = 768` cuts), and detect vortices on each of these slices.
Results will be saved to a `vortices.h5` file in the root directory.

On a modern desktop computer, analysing the full 256³ wave function field using 4 threads with `resampling_factor = 4` (see script) should take less than 1 minute.

To visualise the detected vortices, a [`vortices_to_vtk.jl`](examples/vortices_to_vtk.jl) script can be used to generate VTK files which can then be opened in ParaView:

```bash
julia --project examples/vortices_to_vtk.jl
```

For instance, by opening both the generated `vortices_Z_negative.vtp` and `vortices_Z_positive.vtp` files and using different colours for each dataset, one can obtain a vortex visualisation as in the following image:

![Discrete vortex visualisation.](docs/vortices_GP256_z.png)

## Requirements on input data

The above scripts take velocity or wave-function fields in raw binary format, in double-precision floating point format (`Float64` in Julia):

- Three-dimensional **velocity fields** should be given as three separate files, one for each velocity component.
  Filenames should follow the format `VI{x,y,z}_d.TTT.dat`, where `TTT` is a three-digit number (typically corresponding to a simulation timestep).

- Three-dimensional **wave-function fields** should be given as two separate files, one for the real part and the other for the imaginary part of the field.
  Filenames should follow the format `{Rea,Ima}Psi.TTT.dat`, where `TTT` is, as above, a three-digit number.

Note that the fields are interpreted in [column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order), as usual in languages such as Julia and Fortran.
If the data was generated in row-major order (e.g. if it was generated in Python or in C/C++), this means that the Cartesian directions used in the Julia scripts will be inverted with respect to their original definitions.

Also note that the data is expected to follow the same [endianness](https://en.wikipedia.org/wiki/Endianness) as the system where the scripts are executed.
If the data was generated in the same or a similar system, this is usually not a problem.
However, if the data was generated on an architecture with a different endienness, the data will need to be reordered before performing the analyses.

## Output files

Histograms and moments are written to a binary HDF5 file.
The path to the output file is specified in the parameter file.
A single HDF5 file may contain circulation statistics for the velocity, the
regularised velocity (GP only) and momentum (GP only).
HDF5 files are easy to read in different languages.

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
    /Velocity                   Group
        /kernel_area            Dataset {16}      # area of convolution kernels
        /kernel_lengths         Dataset {16, 2}   # size (rx, ry) of rectangular kernels
        /kernel_shape           Dataset {SCALAR}  # e.g. "RectangularKernel"
        /kernel_size            Dataset {16}      # linear size of convolution kernels
        /resampled_grid         Dataset {SCALAR}  # boolean; true if circulation was computed in resampled grid
        /resampling_factor      Dataset {SCALAR}  # integer

        /Histogram              Group
            /bin_edges          Dataset {4000}
            /hist               Dataset {51, 3999}
            /maximum            Dataset {16}
            /minimum            Dataset {16}
            /total_samples      Dataset {51}

        /Moments                Group
            /M_abs              Dataset {16, 10}  # moments ⟨ |Γ|^p ⟩
            /M_odd              Dataset {16, 5}   # moments ⟨ Γ^p ⟩ for p odd
            /p_abs              Dataset {20}      # values of p associated to M_abs
            /p_odd              Dataset {10}      # values of p associated to M_odd
            /total_samples      Dataset {51}
```

(You can use the command-line utility `h5ls` to see the file structure.)

## References

If you use this software, please cite the following works, where different versions of the software were used.

-  N. P. Müller, J. I. Polanco and G. Krstulovic,
  *Intermittency of Velocity Circulation in Quantum Turbulence*,
  [Phys. Rev. X **11**,
 011053 (2021)](https://doi.org/10.1103/PhysRevX.11.011053).

 - J. I. Polanco, N. P. Müller and G. Krstulovic,
   *Vortex clustering, polarisation and circulation intermittency in classical and quantum turbulence*,
   [Nat. Commun. **12**, 7090 (2021)](https://doi.org/10.1038/s41467-021-27382-6).
