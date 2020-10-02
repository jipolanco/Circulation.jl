module GPStatistics

using FFTW
using HDF5
using LinearAlgebra: mul!
using TimerOutputs
using Base.Threads

using GPFields
using GPFields.Circulation

export init_statistics, analyse!
export CirculationStats, IncrementStats, VelocityLikeFields, reset!

include("statistics/statistics.jl")

end
