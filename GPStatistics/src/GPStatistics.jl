module GPStatistics

using FFTW
using HDF5
using TimerOutputs
using Base.Threads

using GPFields
using GPFields.Circulation

export init_statistics, analyse!
export CirculationStats, IncrementStats, VelocityLikeFields, reset!
export ParamsMoments, ParamsHistogram

include("statistics/statistics.jl")

end
