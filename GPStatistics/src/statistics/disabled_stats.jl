struct DisabledStats <: AbstractBaseStats end

Base.zero(s::DisabledStats) = s
update!(cond, s::DisabledStats, etc...) = s
reduce!(s::DisabledStats, etc...) = s
finalise!(s::DisabledStats) = s
reset!(s::DisabledStats) = s
Base.write(g, ::DisabledStats) = g
