module AtomicArrays

export System, Spin, SpinCollection, CavityMode, CavitySpinCollection,
        GreenTensor, OmegaMatrix, GammaMatrix,
        interaction_module, field_module, geometry_module,
        reducedspin, ReducedSpinBasis, reducedspintransition, reducedspinstate,
        reducedsigmap, reducedsigmam, reducedsigmax, reducedsigmay,
        reducedsigmaz, reducedsigmapsigmam
        # field, ,
        # collective_modes, Omega_k_chain, Gamma_k_chain, Omega_k_2D, Gamma_k_2D


include("system.jl")
include("timeevolution_base.jl")
include("interaction_module.jl")
include("effective_interaction_module.jl")
include("field_module.jl")
include("quantum_module.jl")
include("reducedspin_module.jl")
include("meanfield_module.jl")
include("mpc_module.jl")
include("geometry_module.jl")
include("misc_module.jl")

using .field_module
using .interaction_module
using .quantum_module
using .reducedspin_module
using .meanfield_module
using .mpc_module
using .effective_interaction_module
using .geometry_module
using .misc_module

end # module
