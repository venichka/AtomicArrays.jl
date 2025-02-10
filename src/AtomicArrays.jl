"""
AtomicArrays.jl is a numerical framework that can be used to simulate quantum systems consisting of spatially distributed spins interacting via Dipole-Dipole interaction, optionally coupled to a cavity.
"""
module AtomicArrays

export System, Spin, SpinCollection, CavityMode, CavitySpinCollection,
        FourLevelAtom, FourLevelAtomCollection, CavityFourLevelAtomCollection,
        GreenTensor, OmegaMatrix, GammaMatrix,
        interaction, field, geometry,
        reducedspin, ReducedSpinBasis, reducedspintransition, reducedspinstate,
        reducedsigmap, reducedsigmam, reducedsigmax, reducedsigmay,
        reducedsigmaz, reducedsigmapsigmam,
        collective_modes, Omega_k_chain, Gamma_k_chain, Omega_k_2D, Gamma_k_2D
        # field, ,


include("system.jl")
include("timeevolution_base.jl")
include("geometry.jl")
include("interaction.jl")
include("effective_interaction.jl")
include("effective_interaction_general.jl")
include("effective_interaction_rotated.jl")
include("field.jl")
include("quantum.jl")
include("reducedspin.jl")
include("independent.jl")
include("meanfield.jl")
include("mpc.jl")
include("collective_modes.jl")
include("misc.jl")

using .field
using .interaction
using .quantum
using .reducedspin
using .meanfield
using .mpc
using .effective_interaction
using .geometry
using .collective_modes
using .misc

end # module
