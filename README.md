# AtomicArrays.jl

**`AtomicArrays.jl`** is a Julia library designed for simulating arrays of multi-level atoms, focusing on quantum optical effects and collective phenomena in structured atomic ensembles. It supports complex atomic structures, including two-level ($|g\rangle, \; |e \rangle$) and four-level ($|j = 0, m = 0 \rangle, \; |j=1, m = -1,0,+1 \rangle$) atoms, enabling analysis of collective quantum dynamics and light-matter interactions.

**Features**:

•	Multi-level Atomic Models: Simulate two-level and four-level atoms.

•	Hamiltonian Construction: Include dipole-dipole interaction, driving fields, Zeeman terms.

•	Varied Time-Evolution Methods: Perform system dynamics simulations using:

    •	Master equation

    •	Meanfield approximation

    •	Meanfield equations with correlation corrections

•	Flexible Array Geometry: Model atomic arrays with customizable geometry.

## Installation

Install `AtomicArrays.jl` directly from GitHub using Julia’s package manager:

```julia
using Pkg
Pkg.add(url="https://github.com/venichka/AtomicArrays.jl")
```

## Getting Started

Here’s a practical example demonstrating the simulation of dynamics for a four-level atomic array with Zeeman terms and a driving field using the master equation:

```julia
using QuantumOptics
using AtomicArrays

a = 0.6  # lattice period
positions = AtomicArrays.geometry.rectangle(a, a; Nx=2, Ny=2)  # create 2x2 array
N = length(positions)

pols = AtomicArrays.polarizations_spherical(N)
gam = [AtomicArrays.gammas(0.25)[m] for m=1:3, j=1:N]
deltas = [0.0 for i = 1:N]  # detunings

coll = AtomicArrays.FourLevelAtomCollection(positions;
    deltas = deltas,
    polarizations = pols,
    gammas = gam
) #  create atomic collection

# Define a plane wave field in +z direction:
amplitude = 0.1
k_mod = 2π
angle_k = [0.0, 0.0]  # => +z direction
polarisation = [1.0, 1.0im, 0.0]
pos_0 = [0.0, 0.0, 0.0]

field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation; position_0=pos_0)
external_drive = AtomicArrays.field.rabi(field, AtomicArrays.field.plane, coll)

# Uniform magnetic field
B_f = 0.2  # projection of B onto z axis

# Build the Hamiltonian and jump operators
H = AtomicArrays.fourlevel_quantum.Hamiltonian(coll; magnetic_field=B_f,
                external_drive=external_drive,
                dipole_dipole=true)

Γ_fl, J_ops = AtomicArrays.fourlevel_quantum.JumpOperators(coll; flatten=true)

# Master equation time evolution
b = AtomicArrays.fourlevel_quantum.basis(coll)
# initial state => all ground |0,0>
ψ0 = basisstate(b, [(i == 1) ? AtomicArrays.fourlevel_quantum.idx_e_plus : 
AtomicArrays.fourlevel_quantum.idx_g for i = 1:N])
ρ0 = dm(ψ0)
tspan = [0.0:0.1:200.0;]
t, rho_t = timeevolution.master_h(tspan, ψ0, H, J_ops; rates=Γ_fl)

```

This example initializes a 2x2 array of four-level atoms including Zeeman interactions and a driving field, computes the Hamiltonian, and solves the master equation.

## Documentation

Detailed documentation and function references are available within the `docs/ directory`. To build and view the documentation locally, use the `Documenter.jl` package and the following guide: https://documenter.juliadocs.org/stable/man/guide/


## Examples and Tutorials

Explore provided examples and notebooks in the examples/ directory:

	•	Superradiant and subradiant dynamics

	•	Meanfield vs. correlation-enhanced simulations

	•	Effects of Zeeman terms on collective emission


## Contributing

Contributions are welcome! Please open an issue or submit a pull request with enhancements, bug reports, or suggestions.

## License

`AtomicArrays.jl` is released under the MIT License. See the LICENSE file for more details.

## Acknowledgments

`AtomicArrays.jl` is particularly inspired by packages such as `QuantumOptics.jl` and `CollectiveSpins.jl`.
