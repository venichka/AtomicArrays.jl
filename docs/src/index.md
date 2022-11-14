# AtomicArrays.jl Documentation

Welcome to the documentation page. 

!!! note "Quick-Glimpse Tutorial"
    This tutorial just offered a quick glimpse on Julia's built-in documentation system, make sure to read the docs for more.

```@docs
AtomicArrays
```

# API


## [System](@id API: System)

```@docs
System
```

```@docs
Spin
```

```@docs
SpinCollection
```

```@docs
CavityMode
```

```@docs
CavitySpinCollection
```


## [Geometry](@id API: Geometry)

```@docs
geometry.chain
```

```@docs
geometry.triangle
```

```@docs
geometry.rectangle
```

```@docs
geometry.square
```

```@docs
geometry.hexagonal
```

```@docs
geometry.box
```

```@docs
geometry.cube
```


## [Dipole-Dipole Interaction](@id API: Dipole-Dipole Interaction)

```@docs
AtomicArrays.interaction.Omega
```

```@docs
AtomicArrays.interaction.Gamma
```

```@docs
AtomicArrays.interaction.GammaMatrix
```

```@docs
AtomicArrays.interaction.OmegaMatrix
```

```@docs
AtomicArrays.interaction.GreenTensor
```

## [Effective Interactions General](@id API: Effective Interactions)

```@docs
AtomicArrays.effective_interaction_general.triangle_orthogonal
```

```@docs
AtomicArrays.effective_interaction_general.square_orthogonal
```

```@docs
AtomicArrays.effective_interaction_general.rectangle_orthogonal
```

```@docs
AtomicArrays.effective_interaction_general.cube_orthogonal
```

```@docs
AtomicArrays.effective_interaction_general.box_orthogonal
```

```@docs
AtomicArrays.effective_interaction_general.chain
```

```@docs
AtomicArrays.effective_interaction_general.chain_orthogonal
```

```@docs
AtomicArrays.effective_interaction_general.squarelattice_orthogonal
```

```@docs
AtomicArrays.effective_interaction_general.hexagonallattice_orthogonal
```

```@docs
AtomicArrays.effective_interaction_general.cubiclattice_orthogonal
```

```@docs
AtomicArrays.effective_interaction_general.tetragonallattice_orthogonal
```

```@docs
AtomicArrays.effective_interaction_general.hexagonallattice3d_orthogonal
```


### [Rotated effective interactions](@id API: Rotetated effective interactions)

```@docs
AtomicArrays.effective_interaction_rotated.square_orthogonal
```

```@docs
AtomicArrays.effective_interaction_rotated.cube_orthogonal
```

```@docs
AtomicArrays.effective_interaction_rotated.chain_orthogonal
```


## [Methods](@id API: Methods)

### [Quantum](@id API: Methods-quantum)

```@docs
AtomicArrays.quantum.basis
```

```@docs
AtomicArrays.quantum.blochstate
```

```@docs
AtomicArrays.quantum.dim
```

```@docs
AtomicArrays.quantum.Hamiltonian
```

```@docs
AtomicArrays.quantum.JumpOperators
```

```@docs
AtomicArrays.quantum.JumpOperators_diagonal
```

```@docs
AtomicArrays.quantum.timeevolution_diagonal
```

```@docs
AtomicArrays.quantum.timeevolution
```

```@docs
AtomicArrays.quantum.rotate
```

```@docs
AtomicArrays.quantum.squeeze
```

```@docs
AtomicArrays.quantum.squeezingparameter
```


### [0th order: Independent spins](@id API: Methods-cumulant0)

```@docs
AtomicArrays.independent.blochstate
```

```@docs
AtomicArrays.independent.dim
```

```@docs
AtomicArrays.independent.splitstate
```

```@docs
AtomicArrays.independent.densityoperator
```

```@docs
AtomicArrays.independent.sx
```

```@docs
AtomicArrays.independent.sy
```

```@docs
AtomicArrays.independent.sz
```

```@docs
AtomicArrays.independent.timeevolution
```


### [1st order: Meanfield](@id API: Methods-cumulant1)

```@docs
AtomicArrays.meanfield.ProductState
```

```@docs
AtomicArrays.meanfield.blochstate
```

```@docs
AtomicArrays.meanfield.dim
```

```@docs
AtomicArrays.meanfield.splitstate
```

```@docs
AtomicArrays.meanfield.densityoperator
```

```@docs
AtomicArrays.meanfield.sx
```

```@docs
AtomicArrays.meanfield.sy
```

```@docs
AtomicArrays.meanfield.sz
```

```@docs
AtomicArrays.meanfield.timeevolution
```

```@docs
AtomicArrays.meanfield.timeevolution_symmetric
```

```@docs
AtomicArrays.meanfield.rotate
```


### [2nd order: Meanfield plus Correlations (MPC)](@id API: Methods-cumulant2)

```@docs
AtomicArrays.mpc.MPCState
```

```@docs
AtomicArrays.mpc.blochstate
```

```@docs
AtomicArrays.mpc.dim
```

```@docs
AtomicArrays.mpc.splitstate
```

```@docs
AtomicArrays.mpc.correlation2covariance
```

```@docs
AtomicArrays.mpc.covariance2correlation
```

```@docs
AtomicArrays.mpc.densityoperator(::AtomicArrays.MPCState)
```

```@docs
AtomicArrays.mpc.sx
```

```@docs
AtomicArrays.mpc.sy
```

```@docs
AtomicArrays.mpc.sz
```

```@docs
AtomicArrays.mpc.Cxx
```

```@docs
AtomicArrays.mpc.Cyy
```

```@docs
AtomicArrays.mpc.Czz
```

```@docs
AtomicArrays.mpc.Cxy
```

```@docs
AtomicArrays.mpc.Cxz
```

```@docs
AtomicArrays.mpc.Cyz
```

```@docs
AtomicArrays.mpc.timeevolution
```

```@docs
AtomicArrays.mpc.rotate
```

```@docs
AtomicArrays.mpc.var_Sx
```

```@docs
AtomicArrays.mpc.var_Sy
```

```@docs
AtomicArrays.mpc.var_Sz
```

```@docs
AtomicArrays.mpc.squeeze
```

```@docs
AtomicArrays.mpc.squeezingparameter
```


### [Reduced Spin](@id API: Methods-reduced)

```@docs
ReducedSpinBasis
```

```@docs
reducedspintransition
```

```@docs
reducedsigmap
```

```@docs
reducedsigmam
```

```@docs
reducedsigmax
```

```@docs
reducedsigmay
```

```@docs
reducedsigmaz
```

```@docs
reducedsigmapsigmam
```

```@docs
reducedspinstate
```

## [Collective Modes](@id API: Collective Modes)

```@docs
Omega_k_chain
```

```@docs
Gamma_k_chain
```

```@docs
Omega_k_2D
```

```@docs
Gamma_k_2D
```
