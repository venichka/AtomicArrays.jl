using CollectiveSpins, QuantumOptics, LinearAlgebra
using BenchmarkTools
using Plots

using Revise
using AtomicArrays

function xyz(a_1, a_2; Nx=6, Ny=6, position_0=[0,0,0])
    pos = AtomicArrays.geometry_module.dimer_square_1(a_1, a_2; Nx=Nx, Ny=Ny,
                                        position_0=[
                                          -0.5*((Nx÷2)*a_1 + (Nx-1)÷2*a_2),
                                          -0.5*((Ny÷2)*a_1 + (Ny-1)÷2*a_2),
                                          position_0[3]
                                        ])
    x = [p[1] for p in pos]
    y = [p[2] for p in pos]
    z = [p[3] for p in pos]
    return x, y, z
end

plotly()

x, y, z = xyz(0.3, 0.3);
scatter(x,y,z)
x_1, y_1, z_1 = xyz(0.45, 0.3; position_0 = [0,0,0.5]);
scatter!(x_1,y_1,z_1)