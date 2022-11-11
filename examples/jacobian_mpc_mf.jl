using DifferentialEquations, ModelingToolkit
using CollectiveSpins, QuantumOptics, Sundials, LSODA, LinearAlgebra
using BenchmarkTools
using Plots

using Revise
using AtomicArrays
const EMField = AtomicArrays.field.EMField
const sigma_matrices = AtomicArrays.meanfield.sigma_matrices
const mapexpect = AtomicArrays.meanfield.mapexpect
const mapexpect_mpc = AtomicArrays.mpc.mapexpect
const sigma_matrices_mpc = AtomicArrays.mpc.sigma_matrices


"""System"""

const Nx = 4
const Ny = 4
const Nz = 2
const N = Nx*Ny*Nz
const d_1 = 0.1
const d_2 = 0.42
const L = 0.2
const pos_1 = AtomicArrays.geometry.rectangle(d_1, d_1; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_1/2, 
                                                           -(Ny-1)*d_1/2,
                                                           -L/2])
const pos_2 = AtomicArrays.geometry.rectangle(d_2, d_2; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d_2/2, 
                                                           -(Ny-1)*d_2/2,
                                                           L/2])
const pos = vcat(pos_1, pos_2)
const μ = [(i < 0) ? [1, 0, 0.0] : [1.0, 0.0im, 0.0] for i = 1:N]
const γ_e = [1e-2 for i = 1:N]
const Delt_S = [(i < N) ? 0.0 : 0.0 for i = 1:N]
S = SpinCollection(pos, μ; gammas=γ_e, deltas=Delt_S)
Ω = OmegaMatrix(S)
Γ = GammaMatrix(S)

const DIRECTION = "right"
const E_ampl = 9.7e-4 + 0im
const E_kvec = 2*pi
const E_width = 0.3*d_1*sqrt(Nx*Ny)
if (DIRECTION == "right")
    E_pos0 = [0.0,0.0,0.0]
    E_polar = [1.0, 0im, 0.0]
    E_angle = [0.0, 0.0]  # {θ, φ}
elseif (DIRECTION == "left")
    E_pos0 = [0.0,0.0,0.0*L]
    E_polar = [-1.0, 0im, 0.0]
    E_angle = [π, 0.0]  # {θ, φ}
else
    println("DIRECTION wasn't specified")
end


incident_field = EMField(E_ampl, E_kvec, E_angle, E_polar;
                     position_0 = E_pos0, waist_radius = E_width)
#em_inc_function = AtomicArrays.field.gauss
em_inc_function = AtomicArrays.field.plane

E_vec = [em_inc_function(S.spins[k].position,incident_field)
         for k = 1:N]
Om_R = AtomicArrays.field.rabi(E_vec, μ)

tmax = 10000.
const T = [0:tmax/100:tmax;]
# Initial state (Bloch state)
const phi = 0.
const theta = pi/1.

p_mf = (N, γ_e, Delt_S, Ω, real(Γ), Om_R)
p_mpc = (N, γ_e, Delt_S, Ω, real(Γ), real(Om_R), imag(Om_R))



# Meanfield
state0 = CollectiveSpins.meanfield.blochstate(phi, theta, N)
prob_mf = ODEProblem(AtomicArrays.meanfield.f,state0.data,(T[1],T[end]),p_mf)
@variables u[axes(prob_mf.u0)...] t
u = collect(u)
du = similar(u)
AtomicArrays.meanfield.f_sym(du, u, prob_mf.p, t)
sparsity = Symbolics.jacobian_sparsity(du, u)
jac_mf = Symbolics.jacobian(vec(du), vec(u))
fjac = eval(Symbolics.build_function(jac_mf, u,
            parallel=Symbolics.MultithreadedForm())[2])

sol_mf = solve(prob_mf);
sol_mf_jac_sparse = solve(remake(prob_mf, f= ODEFunction{true}(AtomicArrays.meanfield.f, jac_prototype=float(sparsity))), VCABM());
sol_mf_jac = solve(remake(prob_mf, f= ODEFunction{true}(AtomicArrays.meanfield.f, jac = (du, u, p, t) -> fjac(du, u), jac_prototype = similar(jac_mf, Float64))));
@btime sol_mf = solve(prob_mf);
@btime sol_mf_jac_sparse = solve(remake(prob_mf, f= ODEFunction{true}(AtomicArrays.meanfield.f, jac_prototype=float(sparsity))), VCABM());
@btime sol_mf_jac = solve(remake(prob_mf, f= ODEFunction{true}(AtomicArrays.meanfield.f, jac = (du, u, p, t) -> fjac(du, u), jac_prototype = similar(jac_mf, Float64))));

sol_mf_jac_sparse.u[end] ≈ sol_mf.u[end]
sol_mf_jac.u[end] ≈ sol_mf.u[end]

Plots.plot(sol_mf, tspan=(0, tmax), vars=[1])
Plots.plot!(sol_mf_jac, tspan=(0, tmax), vars=[1])
Plots.plot!(sol_mf_jac_sparse, tspan=(0, tmax), vars=[1])

#tout, state_mf_t = timeevolution_field(T, S, Om_R, state0; alg=Rodas5());
@btime sol = AtomicArrays.meanfield.timeevolution_field(T, S, Om_R, state0; alg=VCABM(), maxiters=1e5, reltol=1e-10, abstol=1e-12);



# MPC
state0 = CollectiveSpins.mpc.blochstate(phi, theta, N)
prob_mpc = ODEProblem(AtomicArrays.mpc.f,state0.data,(T[1],T[end]),p_mpc)
@variables u[axes(prob_mpc.u0)...] t
u = collect(u)
du = similar(u)
du .= 0
AtomicArrays.mpc.f_sym(du, u, prob_mpc.p, t)
sparsity_mpc = Symbolics.jacobian_sparsity(du, u)
jac_mpc = Symbolics.jacobian(du, u)
fjac = eval(Symbolics.build_function(jac_mpc, u,
            parallel=Symbolics.MultithreadedForm())[2])

sol_mpc = solve(prob_mpc);
sol_mpc_jac_sparse = solve(remake(prob_mpc, f= ODEFunction{true}(AtomicArrays.mpc.f, jac_prototype=float(sparsity_mpc))), VCABM());
sol_mpc_jac = solve(remake(prob_mpc, f= ODEFunction{true}(AtomicArrays.mpc.f, jac = (du, u, p, t) -> fjac(du, u), jac_prototype = similar(jac_mpc, Float64))));
@btime sol_mpc = solve(prob_mpc);
@btime sol_mpc_jac_sparse = solve(remake(prob_mpc, f= ODEFunction{true}(AtomicArrays.mpc.f, jac_prototype=float(sparsity))), VCABM());
@btime sol_mpc_jac = solve(remake(prob_mpc, f= ODEFunction{true}(AtomicArrays.mpc.f, jac = (du, u, p, t) -> fjac(du, u), jac_prototype = similar(jac_mpc, Float64))));

sol_mpc_jac_sparse.u[end] ≈ sol_mpc.u[end]
sol_mpc_jac.u[end] ≈ sol_mpc.u[end]

Plots.plot(sol_mpc, tspan=(0, tmax), vars=[6240-96+1])
Plots.plot!(sol_mpc_jac, tspan=(0, tmax), vars=[3])
Plots.plot!(sol_mpc_jac_sparse, tspan=(0, tmax), vars=[6240-96+1])

#tout, state_mf_t = timeevolution_field(T, S, Om_R, state0; alg=Rodas5());
@btime sol = AtomicArrays.mpc.timeevolution_field(T, S, Om_R, state0; alg=VCABM(), maxiters=1e5, reltol=1e-10, abstol=1e-12);