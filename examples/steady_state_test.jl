using DifferentialEquations, Plots
using CollectiveSpins, QuantumOptics, Sundials, LSODA, LinearAlgebra

using Revise
using AtomicArrays
const EMField = AtomicArrays.field_module.EMField
const sigma_matrices = AtomicArrays.meanfield_module.sigma_matrices
const mapexpect = AtomicArrays.meanfield_module.mapexpect
const mapexpect_mpc = AtomicArrays.mpc_module.mapexpect
const sigma_matrices_mpc = AtomicArrays.mpc_module.sigma_matrices
const ProductState = CollectiveSpins.meanfield.ProductState
const splitstate = CollectiveSpins.meanfield.splitstate


function timeevolution_field(T, S::SpinCollection, Om_R::Vector{ComplexF64},
                                       state0::ProductState;
                                       fout=nothing, alg=VCABM(), kwargs...)
    N = length(S.spins)
    @assert N==state0.N
    Ω = real(OmegaMatrix(S))
    Γ = real(GammaMatrix(S))

    function f(dy, y, p, t)
        sx, sy, sz = splitstate(N, y)
        dsx, dsy, dsz = splitstate(N, dy)
        @inbounds for k=1:N
            dsx[k] = (-S.spins[k].delta*sy[k] - 0.5*S.gammas[k]*sx[k] -
                2*sz[k]*imag(Om_R[k])) 
            dsy[k] = (S.spins[k].delta*sx[k] - 0.5*S.gammas[k]*sy[k] +
                2*sz[k]*real(Om_R[k]))
            dsz[k] = (-S.gammas[k]*(1+sz[k]) +
                2*sx[k]*imag(Om_R[k]) -
                2*sy[k]*real(Om_R[k]))
            for j=1:N
                if j==k
                    continue
                end
                dsx[k] += Ω[k,j]*sy[j]*sz[k] + 0.5*Γ[k,j]*sx[j]*sz[k]
                dsy[k] += -Ω[k,j]*sx[j]*sz[k] + 0.5*Γ[k,j]*sy[j]*sz[k]
                dsz[k] += (Ω[k,j]*(sx[j]*sy[k] - sy[j]*sx[k]) -
                    0.5*Γ[k,j]*(sx[j]*sx[k] + sy[j]*sy[k]))
            end
        end
        nothing
    end

    if isa(fout, Nothing)
        fout_(t, state::ProductState) = deepcopy(state)
    else
        fout_ = fout
    end

    return integrate(T, f, state0, fout_; alg=alg, kwargs...)
end


function integrate(T::Vector, f::Function, state0::S, fout::Function;
                    alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm = OrdinaryDiffEq.VCABM(),
                    callback = nothing, kwargs...) where S

    if isa(state0, Vector{<:Real})
        x0 = state0
        N = length(state0)
        fout_diff = (u, t, integrator) -> fout(t, deepcopy(u))
    else
        x0 = state0.data
        N = state0.N
        fout_diff = (u, t, integrator) -> fout(t, S(N, deepcopy(u)))
    end

    out_type = pure_inference(fout, Tuple{eltype(T),typeof(state0)})
    out = DiffEqCallbacks.SavedValues(eltype(T),out_type)
    scb = DiffEqCallbacks.SavingCallback(fout_diff,out,saveat=T,
                                        save_everystep=false,
                                        save_start = false)

    prob = OrdinaryDiffEq.ODEProblem(f, x0, (T[1], T[end]))

    full_cb = OrdinaryDiffEq.CallbackSet(callback, scb)

    sol = OrdinaryDiffEq.solve(prob, alg;
            reltol=1.0e-10,
            abstol=1.0e-12,
          #   save_everystep = false,
          #   save_start = false,
          #   save_end = false,
          #   callback=full_cb,
            kwargs...)

#     out.t, out.saveval
    sol

end


Base.@pure pure_inference(fout,T) = Core.Compiler.return_type(fout, T)

function f(dy::AbstractVector{T}, y::AbstractVector{S}, p, t) where {T,S}
 #    sx, sy, sz = splitstate(N, y)
 #    dsx, dsy, dsz = splitstate(N, dy)
 #    @inbounds for k=1:N
 #        dsx[k] = (-S.spins[k].delta*sy[k] - 0.5*S.gammas[k]*sx[k] -
 #            2*sz[k]*imag(Om_R[k]))
 #        dsy[k] = (S.spins[k].delta*sx[k] - 0.5*S.gammas[k]*sy[k] +
 #            2*sz[k]*real(Om_R[k]))
 #        dsz[k] = (-S.gammas[k]*(1+sz[k]) +
 #            2*sx[k]*imag(Om_R[k]) -
 #            2*sy[k]*real(Om_R[k]))
 #        for j=1:N
 #            if j==k
 #                continue
 #            end
 #            dsx[k] += Ω[k,j]*sy[j]*sz[k] + 0.5*Γ[k,j]*sx[j]*sz[k]
 #            dsy[k] += -Ω[k,j]*sx[j]*sz[k] + 0.5*Γ[k,j]*sy[j]*sz[k]
 #            dsz[k] += (Ω[k,j]*(sx[j]*sy[k] - sy[j]*sx[k]) -
 #                0.5*Γ[k,j]*(sx[j]*sx[k] + sy[j]*sy[k]))
 #        end
 #    end
    @inbounds for k=1:N
        dy[k] = (-0*y[k+1*N] - 0.5*γ_e[k]*y[k] -
            2*y[k+2*N]*imag(Om_R[k]))
        dy[k+1*N] = (0*y[k] - 0.5*γ_e[k]*y[k+1*N] +
            2*y[k+2*N]*real(Om_R[k]))
        dy[k+2*N] = (-γ_e[k]*(1+y[k+2*N]) +
            2*y[k]*imag(Om_R[k]) -
            2*y[k+1*N]*real(Om_R[k]))
       # for j=1:N
       #     if j==k
       #         continue
       #     end
       #     dy[k] += Ω[k,j]*y[j+1*N]*y[k+2*N] + 0.5*Γ[k,j]*y[j]*y[k+2*N]
       #     dy[k+1*N] += -Ω[k,j]*y[j]*y[k+2*N] + 0.5*Γ[k,j]*y[j+1*N]*y[k+2*N]
       #     dy[k+2*N] += (Ω[k,j]*(y[j]*y[k+1*N] - y[j+1*N]*y[k]) -
       #         0.5*Γ[k,j]*(y[j]*y[k] + y[j+1*N]*y[k+1*N]))
       # end
    end
    nothing
end

"""System"""

const Nx = 2
const Ny = 2
N = Nx*Ny
const d = 0.3
const L = 0.6
const pos = AtomicArrays.geometry_module.rectangle(d, d; Nx=Nx, Ny=Ny,
                                               position_0=[-(Nx-1)*d/2, 
                                                           -(Ny-1)*d/2,
                                                           -L/2])
const μ = [(i < 0) ? [1, 0, 0.0] : [1.0, 0.0im, 0.0] for i = 1:Nx*Ny]
const γ_e = [1e-2 for i = 1:Nx*Ny]
S = SpinCollection(pos, μ; gammas=γ_e)
Ω = OmegaMatrix(S)
Γ = GammaMatrix(S)

const DIRECTION = "right"
const E_ampl = 9.7e-4 + 0im
const E_kvec = 2*pi
const E_width = 0.3*d*sqrt(Nx*Ny)
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
#em_inc_function = AtomicArrays.field_module.gauss
em_inc_function = AtomicArrays.field_module.plane

E_vec = [em_inc_function(S.spins[k].position,incident_field)
         for k = 1:Nx*Ny]
Om_R = AtomicArrays.field_module.rabi(E_vec, μ)

tmax = 10000.
const T = [0:tmax/100:tmax;]
# Initial state (Bloch state)
const phi = 0.
const theta = pi/1.

# Meanfield
state0 = CollectiveSpins.meanfield.blochstate(phi, theta, Nx*Ny)
#tout, state_mf_t = timeevolution_field(T, S, Om_R, state0; alg=Rodas5());
sol = timeevolution_field(T, S, Om_R, state0; alg=Rodas5(), maxiters=1e5, reltol=1e-10, abstol=1e-12);

Plots.plot(sol, tspan=(0, tmax))

sx_mf = sum([mapexpect(CollectiveSpins.meanfield.sx, state_mf_t, i) for i=1:Nx*Ny]) ./ (Nx*Ny)
Plots.plot(T, sx_mf, tspan=(0., tmax))#, layout=(3,1))

prob_mm = ODEProblem(f, state0.data, (T[1],T[end]))
sol = solve(prob_mm, Rodas5(), reltol=1e-8, abstol=1e-6)


# function rober(du,u,p,t)
#     y₁,y₂,y₃ = u
#     k₁,k₂,k₃ = p
#     du[1] = -k₁*y₁ + k₃*y₂*y₃
#     du[2] =  k₁*y₁ - k₃*y₂*y₃ - k₂*y₂^2
#     du[3] =  k₂*y₂^2
#     nothing
#   end
#   M = [1. 0  0
#        0  1. 0
#        0  0  0]
#   f = ODEFunction(rober)
#   prob_mm = ODEProblem(rober,[1.0,0.0,0.0],(0.0,1e7),(0.04,3e7,1e4))
#   sol = solve(prob_mm, Rodas5(),reltol=1e-8,abstol=1e-8)
#   prob_ss = SteadyStateProblem(f,[1.0,0.0,0.0],p=(0.04,3e7,1e4))
#   sol_ss = solve(prob_ss, SSRootfind(),reltol=1e-10,abstol=1e-10)
  
#   sol_ss
#   sol[end]

#   Plots.plot(sol, xscale=:log10, tspan=(1e-6, 1e7), layout=(3,1))