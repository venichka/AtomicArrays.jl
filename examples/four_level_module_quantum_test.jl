if pwd()[end-14:end] == "AtomicArrays.jl"
    PATH_ENV = "."
else
    PATH_ENV = "../"
end

using Pkg
Pkg.activate(PATH_ENV)

# using QuantumOptics, LinearAlgebra
# using BenchmarkTools
# using Plots

using Revise
using AtomicArrays

using LinearAlgebra
using QuantumOptics

# Build the collection
positions = [
    [0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0],
    [0.4, 0.0, 0.0],
    [0.6, 0.0, 0.0]
]
N = length(positions)

pols = AtomicArrays.polarizations_spherical(N)
gam = [AtomicArrays.gammas(0.15)[m] for m=1:3, j=1:N]
deltas = [0.0 for i = 1:N]
deltas = [0.1, 0.0, 0.0, 0.0]

coll = AtomicArrays.FourLevelAtomCollection(positions;
    deltas = deltas,
    polarizations = pols,
    gammas = gam
)

println("Constructed FourLevelAtomCollection with realistic sublevel polarizations.")
println("pols size = ", size(coll.polarizations))

# Define a plane wave field in +y direction:
amplitude = 0.2
k_mod = 2π
angle_k = [0.0, π/2]  # => +y direction
polarisation = [1.0, 0.0, 0.0]
pos_0 = [0.0, 0.0, 0.0]

field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation; position_0=pos_0)
external_drive = AtomicArrays.field.rabi(field, AtomicArrays.field.plane, coll)

println("Computed external_drive = ", external_drive)

# Build the Hamiltonian and jump operators
H = AtomicArrays.fourlevel_quantum.Hamiltonian(coll; magnetic_field=0.1,
                external_drive=external_drive,
                dipole_dipole=true)

Γ, J_ops = AtomicArrays.fourlevel_quantum.JumpOperators(coll; flatten=false)
size(Γ)
size(J_ops)
eigen(Γ)
Γ == transpose(Γ)

Ω = AtomicArrays.interaction.OmegaTensor_4level(coll)

function comm(A, B)
    return A*B - B*A
end

(comm(dagger(J_ops[1,1])*J_ops[1,1], J_ops[2,3])).data
(comm(dagger(J_ops[1,1]), J_ops[3,1]) - dagger(J_ops[1,1])*J_ops[3,1]).data
(dagger(J_ops[1,1])*J_ops[2,1]*J_ops[3,4]).data
(J_ops[1,1]*dagger(J_ops[1,1])*J_ops[2,4]).data
(J_ops[1,1]*dagger(J_ops[1,1])).data
(J_ops[1,1]*J_ops[3,2]).data

(comm(J_ops[3,2], J_ops[3,1])).data

Γ[1,1,2,3]

let 
    n = 1
    m = 3
    t = sum([(n == n2) ? 0*J_ops[1,1] : 
        2*dagger(J_ops[m1, n1])*J_ops[m,n]*J_ops[m2,n2] - 
        dagger(J_ops[m1, n1])*J_ops[m2,n2]*J_ops[m,n] -
        J_ops[m,n]*dagger(J_ops[m1, n1])*J_ops[m2,n2] for n1 = 1:N, n2 = 1:N, 
                                                          m1 = 1:3, m2 = 1:3])
   t.data 
end

# Print summary
println("Hamiltonian dimension = ", size(H.data))
println("Number of jump ops = ", length(J_ops), "  => each is dimension ", size(J_ops[1].data))

# time evolution
b = AtomicArrays.fourlevel_quantum.basis(coll)
# initial state => all ground
ψ0 = basisstate(b, [AtomicArrays.fourlevel_quantum.idx_g for i = 1:N])
ρ0 = dm(ψ0)
tspan = [0.0:0.1:200.0;]
t, rho_t = timeevolution.master_h(tspan, ψ0, H, J_ops; rates=Γ)

println("Done.")

begin
    Pkg.activate(temp=true)   # or PATH_ENV if you want to reuse your local approach
    Pkg.add("Plots")

    using Plots
end

function average_values(ops, rho_t)
    T = length(rho_t)  # number of time points

    # Distinguish whether `ops` is a single operator or an array.
    if isa(ops, Operator)
        # Single operator => result is a 1D vector of length T
        av = Vector{Float64}(undef, T)
        for t in 1:T
            av[t] = real(trace(ops * rho_t[t]))
        end
        return av
    else
        s = size(ops)  # shape of the ops array
        outshape = (T, s...)
        av = Array{ComplexF64}(undef, outshape)
        for idx in CartesianIndices(s)
            op_ij = ops[idx]  # this is one operator
            for t in 1:T
                av[t, Tuple(idx)...] = tr(op_ij * rho_t[t])
            end
        end

        return av
    end
end

function population_ops(A::AtomicArrays.FourLevelAtomCollection)
    P_m1 = SparseOpType[embed(AtomicArrays.fourlevel_quantum.basis(A), j,
                     AtomicArrays.fourlevel_quantum.sigmas_ee_[1]) 
                    for j=1:length(A.atoms)]
    P_0 = SparseOpType[embed(AtomicArrays.fourlevel_quantum.basis(A), j,
                     AtomicArrays.fourlevel_quantum.sigmas_ee_[2]) 
                    for j=1:length(A.atoms)]
    P_p1 = SparseOpType[embed(AtomicArrays.fourlevel_quantum.basis(A), j,
                     AtomicArrays.fourlevel_quantum.sigmas_ee_[3]) 
                    for j=1:length(A.atoms)]
    return P_m1, P_0, P_p1
end

av_J = average_values(J_ops, rho_t)
P_m1, P_0, P_p1 = population_ops(coll)

# We'll define arrays to store population vs time: shape = (length(t), N)
pop_e_minus = zeros(length(t), N)
pop_e_0     = zeros(length(t), N)
pop_e_plus  = zeros(length(t), N)

for n in 1:N
    for (k, ρ) in enumerate(rho_t)
        pop_e_minus[k, n] = real(tr(P_m1[n] * ρ))
        pop_e_0[k, n]     = real(tr(P_0[n] * ρ))
        pop_e_plus[k, n]  = real(tr(P_p1[n] * ρ))
    end
end

let
    p_pop = plot(layout=(1, 3), size=(1000, 300),
        title="Populations in excited sublevels", xlabel="t", ylabel="Population")

    # sublevel m=-1
    plot!(p_pop[1], t, pop_e_minus, lab=PermutedDimsArray(hcat(["Atom $n" for n=1:N]), (2,1)), 
        title="m = -1", linewidth=2)

    # sublevel m=0
    plot!(p_pop[2], t, pop_e_0, lab=PermutedDimsArray(hcat(["Atom $n" for n=1:N]), (2,1)),
        title="m = 0", linewidth=2)

    # sublevel m=+1
    plot!(p_pop[3], t, pop_e_plus, lab=PermutedDimsArray(hcat(["Atom $n" for n=1:N]), (2,1)),
        title="m = +1", linewidth=2)

    display(p_pop)  # show the figure
end

let 
    f = plot(t, hcat([real(diag(dense(rho_t[i]).data)) for i in eachindex(t)]...)', lw=2)
    display(f)
end

let 
    f = plot(t, real(av_J), lw=2, xlim=(0, 50))
    display(f)
end




function test_tensor_flattening()
    test_mat = [(i, j) for i=1:3, j=1:4]
    test_vec = vcat(test_mat...)

    # test_tensor = ["a:"*string(i)*","*string(j)*"μ:"*string(k)*","*string(m) for i=1:4, j=1:4, k=1:3, m=1:3]
    test_tensor = [(k,i,m,j) for i=1:4, j=1:4, k=1:3, m=1:3]

    begin
    M = N*3
    # rates_mat = Array{String, 2}(undef, M, M)
    rates_mat = Array{Tuple, 2}(undef, M, M)
    for n in 1:N
        for mu in 1:3
            i = 3*(n-1) + mu
            for nprime in 1:N
                for muprime in 1:3
                    j = 3*(nprime-1) + muprime
                    rates_mat[i,j] = test_tensor[n, nprime,mu, muprime]
                end
            end
        end
    end
    rates_mat
    end

    for i = 1:12
        for j=1:12
            # print(test_vec[i], "-", rates_mat[i,j], "-", test_vec[j], "\n")
            print(test_vec[i][1] == rates_mat[i,j][1], " ", 
                test_vec[i][2] == rates_mat[i,j][2], " ",
                test_vec[j][1] == rates_mat[i,j][3], " ",
                test_vec[j][2] == rates_mat[i,j][4], "\n")
        end
    end
    return test_mat, test_vec, test_tensor, rates_mat
end