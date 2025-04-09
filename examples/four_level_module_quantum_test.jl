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

function comm(A, B)
    return A*B - B*A
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

function scattered_field(r::Vector, A::AtomicArrays.FourLevelAtomCollection,
    sigmas_m::Matrix, k_field::Number=2π)
    M, N = size(A.gammas)
    C = 3.0/4.0 * A.gammas
    return sum(C[m,n] * sigmas_m[m,n] * 
               GreenTensor(r-A.atoms[n].position, k_field) *
               A.polarizations[m,:,n] for m = 1:M, n = 1:N)
end


begin
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
    # deltas = [0.0 for i = 1:N]
    deltas = [0.1, 0.2, 0.0, 0.0]

    coll = AtomicArrays.FourLevelAtomCollection(positions;
        deltas = deltas,
        polarizations = pols,
        gammas = gam
    )

    # Define a plane wave field in +y direction:
    amplitude = 0.1
    k_mod = 2π
    # angle_k = [0.0, π/2]  # => +y direction
    angle_k = [0.0, 0.0]  # => +z direction
    polarisation = [1.0, 0.0im, 1.0im]
    pos_0 = [0.0, 0.0, 0.0]

    B_z = 0.1


    field = AtomicArrays.field.EMField(amplitude, k_mod, angle_k, polarisation; position_0=pos_0, waist_radius=0.3)
    field_func = AtomicArrays.field.gauss
    external_drive = AtomicArrays.field.rabi(field, field_func, coll)
end

begin
    # Build the Hamiltonian and jump operators
    H = AtomicArrays.fourlevel_quantum.Hamiltonian(coll; magnetic_field=B_z,
                    external_drive=external_drive,
                    dipole_dipole=true)

    Γ, J_ops = AtomicArrays.fourlevel_quantum.JumpOperators(coll; flatten=true)
end

begin
    # time evolution
    b = AtomicArrays.fourlevel_quantum.basis(coll)
    # initial state => all ground
    ψ0 = basisstate(b, [AtomicArrays.fourlevel_quantum.idx_g for i = 1:N])
    ρ0 = dm(ψ0)
    tspan = [0.0:0.1:400.0;]
    t, rho_t = timeevolution.master_h(tspan, ψ0, H, J_ops; rates=Γ)
end

begin
    state0 = AtomicArrays.fourlevel_meanfield.ProductState(length(coll.atoms))
    tout, state_mf_t = AtomicArrays.fourlevel_meanfield.timeevolution(tspan, coll, external_drive, B_z, state0);
end

begin
    Pkg.activate(temp=true)   # or PATH_ENV if you want to reuse your local approach
    Pkg.add("Plots")

    using Plots
end


begin
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
end

begin
    # Define grid parameters
    x_range = -1:0.01:1
    y_range = -1:0.01:1
    z0 = 0.1

    # sigmas_m = reshape(av_J[end,:], (3,4))
    sigmas_m, _ = AtomicArrays.fourlevel_meanfield.sigma_matrices(state_mf_t, length(tspan))

    # Preallocate arrays to store field projections
    nx, ny = length(x_range), length(y_range)
    Re_field_x = zeros(nx, ny)
    Abs_field_x = zeros(nx, ny)
    Re_field_y = zeros(nx, ny)
    Abs_field_y = zeros(nx, ny)
    Re_field_z = zeros(nx, ny)
    Abs_field_z = zeros(nx, ny)

    # Extract atom positions (assumed stored in coll.atoms)
    atom_x = [atom.position[1] for atom in coll.atoms]
    atom_y = [atom.position[2] for atom in coll.atoms]

    # Loop over the grid to compute the scattered field at each point
    for (i, x) in enumerate(x_range)
        for (j, y) in enumerate(y_range)
            r = [x, y, z0]
            # Compute the scattered field at r using the provided function.
            # It is assumed that sigmas_m is defined.
            E = AtomicArrays.field.scattered_field(r, coll, sigmas_m)
            E = AtomicArrays.field.total_field(field_func, r, field, coll, sigmas_m)
            # Store the real part and absolute value for each Cartesian component.
            Re_field_x[i, j] = real(E[1])
            Abs_field_x[i, j] = abs(E[1])
            Re_field_y[i, j] = real(E[2])
            Abs_field_y[i, j] = abs(E[2])
            Re_field_z[i, j] = real(E[3])
            Abs_field_z[i, j] = abs(E[3])
        end
    end
end

let
    # Create contour plots for each field component.
    p1 = contourf(x_range, y_range, Re_field_x',
        title = "Re(Eₓ)", xlabel = "x", ylabel = "y", linewidth=0)
    scatter!(p1, atom_x, atom_y, marker = (:circle, 3, :white), label = "Atoms")
    p2 = contourf(x_range, y_range, Abs_field_x',
        title = "Abs(Eₓ)", xlabel = "x", ylabel = "y", linewidth=0)
    scatter!(p2, atom_x, atom_y, marker = (:circle, 3, :white), label = "Atoms")
    p3 = contourf(x_range, y_range, Re_field_y',
        title = "Re(Eᵧ)", xlabel = "x", ylabel = "y", linewidth=0)
    scatter!(p3, atom_x, atom_y, marker = (:circle, 3, :white), label = "Atoms")
    p4 = contourf(x_range, y_range, Abs_field_y',
        title = "Abs(Eᵧ)", xlabel = "x", ylabel = "y", linewidth=0)
    scatter!(p4, atom_x, atom_y, marker = (:circle, 3, :white), label = "Atoms")
    p5 = contourf(x_range, y_range, Re_field_z',
        title = "Re(E_z)", xlabel = "x", ylabel = "y", linewidth=0)
    scatter!(p5, atom_x, atom_y, marker = (:circle, 3, :white), label = "Atoms")
    p6 = contourf(x_range, y_range, Abs_field_z',
        title = "Abs(E_z)", xlabel = "x", ylabel = "y", linewidth=0)
    scatter!(p6, atom_x, atom_y, marker = (:circle, 3, :white), label = "Atoms")

    # Combine the plots into a 3x2 layout.
    plot(p1, p2, p3, p4, p5, p6, layout = (3, 2), size=(900,1100))
end

AtomicArrays.field.transmission_reg(field, field_func, coll,
                                    sigmas_m; samples=100)[1]
AtomicArrays.field.transmission_plane(field, field_func, coll,
                                    sigmas_m; samples=100)[1]

AtomicArrays.field.scattered_field([0.1, 0.1, 0.1], coll, sigmas_m)       # for FourLevelAtomCollection
test = AtomicArrays.field.scattered_field([[0.1, 0.1, 0.1],[0.2,0.2,0.2]], coll, sigmas_m)       # for FourLevelAtomCollection
AtomicArrays.field.scattered_field([[0.1, 0.1, 0.1],[0.2,0.2,0.2]], Ref(coll), Ref(sigmas_m))       # for FourLevelAtomCollection
AtomicArrays.field.total_field.(Ref(field_func), [[0.1, 0.1, 0.1],[0.2,0.2,0.2]], Ref(field), Ref(coll), Ref(sigmas_m))
test_0 = AtomicArrays.field.total_field(field_func, [[0.1, 0.1, 0.1],[0.2,0.2,0.2]], field, coll, sigmas_m)

sum(AtomicArrays.field.intensity.(test))
norm.(test).^2

begin
    pop_e_minus_mf = reshape(vcat(real(
        [AtomicArrays.fourlevel_meanfield.mapexpect(
        AtomicArrays.fourlevel_meanfield.smm, state_mf_t, n, 1, 1
    ) for n in 1:N])...
    ), (length(tspan),N))
    pop_e_0_mf = reshape(vcat(real(
        [AtomicArrays.fourlevel_meanfield.mapexpect(
        AtomicArrays.fourlevel_meanfield.smm, state_mf_t, n, 2, 2
    ) for n in 1:N])...
    ), (length(tspan),N))
    pop_e_plus_mf = reshape(vcat(real(
        [AtomicArrays.fourlevel_meanfield.mapexpect(
        AtomicArrays.fourlevel_meanfield.smm, state_mf_t, n, 3, 3
    ) for n in 1:N])...
    ), (length(tspan),N))
end


let
    p_pop = plot(layout=(1, 3), size=(1000, 300),
        title="Populations in excited sublevels", xlabel="t", ylabel="Population")

    # sublevel m=-1
    plot!(p_pop[1], t, pop_e_minus, lab=PermutedDimsArray(hcat(["Atom $n" for n=1:N]), (2,1)), 
        title="m = -1", linewidth=2)
    plot!(p_pop[1], t, pop_e_minus_mf, lab=PermutedDimsArray(hcat(["Atom mf $n" for n=1:N]), (2,1)), 
        title="m = -1", linewidth=2)

    # sublevel m=0
    plot!(p_pop[2], t, pop_e_0, lab=PermutedDimsArray(hcat(["Atom $n" for n=1:N]), (2,1)),
        title="m = 0", linewidth=2)
    plot!(p_pop[2], t, pop_e_0_mf, lab=PermutedDimsArray(hcat(["Atom mf $n" for n=1:N]), (2,1)),
        title="m = 0", linewidth=2)

    # sublevel m=+1
    plot!(p_pop[3], t, pop_e_plus, lab=PermutedDimsArray(hcat(["Atom $n" for n=1:N]), (2,1)),
        title="m = +1", linewidth=2)
    plot!(p_pop[3], t, pop_e_plus_mf, lab=PermutedDimsArray(hcat(["Atom mf $n" for n=1:N]), (2,1)),
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