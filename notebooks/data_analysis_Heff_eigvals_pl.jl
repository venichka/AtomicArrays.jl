### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loadeds[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 2fb09b00-0c7a-11ed-1dd6-11c64e10c2cd
begin
    if pwd()[end-14:end] == "AtomicArrays.jl"
        PATH_ENV = "."
    else
        PATH_ENV = "../"
    end
	using Pkg
	Pkg.activate(PATH_ENV)
end

# ╔═╡ 060c13ae-9213-4ecc-87b0-a242391e5548
# ╠═╡ show_logs = false
begin
	using QuantumOptics
	using FFTW
	using BenchmarkTools, ProgressMeter, Interpolations
	using PlutoUI, LaTeXStrings, CairoMakie
	using HDF5, FileIO

	using Revise
	using AtomicArrays
	const EMField = AtomicArrays.field.EMField
	const effective_constants = AtomicArrays.effective_interaction.effective_constants
end

# ╔═╡ 0b04054e-ee1d-49fe-93f8-116182c326b1
md"""
# $H_{eff}$ eigenvalues and effective renormalization constants
"""

# ╔═╡ 9ade4a1e-28a2-4e8c-a079-64c8e01c91c8
# ╠═╡ show_logs = false
begin
	dag(x) = conj(transpose(x))
	#  Expectation values
	mapexpect(op, states, num) = map(s->(op(s)[num]), states)
	
	const PATH_FIGS, PATH_DATA = AtomicArrays.misc.path()
	PATH_DATA = replace(PATH_DATA, r"data_2arrays_mpc_mf/$"=>"data_effective/")

	# const PATH_FIG = "/home/nikita/Documents/Work/Projects/two_arrays/Figs/figs_eff_constants/"
	# const PATH_DATA = "/home/nikita/Documents/Work/Projects/two_arrays/Data/effective_constants/"
	
	"""Parameters"""
	c_light = 1.0
	lam_0 = 1.0
	k_0 = 2*π / lam_0
	om_0 = 2.0*pi*c_light / lam_0
	
	Nx = 200
	Ny = 200
	
	NMAX = 200
	
	"""Variables"""
	
	d1_iter = range(0.1, 2.0, NMAX)
	d2_iter = range(0.1, 2.0, NMAX)
	L_iter = range(0.1, 2.0, NMAX)
	gam_iter = range(0.001, 0.1, NMAX)
	Delt_iter = range(-0.5, 0.5, NMAX)
end

# ╔═╡ 3a11dbcd-37ed-49f8-8170-9d26fc58b346
begin
	"""Loading effective Γ and Ω"""
	
	args_eff = [d1_iter, Delt_iter]
	N_dim_eff = length(args_eff)

	Gamma_eff = h5open(PATH_DATA*"gamma_eff_"*string(Nx)*".h5", "r") do file
    	read(file, "gamma_eff")
	end
	Omega_eff = h5open(PATH_DATA*"omega_eff_"*string(Nx)*".h5", "r") do file
    	read(file, "omega_eff")
	end
	Omega_eff_py = h5open(PATH_DATA*"omega_eff_"*string(Nx)*"_py.h5", "r") do file
    	read(file, "omega_eff")
	end
	
	# Interpolate Γ and Ω effective
	
	Ω_int = LinearInterpolation((d1_iter, Delt_iter), Omega_eff);
	Γ_int = LinearInterpolation((d1_iter, Delt_iter), Gamma_eff);
	Ω_int_py = LinearInterpolation((d1_iter), Omega_eff_py);
end

# ╔═╡ 8d355c8d-3ba8-47e1-afb5-e67163b4076c
begin
	CairoMakie.activate!(type = "svg")
	x = d1_iter
	function omega_int(k_a)
	    y0 = Ω_int(x, k_a)
		fig = CairoMakie.Figure(resolution=(600,400))
		lines(fig[1,1], x, y0, label = "interpolated"; 
        axis=(;limits=(nothing, nothing, -3, 1), 
				xlabel=L"$d/\lambda_0$", ylabel=L"$\Omega_{eff}$", 
				xlabelsize=20, ylabelsize=20))
		lines!(fig[1,1], x, Ω_int_py(x)*100, label="python")
		# CairoMakie.ylims!(-3, 1) 
		fig
	end
end

# ╔═╡ c11bed04-fc09-4678-97d2-48d9ab5a5eee
@bind k_a PlutoUI.Slider(Delt_iter, default=Delt_iter[Nx ÷ 2])

# ╔═╡ 7b826dbe-2305-461c-9d89-e9e3fef0410a
begin
	print(k_a)
	omega_int(k_a)
end

# ╔═╡ 8273995d-ee61-477f-9cf4-284ae2b0cb96
fig_1 = omega_int(k_a)

# ╔═╡ ca65ec7d-c105-46ce-8fcc-6faeaee9e3e9
#save(PATH_FIGS * "eff_constants.pdf", fig_1) # here, you save your figure.

# ╔═╡ 33fddd35-bfd0-4fd3-89b8-0f1d1041f337
md"""
## Compute Hamiltonian and its eigenstates

We consider Hamiltonian of the homogenized system of two atoms, which frequencies and decay rates renormalized with $\Omega_{eff}^{(i)}$ and $\Gamma_{eff}^{(i)}$:

$$H = \hbar \sum_{i=1}^2 \left[ \left(\omega_i - \omega_L + \Omega_{eff}^{(i)} \right) \sigma_i^+ \sigma_i - \left( \Omega_i \sigma_i + \Omega_i^* \sigma_i^+ \right) \right] + \hbar \sum_{i,j} \left( \Omega_{ij} - \frac{i}{2} \Gamma_{ij} \right) \sigma_i^+ \sigma_j,$$
where
$$\Gamma_{ii} = \Gamma_{eff}^{(i)}$$
"""

# ╔═╡ 2a635da4-ad17-42a6-8368-7a938960d6df
"""
* `args`: d1, d2, Delt, L, γ, E_ampl, DIRECTION
"""
function eigen_numerical(args)
    
    d1, d2, Delt, L, γ, E_ampl, DIRECTION = args
    

    μ = [(i < 0) ? [1.0, 0, 0.0] : [1.0, 0.0im, 0.0] for i = 1:2]


    """Calculate the collective shift depending on the lattice constant"""
    # Don't forget to multiply by γ

    Omega_1 = γ .* Ω_int(d1, - Delt)
    Gamma_1 = γ .*abs.(Γ_int(d1, - Delt))
    Omega_2 = γ .*Ω_int(d2, Delt)
    Gamma_2 = γ .*abs.(Γ_int(d2, Delt))

    pos = [[0,0,-0.5*L], [0,0,0.5*L]]
    S_1 = Spin(pos[1], delta=Omega_1 - Delt)
    S_2 = Spin(pos[2], delta=Omega_2 + Delt)
    S = SpinCollection([S_1, S_2], μ, gammas=[Gamma_1, Gamma_2])
    N = length(S.spins)


    # Incident field parameters

    om_f = om_0
    E_kvec = om_f/c_light
    E_width = 0.3*d1*sqrt(Nx*Ny)
    if (DIRECTION == "right")
        E_pos0 = [0.0,0.0,0.0]
        E_polar = [1.0, 0im, 0.0]
        E_angle = [0.0, 0.0]  # {θ, φ}
    elseif (DIRECTION == "left")
        E_pos0 = [0.0,0.0,0.0]
        E_polar = [-1.0, 0im, 0.0]
        E_angle = [π, 0.0]  # {θ, φ}
    else
        println("DIRECTION wasn't specified")
    end


    E_inc = EMField(E_ampl, E_kvec, E_angle, E_polar;
                         position_0 = E_pos0, waist_radius = E_width)
    #em_inc_function = AtomicArrays.field.gauss
    em_inc_function = AtomicArrays.field.plane
    
    # Atoms -- field interaction
    E_vec = [em_inc_function(spin.position, E_inc) for spin in S.spins]
    Om_R = AtomicArrays.field.rabi(E_vec, S.polarizations)

    """System Hamiltonian"""

    Γ, J = AtomicArrays.quantum.JumpOperators(S)
    Jdagger = [dagger(j) for j = J]
    Ω = AtomicArrays.interaction.OmegaMatrix(S)
    H = AtomicArrays.quantum.Hamiltonian(S) - sum(Om_R[j] * J[j] +
                                                              conj(Om_R[j]) * Jdagger[j]
                                                              for j = 1:N)
    H_eff = AtomicArrays.quantum.Hamiltonian_eff(S) - sum(Om_R[j] * J[j] +
                                                              conj(Om_R[j]) * Jdagger[j]
                                                              for j = 1:N)
    
    return eigenenergies(dense(H)), eigenenergies(dense(H_eff); warning=false)
end

# ╔═╡ dbafda8e-4546-46c6-8e2e-6dad701a1daf
begin
	E0 = 10.0.^range(-5,-2,NMAX)
	Delt0_iter = range(0.0, 0.5, NMAX)
	dirs = ["right" =>"right", "left"=>"left"]#Dict(zip(["right", "left"], ["right", "left"]))
	function eigenvalues_plot(dir, Δ, d1, d2, L, γ)
	    y = reduce(vcat,transpose.([eigen_numerical([d1, d2, Δ, L, γ, E0[i], dir])[2] 
	                    for i = 1:NMAX]))
		fig = Figure(resolution=(600,400))
		ax1, p1 = scatter(fig[1,1], E0, real(y[:,1]), axis=(;title=L"\mathrm{Real } \lambda", xlabel=L"E"))
		ax2, p2 = scatter(fig[1,2], E0, imag(y[:,1]), axis=(;title=L"\mathrm{Imag } \lambda", xlabel=L"E"))
		for i in 2:4
			scatter!(fig[1,1], E0, real(y[:,i]))
			scatter!(fig[1,2], E0, imag(y[:,i]))
		end
		return fig
	end 
end

# ╔═╡ 70571b6d-153f-4e63-9041-48baec739395
md"""
Direction: $(@bind dir PlutoUI.Select(dirs))
  Delta:
$(@bind Δ PlutoUI.Slider(Delt0_iter, default=Delt0_iter[1]))
  d1: 
$(@bind d1 PlutoUI.Slider(d1_iter, default=d1_iter[NMAX ÷ 2]))

d2: 
$(@bind d2 PlutoUI.Slider(d1_iter, default=d1_iter[NMAX ÷ 2]))
L: 
$(@bind L PlutoUI.Slider(d1_iter, default=d1_iter[NMAX ÷ 2]))

gamma: 
$(@bind γ PlutoUI.Slider(gam_iter, default=gam_iter[1]))
"""

# ╔═╡ 1a72eef8-ef11-4230-8d04-ec4830cbc724
(dir, Δ, d1, d2, L, γ)

# ╔═╡ 14ff182c-3c95-44ab-9b07-2c2f12005daf
test = eigenvalues_plot(dir, Δ, d1, d2, L, γ)

# ╔═╡ Cell order:
# ╟─0b04054e-ee1d-49fe-93f8-116182c326b1
# ╠═2fb09b00-0c7a-11ed-1dd6-11c64e10c2cd
# ╠═060c13ae-9213-4ecc-87b0-a242391e5548
# ╠═9ade4a1e-28a2-4e8c-a079-64c8e01c91c8
# ╠═3a11dbcd-37ed-49f8-8170-9d26fc58b346
# ╠═8d355c8d-3ba8-47e1-afb5-e67163b4076c
# ╠═c11bed04-fc09-4678-97d2-48d9ab5a5eee
# ╠═7b826dbe-2305-461c-9d89-e9e3fef0410a
# ╠═8273995d-ee61-477f-9cf4-284ae2b0cb96
# ╠═ca65ec7d-c105-46ce-8fcc-6faeaee9e3e9
# ╟─33fddd35-bfd0-4fd3-89b8-0f1d1041f337
# ╟─2a635da4-ad17-42a6-8368-7a938960d6df
# ╠═dbafda8e-4546-46c6-8e2e-6dad701a1daf
# ╟─70571b6d-153f-4e63-9041-48baec739395
# ╠═1a72eef8-ef11-4230-8d04-ec4830cbc724
# ╠═14ff182c-3c95-44ab-9b07-2c2f12005daf
