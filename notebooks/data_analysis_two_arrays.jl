### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
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
    using CollectiveSpins
    using QuantumOptics
    using FFTW
    using BenchmarkTools, Interpolations
    using PlutoUI, LaTeXStrings
    using Makie, CairoMakie, WGLMakie, JSServe, Observables
    using LinearAlgebra, EllipsisNotation
    using HDF5, FileIO

    using Revise
    using AtomicArrays
    const EMField = AtomicArrays.field_module.EMField
    const effective_constants = AtomicArrays.effective_interaction_module.effective_constants

    Page(exportable=true, offline=true)
end

# ╔═╡ 0b04054e-ee1d-49fe-93f8-116182c326b1
md"""
# Analysis of $\sigma_{tot}$ on different parameters
"""

# ╔═╡ a47b7f22-9619-42c2-9ff2-2ecf82026415
PlutoUI.TableOfContents()

# ╔═╡ 28f9e5d7-fd1b-4148-a24d-64394bc8a179
md"""
## Loading data
"""

# ╔═╡ 487bb476-79fb-4d6b-9bd2-f1edf893df5c
"""
* `args`: `N_x`, `N_y`, `lat_type`, `eq_type`, `dim_vars`, `path_data`
* `output`: `filename_fs`, `filename_obj`
"""
function filename_create(args)
    N_x, N_y, lat_type, eq_type, dim_vars, path_data = args
    filename_fs = ("fs" * string(dim_vars) * "D" * "_" * lat_type * "_" * string(N_x)
                   * "x" * string(N_y) * "_" * eq_type * ".h5")
    filename_obj = ("obj" * string(dim_vars) * "D" * "_" * lat_type * "_" * string(N_x)
                    * "x" * string(N_y) * "_" * eq_type * ".h5")
    @assert isfile(path_data * filename_fs) "There is no file $filename_fs in the directory $path_data"
    @assert isfile(path_data * filename_obj) "There is no file $filename_obj in the directory $path_data"
    return filename_fs, filename_obj
end

# ╔═╡ 97fdfad9-5a87-45bb-b292-2d50dff46f2c
"""
* `args`: `path_data::String, file_fs::String, file_obj::String`
* `output`: `dict_fs, dict_obj`
"""
function load_dict(path_data::String, file_fs::String, file_obj::String)
    dict_fs = load(path_data * file_fs)
    dict_obj = load(path_data * file_obj)
    if length(dict_fs["order"]) != length(size(dict_fs["sigma_tot"]))
        keys_dict_fs = [isassigned(dict_fs.keys, i) ? dict_fs.keys[i] : "0"
                        for i in 1:length(dict_fs.keys)]
        deleteat!(dict_fs["order"], findall(x -> x == true,
            [isempty(findall(x -> x == item, keys_dict_fs))
             for item in dict_fs["order"]]))
    end
    if length(dict_obj["order"]) != length(size(dict_obj["obj"]))
        keys_dict_obj = [isassigned(dict_obj.keys, i) ? dict_obj.keys[i] : "0"
                         for i in 1:length(dict_obj.keys)]
        deleteat!(dict_obj["order"], findall(x -> x == true,
            [isempty(findall(x -> x == item, keys_dict_obj))
             for item in dict_obj["order"]]))
    end
    return dict_fs, dict_obj
end

# ╔═╡ 45deca37-2607-400c-b6bc-096e902c3dfe
md"""
### Parameters
"""

# ╔═╡ 9ade4a1e-28a2-4e8c-a079-64c8e01c91c8
# ╠═╡ show_logs = false
begin

    """Parameters"""

    const c_light = 1.0
    const lam_0 = 1.0
    const k_0 = 2 * π / lam_0
    const om_0 = 2.0 * pi * c_light / lam_0

    const Nx = 4
    const Ny = 4

    const LAT_TYPE = "lat"
    const EQ_TYPE = "mpc"
    const DIM_VARS = 4

    const NMAX = 30

    const PATH_FIGS, PATH_DATA = AtomicArrays.misc_module.path()

end

# ╔═╡ d8cc3527-e0bc-4cd6-9f19-a2788ffea6e7
md"""
### Defining files, loading dicts, interpolation
"""

# ╔═╡ 6682fc31-36f7-444f-bd16-14648f8bd8d1
begin
    # File names
    args_files = [Nx, Ny, LAT_TYPE, EQ_TYPE, DIM_VARS, PATH_DATA]
    const FILE_FS, FILE_OBJ = filename_create(args_files)

    # Loading data
    dict_fs, dict_obj = load_dict(PATH_DATA, FILE_FS, FILE_OBJ)
    dict_obj["order"]
end

# ╔═╡ 3d9d8489-9087-45e1-bc31-d3727d94412b
begin
    # Interpolation
    if LAT_TYPE == "dimer"
        σ_tot = LinearInterpolation(
            (dict_fs["E"], dict_fs["L"], dict_fs["a_1"], dict_fs["delt"], [0, 1]),
            dict_fs["sigma_tot"])
        obj = LinearInterpolation(
            (dict_obj["E"], dict_obj["L"], dict_obj["a_1"], dict_obj["delt"]),
            dict_obj["obj"])

        # Variables
        E = range(dict_obj["E"][1], dict_obj["E"][end], NMAX)
        L = range(dict_obj["L"][1], dict_obj["L"][end], NMAX)
        a_1 = range(dict_obj["a_1"][1], dict_obj["a_1"][end], NMAX)
        delt = range(dict_obj["delt"][1], dict_obj["delt"][end], NMAX)

        "Interpolation's done"
    elseif LAT_TYPE == "lat"
        σ_tot = LinearInterpolation(
            (dict_fs["E"], dict_fs["L"], dict_fs["d"], dict_fs["delt"], [0, 1]),
            dict_fs["sigma_tot"])
        obj = LinearInterpolation(
            (dict_obj["E"], dict_obj["L"], dict_obj["d"], dict_obj["delt"]),
            dict_obj["obj"])

        # Variables
        E = range(dict_obj["E"][1], dict_obj["E"][end], NMAX)
        L = range(dict_obj["L"][1], dict_obj["L"][end], NMAX)
        a_1 = range(dict_obj["d"][1], dict_obj["d"][end], NMAX)
        delt = range(dict_obj["delt"][1], dict_obj["delt"][end], NMAX)
        "Interpolation's done"
    end
end

# ╔═╡ b41f48bf-7e0f-4577-ae20-b4611b4580dc
md"""
## Plotting
"""

# ╔═╡ 50252dab-8293-40e2-a837-0b13db8e3d24
begin
    # Find optimal parameters
    if LAT_TYPE == "dimer"
        obj_max = maximum(dict_obj["obj"])
        opt_idx = indexin(obj_max, dict_obj["obj"])[1]
        E_opt = dict_obj["E"][opt_idx[1][1]]
        L_opt = dict_obj["L"][opt_idx[2][1]]
        a_1_opt = dict_obj["a_1"][opt_idx[3][1]]
        delt_opt = dict_obj["delt"][opt_idx[4][1]]
        "optimal parameters"
    elseif LAT_TYPE == "lat"
        obj_max = maximum(dict_obj["obj"])
        opt_idx = indexin(obj_max, dict_obj["obj"])[1]
        E_opt = dict_obj["E"][opt_idx[1][1]]
        L_opt = dict_obj["L"][opt_idx[2][1]]
        a_1_opt = dict_obj["d"][opt_idx[3][1]]
        delt_opt = dict_obj["delt"][opt_idx[4][1]]
        "optimal parameters"
    end
end

# ╔═╡ 03c1dbe4-8837-42f0-ab25-3a08415d510e
md"""
### Maximum value of objective function and parameters

Objective max:
$(maximum(dict_obj["obj"]))

E = $(E_opt)

L = $(L_opt)

a\_1 = $(a_1_opt)

δ = $(delt_opt)

"""

# ╔═╡ b1c34a5d-60c5-4930-ba6e-245c03a064b5
dict_obj["order"]

# ╔═╡ c8eb0729-fc4a-42c5-895f-c79f55aa2128
md"""
### Objective function
"""

# ╔═╡ 8a0abedb-5205-4783-b9ec-92b86bac2fbf
begin
    WGLMakie.activate!()
    "Activate WGLMakie"
end

# ╔═╡ 64230562-a8e8-49fa-953d-845f543c9bbd
function volume_obj(E_i)
    App() do session::Session
        slider_delt = JSServe.Slider(1:NMAX)

        delt_val = map(slider_delt) do i
            return delt[i]
        end

        cube = obj(E_i, L, a_1, delt)
        colorrange = (minimum(cube), maximum(cube))

        slice = map(slider_delt) do i
            return cube[:, :, i]
        end

        line = [maximum(obj(E_i, L, a_1, delt_i)) for delt_i in delt]

        fig = Figure(resolution=(1000, 800))
        colormap = to_colormap(:lightrainbow)  # colormap
        colormap[1] = Makie.RGBA{Float32}(1, 1, 1, 0)  # create opacity

        # Plot cube
        ax, cplot = volume(fig[1, 1], L, a_1, delt, cube,
            algorithm=:absorption, absorption=4.0f0, colormap=colormap,
            colorrange=colorrange,
            transparency=true, axis=(; title="Absorption"))

        # Plot rectangular
        rectplot = linesegments!(ax, Rect(minimum(L), minimum(a_1),
                maximum(L) - minimum(L),
                maximum(a_1) - minimum(a_1)),
            linewidth=2, color=:red)
        on(slider_delt) do idx
            translate!(rectplot, 0, 0, delt[idx])
        end

        # Plot slice
        ax_1, cm = heatmap(fig[1, 2], L, a_1, slice,
            colormap=:lightrainbow,
            axis=(; xlabel=L"$L$", ylabel=L"$a_1$", xlabelsize=25, ylabelsize=25))
        Colorbar(fig[2, 2], cm, vertical=false)

        # Plot line
        ax_2, lin = lines(fig[3, :], delt, line,
            linewidth=4, axis=(; xlabel=L"$\delta$", limits=(nothing, nothing, 0, maximum(obj(E, L, a_1, delt)))), xlabelsize=25)

        slider = DOM.div("delt: ", slider_delt, delt_val)
        dom = DOM.div(slider, fig)
        return JSServe.record_states(session, dom)
    end
end

# ╔═╡ ac564c14-c96e-4c73-a5c0-2e52d9853239
md"""
E\_i: 
$(@bind E_i PlutoUI.Slider(E))
"""

# ╔═╡ 5e0f3f83-2b95-4ca2-9909-9740ba03f05a
E_i

# ╔═╡ dbe7cb57-3f79-4c1f-9282-cfacf9a04b6a
volume_obj(E_i)

# ╔═╡ 54265506-37a9-4c3c-95cd-97097fdddeca
md"""
### Total scattering cross section
"""

# ╔═╡ b9159d0f-7998-444b-9e04-3350851f26d3
function volume_sc(E_i)
    App() do session::Session

        slider_delt = JSServe.Slider(1:NMAX)

        delt_val = map(slider_delt) do i
            return delt[i]
        end

        cube_1 = σ_tot(E_i, L, a_1, delt, 0)
        colorrange_1 = (minimum(cube_1), maximum(cube_1))
        cube_2 = σ_tot(E_i, L, a_1, delt, 1)
        colorrange_2 = (minimum(cube_2), maximum(cube_2))

        slice_1 = map(slider_delt) do i
            return cube_1[:, :, i]
        end
        slice_2 = map(slider_delt) do i
            return cube_2[:, :, i]
        end

        # Figure
        fig_s = Figure(resolution=(1000, 900))
        colormap = to_colormap(:lightrainbow)  # colormap
        colormap[1] = Makie.RGBA{Float32}(1, 1, 1, 0)  # create opacity

        # Plot cube
        ax_11, cplot_11 = volume(fig_s[1, 1], L, a_1, delt, cube_1,
            algorithm=:absorption, absorption=4.0f0, colormap=colormap,
            colorrange=colorrange_1,
            transparency=true, axis=(; type=Axis3, title=L"$\sigma_0$"))
        ax_12, cplot_12 = volume(fig_s[1, 2], L, a_1, delt, cube_2,
            algorithm=:absorption, absorption=4.0f0, colormap=colormap,
            colorrange=colorrange_2,
            transparency=true, axis=(; type=Axis3, title=L"$\sigma_\pi$"))
        # Plot rectangular
        rectplot_1 = linesegments!(ax_11, Rect(minimum(L), minimum(a_1),
                maximum(L) - minimum(L),
                maximum(a_1) - minimum(a_1)),
            linewidth=2, color=:red)
        rectplot_2 = linesegments!(ax_12, Rect(minimum(L), minimum(a_1),
                maximum(L) - minimum(L),
                maximum(a_1) - minimum(a_1)),
            linewidth=2, color=:red)
        on(slider_delt) do idx
            translate!(rectplot_1, 0, 0, delt[idx])
            translate!(rectplot_2, 0, 0, delt[idx])
        end

        # Plot slice
        ax_21, cm_1 = heatmap(fig_s[2, 1], L, a_1,
            slice_1, colormap=:lightrainbow,
            axis=(; xlabel=L"$L$", ylabel=L"$a_1$", xlabelsize=25, ylabelsize=25))
        Colorbar(fig_s[3, 1], cm_1, vertical=false)
        ax_22, cm_2 = heatmap(fig_s[2, 2], L, a_1,
            slice_2, colormap=:lightrainbow,
            axis=(; xlabel=L"$L$", ylabel=L"$a_1$", xlabelsize=25, ylabelsize=25))
        Colorbar(fig_s[3, 2], cm_1, vertical=false)

        slider = DOM.div("delt: ", slider_delt, delt_val)
        dom = DOM.div(slider, fig_s)
        return JSServe.record_states(session, dom)
    end
end

# ╔═╡ 82be5e81-9986-4c02-a6af-12012b00793f
md"""
E\_j: 
$(@bind E_j PlutoUI.Slider(E))
"""

# ╔═╡ ac009e68-4437-4666-84c2-762565fc5ae7
E_j

# ╔═╡ 7c1fa849-297f-4d17-acea-9262b1ccad02
volume_sc(E_j)

# ╔═╡ 220ed827-4c03-42a7-b32e-908907918367
function surface_sc(E_i)
    App() do session::Session
        slider_L = JSServe.Slider(1:NMAX)

        L_val = map(slider_L) do i
            return L[i]
        end

        surf_1 = map(slider_L) do i
            return σ_tot(E_i, L[i], a_1, delt, 0)
        end
        surf_2 = map(slider_L) do i
            return σ_tot(E_i, L[i], a_1, delt, 1)
        end

        fig = Figure(resolution=(600, 600))

        # Plot surfaces
        wireframe(fig[1, 1], a_1, delt, surf_1,
            axis=(type=Axis3, title=L"$\sigma_0$, $\sigma_\pi$",
                xlabel=L"a_1", ylabel=L"\delta", xlabelsize=24, ylabelsize=24,
                titlesize=30),
            color=:red)
        wireframe!(a_1, delt, surf_2,
            color=:blue)

        slider = DOM.div("L: ", slider_L, L_val)
        dom = DOM.div(slider, fig)
        return JSServe.record_states(session, dom)
    end
end

# ╔═╡ 8acb3337-eac6-43a9-ab4f-7b50400130c2
md"""
E\_k: 
$(@bind E_k PlutoUI.Slider(E))
"""

# ╔═╡ 03824a9b-6317-4a38-9fe6-920599327018
E_k

# ╔═╡ d03d7b31-736c-4557-8c1e-8101c41eed30
surface_sc(E_k)

# ╔═╡ Cell order:
# ╟─0b04054e-ee1d-49fe-93f8-116182c326b1
# ╠═2fb09b00-0c7a-11ed-1dd6-11c64e10c2cd
# ╠═060c13ae-9213-4ecc-87b0-a242391e5548
# ╠═a47b7f22-9619-42c2-9ff2-2ecf82026415
# ╟─28f9e5d7-fd1b-4148-a24d-64394bc8a179
# ╠═487bb476-79fb-4d6b-9bd2-f1edf893df5c
# ╠═97fdfad9-5a87-45bb-b292-2d50dff46f2c
# ╟─45deca37-2607-400c-b6bc-096e902c3dfe
# ╠═9ade4a1e-28a2-4e8c-a079-64c8e01c91c8
# ╟─d8cc3527-e0bc-4cd6-9f19-a2788ffea6e7
# ╠═6682fc31-36f7-444f-bd16-14648f8bd8d1
# ╠═3d9d8489-9087-45e1-bc31-d3727d94412b
# ╟─b41f48bf-7e0f-4577-ae20-b4611b4580dc
# ╠═50252dab-8293-40e2-a837-0b13db8e3d24
# ╟─03c1dbe4-8837-42f0-ab25-3a08415d510e
# ╟─b1c34a5d-60c5-4930-ba6e-245c03a064b5
# ╟─c8eb0729-fc4a-42c5-895f-c79f55aa2128
# ╟─8a0abedb-5205-4783-b9ec-92b86bac2fbf
# ╟─64230562-a8e8-49fa-953d-845f543c9bbd
# ╟─ac564c14-c96e-4c73-a5c0-2e52d9853239
# ╟─5e0f3f83-2b95-4ca2-9909-9740ba03f05a
# ╠═dbe7cb57-3f79-4c1f-9282-cfacf9a04b6a
# ╟─54265506-37a9-4c3c-95cd-97097fdddeca
# ╟─b9159d0f-7998-444b-9e04-3350851f26d3
# ╟─82be5e81-9986-4c02-a6af-12012b00793f
# ╟─ac009e68-4437-4666-84c2-762565fc5ae7
# ╠═7c1fa849-297f-4d17-acea-9262b1ccad02
# ╟─220ed827-4c03-42a7-b32e-908907918367
# ╟─8acb3337-eac6-43a9-ab4f-7b50400130c2
# ╟─03824a9b-6317-4a38-9fe6-920599327018
# ╠═d03d7b31-736c-4557-8c1e-8101c41eed30
