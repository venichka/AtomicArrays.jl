using Markdown
using JSServe, Observables
using JSServe: Session, evaljs, linkjs
using JSServe: @js_str, onjs, Button, Slider, Asset
using WGLMakie, AbstractPlotting
using JSServe.DOM

app = App() do session::Session
    cmap_button = Button("change colormap")
    algorithm_button = Button("change algorithm")
    algorithms = ["mip", "iso", "absorption"]
    algorithm = Observable(first(algorithms))
    dropdown_onchange = js"JSServe.update_obs($algorithm, this.options[this.selectedIndex].text);"
    algorithm_drop = DOM.select(DOM.option.(algorithms); class="bandpass-dropdown", onclick=dropdown_onchange)

    data_slider = Slider(LinRange(1f0, 10f0, 100))
    iso_value = Slider(LinRange(0f0, 1f0, 100))
    N = 100
    slice_idx = Slider(1:N)

    signal = map(Observables.async_latest(data_slider.value)) do α
        a = -1; b = 2
        r = LinRange(-2, 2, N)
        z = ((x,y) -> x + y).(r, r') ./ 5
        me = [z .* sin.(α .* (atan.(y ./ x) .+ z.^2 .+ pi .* (x .> 0))) for x=r, y=r, z=r]
        return me .* (me .> z .* 0.25)
    end

    slice = map(signal, slice_idx) do x, idx
        view(x, :, idx, :)
    end
    fig = WGLMakie.Figure()
# ambient=Vec3{Float64}(0.8),
    vol = WGLMakie.volume(fig[1,1], signal; algorithm=map(Symbol, algorithm),  isovalue=iso_value)

    colormaps = collect(AbstractPlotting.all_gradient_names)
    cmap = map(cmap_button) do click
        return colormaps[rand(1:length(colormaps))]
    end

    heat = WGLMakie.heatmap(fig[1, 2], slice, colormap=cmap)

    dom = md"""
    # Plots:
    $(DOM.div("data param", data_slider))
    $(DOM.div("iso value", iso_value))
    $(DOM.div("y slice", slice_idx))
    $(algorithm_drop)
    $(cmap_button)
    ---
    $(fig.scene)
    ---
    """#JSServe.DOM.div(JSServe.MarkdownCSS, JSServe.Styling,
    return JSServe.record_states(session, JSServe.DOM.div(JSServe.MarkdownCSS, JSServe.Styling, dom))
end

isdefined(Main, :server) && close(server)
server = JSServe.Server(app, "127.0.0.1", 8081)

display(app)