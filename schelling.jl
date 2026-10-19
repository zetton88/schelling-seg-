using Agents
using GLMakie
using Random
using Statistics

# One-time setup (run in the Julia REPL):
#   using Pkg; Pkg.add(["Agents", "GLMakie"])
# GLMakie is only needed for the interactive part at the bottom.

@agent struct SchellingAgent(GridAgent{2})
    mood::Bool = false   # true = happy
    group::Int
end

function schelling_step!(agent, model)
    # count same-group neighbors (8-neighborhood by default)
    same = 0
    for nb in nearby_agents(agent, model)
        same += (nb.group == agent.group)
    end
    if same >= model.min_to_be_happy
        agent.mood = true
    else
        agent.mood = false
        move_agent_single!(agent, model)   # jump to a random empty site
    end
    return
end

function initialize(; n_agents = 3000, gridsize = (60, 60),
                      min_to_be_happy = 3, seed = 41)
    space = GridSpaceSingle(gridsize; periodic = true)
    properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Random.Xoshiro(seed)
    model = StandardABM(
        SchellingAgent, space;
        agent_step! = schelling_step!,
        properties, rng,
        container = Dict,
        scheduler = Schedulers.Randomly(),  # asynchronous, random order each step
    )
    for n in 1:n_agents
        add_agent_single!(SchellingAgent, model; group = isodd(n) ? 1 : 2)
    end
    return model
end

# --- Measurements -----------------------------------------------------------

# Mean fraction of same-group neighbors, averaged over agents with >= 1 neighbor
function segregation(model)
    fracs = Float64[]
    for a in allagents(model)
        nbs = collect(nearby_agents(a, model))
        isempty(nbs) && continue
        push!(fracs, count(nb -> nb.group == a.group, nbs) / length(nbs))
    end
    return isempty(fracs) ? 0.0 : mean(fracs)
end

frac_happy(model) = mean(a.mood for a in allagents(model))

# --- Headless run: track segregation and happiness over time ---------------

function run_headless(; steps = 100, kwargs...)
    model = initialize(; kwargs...)
    _, mdf = run!(model, steps;
                  mdata = [segregation, frac_happy],
                  when = true)
    return mdf   # DataFrame with columns :time, :segregation, :frac_happy
end

# Parameter sweep over the happiness threshold
function sweep(thresholds = 0:6; steps = 100)
    results = Dict{Int, Float64}()
    for t in thresholds
        mdf = run_headless(; steps, min_to_be_happy = t)
        results[t] = last(mdf.segregation)
    end
    return results
end
function grid_matrix(model)
    M = zeros(Int, spacesize(model))
    for a in allagents(model)
        M[a.pos[1], a.pos[2]] = a.group
    end
    return M
end

function watch(; steps = 300, delay = 0.05, kwargs...)
    model = initialize(; kwargs...)
    M = Observable(grid_matrix(model))
    title = Observable("step 0")
    fig = Figure()
    ax = Axis(fig[1, 1]; title = title, aspect = DataAspect())
    hidedecorations!(ax)
    heatmap!(ax, M; colormap = [:white, :royalblue, :orange], colorrange = (0, 2))
    screen = display(fig)
    for t in 1:steps
        step!(model, 1)
        M[] = grid_matrix(model)
        title[] = "step $t   segregation = $(round(segregation(model), digits = 3))"
        sleep(delay)
    end
    # when run as a script (not in a REPL), keep the window open until closed
    isinteractive() || wait(screen)
    return model
end

# Runs automatically whenever the file is run. Edit the arguments to taste,
# or comment this line out if you only want the functions loaded.
watch(steps = 100, min_to_be_happy = 5)
