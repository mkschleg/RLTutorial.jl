### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 3c50149f-1cb5-4ddb-afaf-a5ca3f1f68e2
using Plots, ProgressMeter, Statistics, Random, PlutoUI

# ╔═╡ 9d216976-c95f-11eb-0501-37164fe97a8f
md"""
# RL Interface

There are countless ways to setup the interface between the agent and its environment. In this tutorial, I advocate for the interface developed by Brian Tanner, Adam White, and Rich Sutton through RLGlue. I've been using this basic framework for an interface for some time, and have always found it to align well with the RL problem and helps reduce error in managing what the agent can and cannot see about the environment.

## RLGlue

Many people will be familiar with the OpenAI Gym API for building the interface of agents and their world. While this is easily doable in Julia, I tend to use the RLGlue interface developed by Brian Tanner and Adam White while at the University of Alberta. 

## Environment


Lets start with the world. The interface is easy enough
"""

# ╔═╡ 2707871e-db71-4a6f-8356-6454360729d1
begin
	abstract type AbstractEnvironment end

	function start! end
	function step! end

	function get_state end
	function get_reward end
	function is_terminal end
end

# ╔═╡ 5817820a-cf09-4c18-a79a-2033ff9b141a
md"""
Now this is a really basic interface, where the user has total control over what start, and step return. While this may be beneficial in some instances, we often want our environments to accept and return a consistent set of information. Specifically, we want our agents to accept an action and return the resulting transition tuple.
"""

# ╔═╡ 5bc4a293-57c6-4ef0-98e9-31c0f171e980
md"""
Now when a user is implementing a type of AbstractEnvironment, they have to implement the `_start!`, `_step!`, `get_state`, `get_reward`, and `is_terminal` functions and they can use start! and step! to ensure similarity in return structure. You may be noticing the `args...` parameters signifying a variable number of args in the function call. While not critical, these can help the user define custom behavior that isn't implemented in the basic functions here. For instance, say a user wanted to maintain their own random number generator (which I am often known to do if I'm taking advantage of Julia's threading (pre 1.3) to do experiments).


Now that we have a basic interface built. Lets do a simple example on how you could implement a simple Markov chain.
"""

# ╔═╡ 0e854488-9048-43d6-975a-cfe8248c0bfe
begin
	"""
    	MarkovChain(num_states)
	"""

	mutable struct MarkovChain <: AbstractEnvironment
    	size::Int # The size of the markov chain
    	state::Int
    	MarkovChain(size) = new(size, size ÷ 2) # the ÷ symbol throws away the remainder.
	end

	function _start!(env::MarkovChain)
		env.state = env.size ÷ 2
		range = ((env.size ÷ 2) - env.size ÷ 4):((env.size ÷ 2)+(env.size ÷ 4))
		env.state = rand(range)
	end

	function _step!(env::MarkovChain, action)

		if action == 1 # LEFT
			env.state -= 1
		elseif action == 2 # RIGHT
			env.state += 1
		else
			throw("Error")
		end
	end

	get_actions(env::MarkovChain) = (1, 2) # Remember julia starts indexing at 1!

	get_state(env::MarkovChain) = env.state
	get_reward(env::MarkovChain) = env.state == env.size ? 1.0 : -1.0
	is_terminal(env::MarkovChain) = env.state == env.size || env.state == 1
end

# ╔═╡ fb537595-2405-4f93-ab9e-0d14d6421c33
begin
	function start!(env::AbstractEnvironment, args...)
    	_start!(env, args...)
    	return get_state(env)
	end

	function step!(env::AbstractEnvironment, action, args...)
    	_step!(env, action, args...)
    	return get_state(env), get_reward(env), is_terminal(env)
	end

	function _start! end
	function _step! end
end

# ╔═╡ 4da83595-4117-47e9-857e-c60f4345d029
md"We now have a simple environment API, and a Markov Chain to play with. Lets do a quick simulation to make sure things are working correctly!"

# ╔═╡ 376c555d-ca83-4695-89e9-d1afdc30c7c2
md"""
## Agent

Now we need to move on to building an agent that will act in the environment. We are going to follow the same idea as in the environment to implement a basic tabular QLearning agent (if you need a review see [Rich's RL Book](http://incompleteideas.net/book/the-book-2nd.html).
"""

# ╔═╡ 130c152e-ff42-4722-b06f-5007245fedea
abstract type AbstractAgent end

# ╔═╡ 06ea1f1c-5272-4c16-bdb4-7013dec5994a
mutable struct TabularQLearningAgent <: AbstractAgent
    values::Array{Float64, 2}
    α::Float64
    γ::Float64
    ϵ::Float64
    prev_s::Int
    action::Int
    TabularQLearningAgent(size, num_actions, learning_rate, discount, epsilon) = 
        new(zeros(size, num_actions), learning_rate, discount, epsilon, 0, 0)
end

# ╔═╡ 8cfec52a-747c-4209-8b7e-c1d81905a602
function reset!(agent::TabularQLearningAgent)
    agent.values .= randn(size(agent.values)...)
end

# ╔═╡ 25860151-bb59-4997-9c1a-966251bbc597
function start!(agent::TabularQLearningAgent, state)
#     fill!(agent.values, 0.5)
    agent.prev_s = state
    agent.action = if rand() > agent.ϵ
        findmax(agent.values[state, :])[2]
    else
        rand(1:size(agent.values)[2])
    end
    
    agent.action
end

# ╔═╡ 0273ea0e-5139-4a86-9e21-dd9473d04380
function step!(agent::TabularQLearningAgent, state, rew, term)
    
    # update action-state values
    q = agent.values[agent.prev_s, agent.action]
    q_prime = maximum(agent.values[state, :])
    
    # Notice the difference in update for terminal and not terminal!
    δ = if !term
        rew + agent.γ * q_prime - q
    else
        rew - q
    end
    agent.values[agent.prev_s, agent.action] += agent.α * δ

    agent.prev_s = state
    agent.action = if rand() > agent.ϵ
        findmax(agent.values[state, :])[2]
    else
        rand(1:size(agent.values)[2])
    end
    
    agent.action
end

# ╔═╡ 9ae9934a-2f4e-4776-9f14-dc5082e3828f
let
	env = MarkovChain(10)
	states = Int[]
	s = start!(env)
	push!(states, s)

	while !is_terminal(env)
    	s, r, t = step!(env, rand(get_actions(env)))
    	push!(states, s)
	end

	states
end

# ╔═╡ 5eaadfc0-dcdc-45df-b05d-09aae82b2cb7
md"""
## Experiments/Episodes

We've implemented an agent and an environment, now we need to write some code to glue these together to run an episode. The function `run_episode!` is a simplified version of the implementation found in [MinimalRLCore](https://github.com/mkschleg/MinimalRLCore.jl/blob/master/src/episode.jl)
"""

# ╔═╡ c51b634d-c66d-45f9-a39f-f8f1dac11e37
function run_episode!(env, agent)
    s = start!(env)
    a = start!(agent, s)

    total_rews = 0.0
    steps = 0
    
    while !is_terminal(env)
        s, r, t = step!(env, a)
        a = step!(agent, s, r, t)
        total_rews += r
        steps += 1
    end
    total_rews, steps
end

# ╔═╡ e0c961b6-c6ad-4cde-acde-3c7d4a6d5ad2
function run_experiment(num_runs, num_episodes, seed=1032; markov_chain_size=100, α=0.1, γ=0.9, ϵ=0.01)

    Random.seed!(seed)
    
    env = MarkovChain(markov_chain_size)
    agent = TabularQLearningAgent(markov_chain_size, 2, α, γ, ϵ)

    reset!(agent)

    returns = zeros(num_episodes, num_runs)
    steps = zeros(Int, num_episodes, num_runs)

    @showprogress 0.1 "Runs: " for r in 1:num_runs
        reset!(agent)
        for i in 1:num_episodes
            total_rew, num_steps = run_episode!(env, agent)
            returns[i, r] = total_rew
            steps[i, r] = num_steps
        end
    end
    plot(mean(returns; dims=2), ribbon=std(returns; dims=2), legend=false)
end

# ╔═╡ 5867177d-305e-424b-806e-d27b758fd954
md"""
Number of runs: $(@bind runs NumberField(1:10_000, default=100))
Number of Episodes: $(@bind episodes NumberField(1:1_000, default=500))
Epsilon: $(@bind epsilon Slider(0.0:0.1:0.9, default=0.1, show_value=true))
"""

# ╔═╡ 7f41a327-ffea-41ee-aea5-58dabc53fe6a
run_experiment(runs, episodes; ϵ=epsilon)

# ╔═╡ d7eff463-353c-469d-98f6-607bc68ea3ed
md"""
You may have noticed the long runtime after the progress bar seemed complete. This has to do w/ one of the main issues w/ Julia (or the time-to-first-plot issue). What is happening is a bunch of code in the `plot` function is having to be compiled (plotting is really complicated). While annoying, after subsequent evaluations of the cell you can see that plotting is now fast. While out of scope for this document, there are a few fixes for this issue with the recommended issue being [PackageCompiler.jl](https://github.com/JuliaLang/PackageCompiler.jl). The way this works is by precompiling packages to put into the stdlib (much like those packages found in Base), and will significantly speed up the first run of `plot`.

Now you have a function encapsulating your entire experiment to play around with! This is a pattern you will see often in Julia. Instead of scripting (like in python) you will want to wrap pieces of code into functions so the compiler can work its optimization magic on it. This will also make it so you can use the repl to test code and re-run code much more easily (of course using tools like [Revise](https://timholy.github.io/Revise.jl/stable/) which we'll talk about in time). 

# Gym Interface

Another populare interface is that provided by [OpenAI Gym](https://gym.openai.com). While I don't use this interface, it is still very possible to define and use within Julia. My preference is not any indication on the usefullness of the interface, but is an indication of me learning RL using the RL Glue interface. The Gym interface seems to have been widely adopted in the Python RL community, and many environment packages follow this API.

While not provided in this tutorial, it should be clear how to implement the Gym API. It is also possible to extend any environment to either interface thanks to multiple dispatch!
"""

# ╔═╡ Cell order:
# ╟─9d216976-c95f-11eb-0501-37164fe97a8f
# ╠═2707871e-db71-4a6f-8356-6454360729d1
# ╟─5817820a-cf09-4c18-a79a-2033ff9b141a
# ╠═fb537595-2405-4f93-ab9e-0d14d6421c33
# ╟─5bc4a293-57c6-4ef0-98e9-31c0f171e980
# ╠═0e854488-9048-43d6-975a-cfe8248c0bfe
# ╟─4da83595-4117-47e9-857e-c60f4345d029
# ╠═9ae9934a-2f4e-4776-9f14-dc5082e3828f
# ╟─376c555d-ca83-4695-89e9-d1afdc30c7c2
# ╠═130c152e-ff42-4722-b06f-5007245fedea
# ╠═06ea1f1c-5272-4c16-bdb4-7013dec5994a
# ╠═8cfec52a-747c-4209-8b7e-c1d81905a602
# ╠═25860151-bb59-4997-9c1a-966251bbc597
# ╠═0273ea0e-5139-4a86-9e21-dd9473d04380
# ╟─5eaadfc0-dcdc-45df-b05d-09aae82b2cb7
# ╠═3c50149f-1cb5-4ddb-afaf-a5ca3f1f68e2
# ╠═c51b634d-c66d-45f9-a39f-f8f1dac11e37
# ╠═e0c961b6-c6ad-4cde-acde-3c7d4a6d5ad2
# ╟─5867177d-305e-424b-806e-d27b758fd954
# ╠═7f41a327-ffea-41ee-aea5-58dabc53fe6a
# ╟─d7eff463-353c-469d-98f6-607bc68ea3ed
