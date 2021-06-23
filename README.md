# RL in Julia - A tutorial

This tutorial is aimed at getting reinforcement learning researchers up and running in using Julia for their daily driver.
The aim of these tutorials is to give a brief overview of Julia, and start discussing various design patterns which I've
found useful in my research. While this may seem straightforward if coming from a language like Python, I recommend giving
it a full read. While Julia and Python have very similar syntaxes, they differ wildly in the core concepts of the language
leading to major shifts in designs (many of which I find more appealing than what is offered by OO languages). While this 
is pitched towards RL researchers, ML researchers would also find a lot of use as there is considerable overlap.

A brief overview:
- [Basic Julia syntax and core concepts](1-IntroductionToJulia.ipynb)
- [Introduction to an RL interface](2-RLGlueInterface.ipynb)
- [Getting down and dirty with flux](3-Flux.ipynb)
- Running experiments and saving data
- Project structure and organization
- Advanced topics


# Getting Started

To get started using these notebooks you need a Julia installation of either Julia 1.3.x or Julia 1.4.x (either should work).
I also recommend creating aliases or links to the julia executable in your path.
You will also need to clone or download the repository how you choose. Because Julia uses environments we will have to do a bit
of work to setup the environments (similar to Python), and this is a good chance to see why Julia is a good choice for reproducible
science!

First in the directory of the cloned repository start a Julia repl

```bash
./path/to/julia
```

You should see Julia startup and open a prompt. Now because we are using Jupyter we will need to install IJulia to get access to the julia kernel.
Because the package manager for julia is built right into the language this is quite simple (note the square bracket is *not* a typo):

```julia
julia> ]
(@v1.4)> add IJulia
(@v1.4)> build IJulia
```

The square bracket activates the "package manager mode" of the julia repl (there are other modes also available including a shell mode accessible through `;`). 
You can leave these modes through a backspace.
The `add` command adds the package `IJulia` which contains the code needed to install the julia kernel. The build command installs the kernel. If you don't have
jupyter already installed and on your path this will also install jupyter, but if you want to make sure to use an already installed version of jupyter you can consult
[IJulia's Docs](https://github.com/JuliaLang/IJulia.jl).

This has installed IJulia into your base environment. This will now be accessible to all your projects, which is a neat feature allowing julia projects to be 
isolated from your workflow packages. This means you no longer have to enforce your workflow on others (wonderful)!

Next it is time to instantiate the project environment for the jupyter notebooks.

```julia
(@v1.4)> activate .
(RL-In-Julia)> instantiate
```

This installs all the packages listed in the [`Project.toml`](Project.toml) and their dependencies (following the Compat rules listed).

Now we are ready to startup the jupyter instance and play with the notebooks. I personally like using jupyter Lab, but you can also use a notebook or nteract. 
To start a jupyter instance in julia (with the RL-In-Julia project activated).
```julia
julia> using IJulia
julia> jupyterlab(;dir=".")
```

# The Julia Ecosystem for ML and RL research

There is a lot of really useful packages for RL and ML research in Julia. While this tutorial will be using the packages I've built (I'm quite biased in favor of these), that doesn't mean they are the best packages out there for your needs!

One great example of the RL initiative in Julia is the JuliaReinforcementLearning group and their main packages [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl). They have a bunch of models implemented and a nice core framework for implementing and reproducing major results in Julia. Their work in creating bindings to [environments](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl) is also a great contribution to the ecosystem. You may ask, "If these are great packages, why don't you use them in your research". My answer is the same I had when I avoided Python packages like [DeepMind's Dopamine](https://github.com/google/dopamine) or [OpenAI Baselines](https://openai.com/blog/openai-baselines-dqn/). My research often don't conform well to these kinds of frameworks, and it is often easier to get something up and running with a light framework where I have more control over all the parts and more flexibility (thus [MinimalRLCore.jl](https://github.com/mkschleg/MinimalRLCore.jl)).

There are two main packages for ANNs in Julia. The first is [Flux.jl](https://fluxml.ai). This is the premier deep learning package in Julia and used by a large portion of the community and what we will be using in this tutorial. There is also [Knet.jl](https://github.com/denizyuret/Knet.jl), which is a bit more feature rich (with flux fast approaching) but not written entirely in Julia (being more like Pytorch or Tensorflow). What makes Flux special is that it is written entirely in Julia, meaning you have access to every component and very easily write code which is as fast as the Flux base code in pure Julia. This creates an ecosystem where nothing is "first class" in the same sense as in Tensorflow or Pytorch. This has the potential to make development and deployment of new model types much quicker. Of course, being relatively new there are still some hiccups. But the package is very quickly hurtling towards 1.0 status bringing about a nice bit of stability.

Flux also provides a handy list of packages they have found useful: [link](https://fluxml.ai/Flux.jl/stable/ecosystem/). I use several of these packages and have developed [some of my own](https://github.com/mkschleg/Reproduce.jl) around the types of experiments typically found in RL (i.e. parameter sweeps and the like). One thing to keep in mind is Julia's ecosystem is quite different than what you find in Python. Because Julia code can be efficient, you don't have to use Flux or Knet's version of things. This often leads to confusion to new users, as they expect Flux to provide the entire stack for logging and data management. Flux focuses on neural nets and auto-diff, and leaves these other needs to other smaller packages. This means you can often use the work flow that you like rather than being forced into your frameworks (potentially) narrow design.


# License

All code here is licensed under the [MIT License](https://opensource.org/licenses/MIT) and 
all other content is licensed under a 
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a>
