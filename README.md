# RL in Julia - A tutorial

This tutorial is aimed at getting reinforcement learning researchers up and running in using Julia for their daily driver.
The aim of these tutorials is to give a brief overview of Julia, and start discussing various design patterns which I've
found useful in my research. While this may seem straightforward if coming from a language like Python, I recommend giving
it a full read. While Julia and Python have very similar syntaxes, they differ wildly in the core concepts of the language
leading to major shifts in designs (many of which I find more appealing than what is offered by OO languages). While this 
is pitched towards RL researchers, ML researchers would also find a lot of use as there is considerable overlap.

A brief overview:
- Basic Julia syntax and core concepts
- Introduction to an RL interface
- Getting down and dirty with flux
- Project structure and organization
- Running experiments and saving data
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
(IJulia's Docs)[https://github.com/JuliaLang/IJulia.jl].

This has installed IJulia into your base environment. This will now be accessible to all your projects, which is a neat feature allowing julia projects to be 
isolated from your workflow packages. This means you no longer have to enforce your workflow on others (wonderful)!

Next it is time to instantiate the project environment for the jupyter notebooks.

```julia
(@v1.4)> activate .
(RL-In-Julia)> instantiate
```

This installs all the packages listed in the `Project.toml` and their dependencies (following the Compat rules listed).

Now we are ready to startup the jupyter instance and play with the notebooks. I personally like using jupyter Lab, but you can also use a notebook or nteract. 
To start a jupyter instance in julia (with the RL-In-Julia project activated).
```julia
julia> using IJulia
julia> jupyterlab(;dir=".")
```

# License

All code here is licensed under the [MIT License](https://opensource.org/licenses/MIT) and 
all other content is licensed under a 
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a>
