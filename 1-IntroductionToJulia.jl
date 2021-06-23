### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ f5de53c4-8a33-40f3-9743-07d48830e3bb
begin
	using LinearAlgebra
	a = rand(2) # get a random column vector of size 10 and set it to the variable a
	b = rand(2)
	s = rand()
	M = rand(2,2)
end;

# ╔═╡ 9b54663f-3886-43aa-a6da-9cf84aeedb4d
using Random # Already in the language, you are just accessing the namespace

# ╔═╡ 0216ddd0-c95a-11eb-202a-b5ac98989c51
md"""
# Using Julia to do modern Reinforcement Learning Research

Looking at the Julia's current surge in popularity in scientific fields and the current amazing work being done in the open source community towards using RL, it makes sense to start considering Julia for doing researching in the machine learning fields. 

This series of tutorials will act as an introduction to using julia to do research in reinforcement learning. We will talk about important details when constructing experiments (for reproducibility) and several facets of Julia which make life much easier for researchers. These tutorials were constructed with jupyter notebooks. You can also find the full documentation at [JuliaLang](julialang.org).

# Why Julia

You may be asking yourself, why should we think about using Julia as Python is ubiquious in the field? This is a great question, and one many people currently using Julia have also thought through. I won't try and convince you here, but here are a few reasons I have for using Julia:

* Mulitple dispatch makes code simpler as compared to Object Orientated Programming.
* Linear algebra accelerated through BLAS is built in to the language. (i.e. arrays of numbers are BLAS by default).
* Plotting code is usually simpler using Plots.jl.

These are only a few reasons, but those I find most beautiful about julia. As we develop our reinforcement learning framework from scratch, I would like you to consider how you would implement these ideas in python. For example, what libraries would you need? What class structure? How would I change behavior? I've found my code significantly simplifies as I try to do things the Julia way (taking full advantage of multiple dispatch).
"""

# ╔═╡ 15217efb-26ec-43e8-b136-f089084bffa4
md"""
## Linear Algebra

Lets start simple, and perform some typical linear algebra operations using Julia to get a handle of the language. While these contain only a subset of what is baked into julia, you should be able to extrapolate to other operations you care about. 

"""

# ╔═╡ 851249e9-3d96-4c6c-98c6-0bc96d29ae3d
md"##### Vector Addtion:"

# ╔═╡ 87a7a91d-553c-4e00-88eb-bfbb7da1a5ba
a + b

# ╔═╡ 505a7663-1e7e-4038-93c7-71c91ef6b2ab
md"##### Scalar multiplication:"

# ╔═╡ 112055b7-4776-4164-892d-685b098f101a
2a, 2*a

# ╔═╡ 0871989c-298d-458a-a220-7f597a1a07f4
md"##### Inner product of vectors:"

# ╔═╡ 2fc9a30b-64bd-4980-910f-d8b8e5083232
a' * b, dot(a, b) # dot is from LinearAlgebra

# ╔═╡ 35b467e3-6bf5-4498-81ab-86376fd15101
md"##### Outer product of vectors:"

# ╔═╡ c4d7efd8-f99e-48f8-9892-38946eb89a17
a * b', a * transpose(b)

# ╔═╡ 8c3e53b8-037c-4f2a-9cc2-e39358216aa2
md"##### Matrix mulitplication of vectors:"

# ╔═╡ 0e5f5e35-53bb-4d17-bd2b-34bf8fab712d
M * a

# ╔═╡ 600ad0bd-0eb6-419a-a19f-d0c701910968
md"##### Matrix element wise product with outer product operation"

# ╔═╡ 6c6bec44-ed8e-499b-b1f4-c8753d07698e
M .* (a * b')

# ╔═╡ 48a4c4b9-e32f-4fbb-9e57-fe456973032c
md"##### Element wise vector multiplication (broadcasting):"

# ╔═╡ 783ae651-f8c4-4eca-81e2-57c6a807f46a
a .* b

# ╔═╡ 1d595759-9a4f-46f1-b030-3ab96df94fb0
md"##### Broadcast scalar-vector addtion"

# ╔═╡ 30d68c48-1683-4da2-98f5-e910038f0a45
a .+ s

# ╔═╡ 7352b072-5933-455c-b757-b8994b4dbe34
md"""
Along with these operations which are always available, there is a Base package `LinearAlgebra`

If you have a specfic linear algebra operation you want but can't find it in Base, you will need to explicitly load the `LinearAlgebra` package. This is already available to you in `Base`.

"""

# ╔═╡ 140391fb-60d1-472a-816f-c9a5800922b8
M_svd = svd(M)

# ╔═╡ a95d12ac-fe5b-4476-a572-4f5a44d6e9c5
all((M_svd.U * diagm(M_svd.S) * M_svd.Vt) .≈ M)

# ╔═╡ 3359dea1-cad7-41eb-b93a-8de616df43a2
md"""
# Random Numbers

Being considerate about your random number generators is one of the most important aspects of making experiments reproducible (i.e. setting your random seed). Julia lets you set the seed of a Global random number generator, as well as construct and manage your own.
"""

# ╔═╡ 61d9d4d6-30db-47c9-a65f-62907991118a
md"""
There is a global random number generator at `Random.GLOBAL_RNG`, which we can seed using

"""

# ╔═╡ 6e5c8e10-e306-401c-a24a-6d48a8babcc4
Random.seed!(10)

# ╔═╡ 2f931d88-c8aa-47bf-9637-9adaf8fabe5a
md"""
We can generate random numbers via:
"""

# ╔═╡ 64f0fb7a-9b7f-4577-840e-8e24eb4edf1f
rand(2), rand(Float32, 2, 2, 2, 2)

# ╔═╡ 40009e47-22af-40e6-bce0-025b5d267dcc
md"""

Note that we can generate specific types through the call, or just use a default type of `Float64`.

This random number generator is thread local, so when a new thread is created and uses the global rng each thread's global rng will be independent (as of `1.5.x` I believe).

We can also use our own managed RNG:

"""

# ╔═╡ 66da8ba9-08e7-4be0-b3de-dc19114a40a4
rng = Random.MersenneTwister(10)

# ╔═╡ f4e8e1c5-f99d-4ef9-ad3b-41f2a0ba653b
rand(rng, 2), rand(rng, Float32, 2, 2, 2, 2)

# ╔═╡ 01fda67b-3104-475d-baa9-39f291f9d58a
md"""
# Multiple Dispatch

Multiple dispatch is the central design ideology of Julia (much like OOP is central to Python or Java). At first glance, it seems very similar to function overloading of other languages (i.e. C++), but it has much more utility because of the ability to dispatch on all argument types (not just one or two)!  This will be useful later, for now I am only going to simple show how you can take advantage.
"""

# ╔═╡ c85958df-25c2-4fef-b22f-49e76b74cd44
f(x) = "default"

# ╔═╡ 857d7599-2246-4720-93db-f30867be4e42
f(x::Integer) = "Int"

# ╔═╡ 0fa37d34-fa05-42e2-97fb-a4e2d17d0a01
f(x::AbstractFloat) = "Float"

# ╔═╡ a930f429-a89c-446b-9c33-f4374bfa3f41
f("Hello")

# ╔═╡ 2f035f65-5a44-4622-93be-bb682edf65a8
f(1)

# ╔═╡ 1fa7997e-c0f9-46ff-be45-ac4bb205e2ae
f(200f0) # 200f0 is a single precision floating point number

# ╔═╡ 25244e40-c3ba-4146-bb89-2ccd85b04540
md"""
A method is Julia's term for a specialized version of a function. Above we wrote a function `f` and hand-made specialized methods for integers and floats. While this may seem like the compiler is only working on the specialized versions, this is incorrect! The compiler will create a specialized method automatically from the generic function, meaning you get the performance of a hand-specialized method. The overriden methods are useful for when there are code changes for different types (which we'll see later on).

If you specialize a function w/o a generic fallback version you will get an exception that there is no matching method.
"""

# ╔═╡ 08fc585c-27ee-4944-b08d-594e5b28e9e3
function greet(s::String)
    "Hello $(s)"
end

# ╔═╡ effcbd70-c7f9-4602-945d-b1088dbc8ac3
greet("Matthew")

# ╔═╡ 1a59f962-ee18-4325-8462-273f20e0d578
greet(1) # This should throw an exception! greet is not defined for integers!

# ╔═╡ 7cc40fa6-6994-4a27-b882-c5ea079b19e2
md"""
Later in the series you will see how to take advantage of multiple dispatch to design an RL interface and use it to make design easier with composition.
"""

# ╔═╡ 4afda8c7-ae64-4e33-a2f6-535e243318d7
md"""
# Types and Data

Now that we have some of the fundamental building blocks of what makes julia tick, we can start thinking about custom types. First, lets just build a basic struct which contains some data we can act on. As a simple example, lets make a struct A which stores an integer (you can imagine this struct being an agent, environment, or really anything), with a simple function.

Note: The struct and functions are in a `begin..end` block. This has to do with how Pluto works, but not a standard pattern in Julia. 

```julia
struct A
   	data::Int
end

function double(a::A)
    a.data * 2
end
```

This just returns double the data stored in A. Lets make another struct B which holds a string this time

```julia
struct B
   data::String
end
```

we can dispatch on `double` by specializing:

```julia
function double(b::B)
    ret = tryparse(Int, b.data)
    if ret == nothing
        0
    else
        2*ret
    end
end
```

This parses the data in b as an Int and doubles. If it is unable to parse (i.e. the data isn’t an Int) it returns 0.

Great!

Now we can use this in a more complex, but general function

```julia
function complicated_function(a_or_b, args...)
    # ... Stuff goes here ...
    data_doubled = double(a_or_b)
    # other stuff
end
```

Notice how I didn’t specialize the a_or_b parameter above and instead kept it generic. This means any struct which specializes double will slot in the correct function when complicated_function is compiled!

Now it should be pretty clear how you can use multiple dispatch to get the kind of generics you are wanting (even though these are contrived examples). We can abstract one more layer and make this even more usable using abstracts:
"""

# ╔═╡ ec3948d9-3b69-4e4f-b95d-f0e1c7fdc447
begin
	abstract type AbstractAB end

	double(aab::AbstractAB) = data_as_int(aab)*2

	my_func(aab::AbstractAB) = 20*(data_as_int(aab)^2 + 10)

	struct A <: AbstractAB
   		data::Int
	end

	data_as_int(a::A) = a.data

	struct B <: AbstractAB
   		data_str::String
	end

	function data_as_int(b::B) 
  		ret = tryparse(Int, b.data_str)
   		if ret == nothing
      		0
   		else
      		ret
   		end
	end
end

# ╔═╡ fe98d911-abbe-4194-bde2-c0589efdd864
my_func(A(12)), my_func(B("12"))

# ╔═╡ bb4a15bb-6add-4ac3-81a7-3c19ee425011
md"""
Notice that we moved the complex and specialized code into more restrictive functions so the general functions can be reused. While here we used actual Abstract typing to make dispatch work the way we want, you can also build this exact same interface using duck typing!

# Design patterns

While I won't go into detail about design patterns which emerge from Julia's multiple dispatch and typing system, you should read [this blog](https://www.stochasticlifestyle.com/type-dispatch-design-post-object-oriented-programming-julia/) by Christopher Rackauckas (who is an active user of the language doing research in applying ML/AI methods to Scientific pursuits). 

# Good practices

- [Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- [Differences from other Languages](https://docs.julialang.org/en/v1/manual/noteworthy-differences/)
- [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)

"""

# ╔═╡ Cell order:
# ╟─0216ddd0-c95a-11eb-202a-b5ac98989c51
# ╟─15217efb-26ec-43e8-b136-f089084bffa4
# ╠═f5de53c4-8a33-40f3-9743-07d48830e3bb
# ╟─851249e9-3d96-4c6c-98c6-0bc96d29ae3d
# ╠═87a7a91d-553c-4e00-88eb-bfbb7da1a5ba
# ╟─505a7663-1e7e-4038-93c7-71c91ef6b2ab
# ╠═112055b7-4776-4164-892d-685b098f101a
# ╟─0871989c-298d-458a-a220-7f597a1a07f4
# ╠═2fc9a30b-64bd-4980-910f-d8b8e5083232
# ╟─35b467e3-6bf5-4498-81ab-86376fd15101
# ╠═c4d7efd8-f99e-48f8-9892-38946eb89a17
# ╟─8c3e53b8-037c-4f2a-9cc2-e39358216aa2
# ╠═0e5f5e35-53bb-4d17-bd2b-34bf8fab712d
# ╟─600ad0bd-0eb6-419a-a19f-d0c701910968
# ╠═6c6bec44-ed8e-499b-b1f4-c8753d07698e
# ╟─48a4c4b9-e32f-4fbb-9e57-fe456973032c
# ╠═783ae651-f8c4-4eca-81e2-57c6a807f46a
# ╟─1d595759-9a4f-46f1-b030-3ab96df94fb0
# ╠═30d68c48-1683-4da2-98f5-e910038f0a45
# ╟─7352b072-5933-455c-b757-b8994b4dbe34
# ╠═140391fb-60d1-472a-816f-c9a5800922b8
# ╠═a95d12ac-fe5b-4476-a572-4f5a44d6e9c5
# ╟─3359dea1-cad7-41eb-b93a-8de616df43a2
# ╠═9b54663f-3886-43aa-a6da-9cf84aeedb4d
# ╟─61d9d4d6-30db-47c9-a65f-62907991118a
# ╠═6e5c8e10-e306-401c-a24a-6d48a8babcc4
# ╟─2f931d88-c8aa-47bf-9637-9adaf8fabe5a
# ╠═64f0fb7a-9b7f-4577-840e-8e24eb4edf1f
# ╟─40009e47-22af-40e6-bce0-025b5d267dcc
# ╠═66da8ba9-08e7-4be0-b3de-dc19114a40a4
# ╠═f4e8e1c5-f99d-4ef9-ad3b-41f2a0ba653b
# ╟─01fda67b-3104-475d-baa9-39f291f9d58a
# ╠═c85958df-25c2-4fef-b22f-49e76b74cd44
# ╠═857d7599-2246-4720-93db-f30867be4e42
# ╠═0fa37d34-fa05-42e2-97fb-a4e2d17d0a01
# ╠═a930f429-a89c-446b-9c33-f4374bfa3f41
# ╠═2f035f65-5a44-4622-93be-bb682edf65a8
# ╠═1fa7997e-c0f9-46ff-be45-ac4bb205e2ae
# ╟─25244e40-c3ba-4146-bb89-2ccd85b04540
# ╟─08fc585c-27ee-4944-b08d-594e5b28e9e3
# ╟─effcbd70-c7f9-4602-945d-b1088dbc8ac3
# ╠═1a59f962-ee18-4325-8462-273f20e0d578
# ╟─7cc40fa6-6994-4a27-b882-c5ea079b19e2
# ╟─4afda8c7-ae64-4e33-a2f6-535e243318d7
# ╠═ec3948d9-3b69-4e4f-b95d-f0e1c7fdc447
# ╠═fe98d911-abbe-4194-bde2-c0589efdd864
# ╟─bb4a15bb-6add-4ac3-81a7-3c19ee425011
