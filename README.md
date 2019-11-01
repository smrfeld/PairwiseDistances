# Pairwise particle Distances

## Requirements

Python 3 & Numpy.

## Installation and usage

Use `pip`:
```
pip install pairwiseDistances
```
or manually:
```
python setup.py install
```
and in your code:
```
from pairwiseDistances import *
```
See also the [examples](examples) folder.

## Idea

Given a set of `n` particles with positions in `d`-dimensional space denoted by `x_i` for `i=0,1,...,n`.

We want to sample a pair of particles `i,j` where `i =/= j`, where the probability for sampling this pair is given by:
```
p(i,j) ~ exp( - |x_i - x_j|^2 / 2 sigma^2 )
```
where we use `|x|` to denote the `L_2` norm, and `sigma` is some chosen standard deviation.

This problem is easy to write down, but difficult to implement for large numbers of particles since it requires computing `N^2` distances.

A further problem is that we may want to:
 1. Add a particle.
 2. Remove a particle.
 3. Move a particle.

In this case, not all distances are affected - these operations should be of order `N`. However, if we sample the discrete distribution by forming the CDF, we will need to recalculate it, which is expensive. Alternatively, if we use rejection sampling, we must have a good candidate (envelope) distribution such that the acceptance ratio is high.

This library attempts to come up with the most efficient way to perform these operations in Python.

A key way this library reduces computational cost is by introducing a cutoff for particle distances, where pairs of particles separated by a distance greater than the cutoff are not considered for sampling. It is natural to let this be some chosen multiple of the std. dev., i.e. `m*sigma` for some `m`. If we use rejection sampling where candidates are drawn from a uniform distribution, the acceptance ratio should be approximately `( sqrt(2 * pi) * sigma ) / ( 2 * m * sigma ) = 1.253 / m`. (in the first equation: the area of the Gaussian is `1`, divided by the area of the uniform distribution of width `2 * m * sigma` and height `1 / (sqrt(2 * pi) * sigma )`).

In general, we avoid all use of for loops, and rely extensively on array operations using numpy.

### Multiple species

Multiple species are also supported, where we have multiple species but want to draw two particles of the **same species** (two particles of any species can be done by simply ignoring the species labels).

Specifically, the classes `ProbCalculatorMultiSpecies` and `SamplerMultiSpecies` implement this.

## Examples

See the [examples](examples) folder.
