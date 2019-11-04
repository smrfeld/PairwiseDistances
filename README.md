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

A common problem is to calculate pairwise distances for `n` particles. This occurs in particle simulations (e.g. electrostatics) and many other scientific fields. This simple module aims to perform this calculation and implements several common efficiency strategies:

1. *Storage*: Calculating pairwise distances is `O(n^2)` (actually `n` choose 2). It is inefficient to repeatedly calculate this. In many codes, only one or a few particles are changed at a time. Adding a particle, removing a particle, or moving a particle are all `O(n)` operations. They are implemented here, such that when a particle is added/removed/moved, not all pairwise distances are recalculated.

2. *Cutoff distance*: A cutoff distance (or radius) usually exist beyond which interactions between particles are negligible. This is implemented here as well.

A further improvement used in most applications (but not implemented for you here) is to divided space into some partitions (a.k.a. voxels or bins) that are approximately the size of the cutoff distance, such that far interactions are not considered. This code can be used in this regard as well, where each partition would contain the pairwise interactions with particles in the same and neighboring voxels.

3. *Labels*: Optional labels can be attached to every particle to keep track of them. The method `get_particle_idx_from_label` can be used to retrieve an index from a label by searching a numpy array with `argwhere` (note that this may be slow).

4. *Centers*: Optional centers `(x_i + x_j)/2` can also be computed and tracked for every pair of particles. As before, adding/removing/moving particles are all `O(n)` operations here.

5. *Different species*: All the same jazz as above, but now the input are two lists of particles, and we are only interested in the cross-species (cross-list) distances between particles. In other words, let the first list be the particles of species `A`, and the second list of species `B`. In this application, we are only interested in the distances between `A-B`, and not `A-A` or `B-B`. If they are of length `nA` and `nB`, this is then `nA*nB` distances, rather than `((nA+nB) choose 2) = (nA+nB) * (nA+nB-1) / 2` distances.

6. *Disable keeping track of pairwise distances*: Keeping track of the pairwise distances can be disabled in the add/remove/move particle methods with the `keep_dists_valid` flag. They can then be recomputed with the `compute_dists` method.

## Examples

See the [examples](examples) folder.

## Development

Tests using `pytest` are in the `tests` folder.
