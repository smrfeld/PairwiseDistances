# Add the path to the module
import sys
sys.path.append('../')

from pairwiseDistances import *

import numpy as np
import copy

class Test:

    def make_particles(self):

        # Make some particles
        self.dim = 3
        self.n = 100
        self.posns = np.random.rand(self.n,self.dim)

    def test_dists_shape(self):

        self.make_particles()

        pdc = PairDistCalculator(self.posns, self.dim)

        # Check shape
        shape = pdc.dists_squared.shape
        n_choose_2 = pdc.n * (pdc.n-1) / 2
        assert shape == (n_choose_2,)

    def test_cutoff(self):

        self.make_particles()

        cutoff_distance = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_distance=cutoff_distance)

        # Check shape
        shape = pdc.cutoff_dists_squared.shape
        n_choose_2 = pdc.n * (pdc.n-1) / 2
        assert shape < (n_choose_2,)

    def test_add_particle(self):

        self.make_particles()

        cutoff_distance = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_distance=cutoff_distance)

        original_shape = copy.copy(self.n)

        posn_add = np.random.rand(3)
        idx = 10
        pdc.add_particle(idx, posn_add)

        # Check shape
        # Should be +1
        assert pdc.n == original_shape + 1

        # Pairs should be updated too
        shape = pdc.dists_squared.shape
        n_choose_2 = pdc.n * (pdc.n-1) / 2
        assert shape == (n_choose_2,)

    def test_remove_particle(self):

        self.make_particles()

        cutoff_distance = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_distance=cutoff_distance)

        original_shape = copy.copy(self.n)

        idx = 10
        pdc.remove_particle(idx)

        # Check shape
        # Should be -1
        assert pdc.n == original_shape - 1

        # Pairs should be updated too
        shape = pdc.dists_squared.shape
        n_choose_2 = pdc.n * (pdc.n-1) / 2
        assert shape == (n_choose_2,)

    def test_move_particle(self):

        self.make_particles()

        cutoff_distance = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_distance=cutoff_distance)

        original_shape = copy.copy(self.n)

        idx = 10
        posn_new = np.random.rand(3)
        pdc.move_particle(idx,posn_new)

        # Check shape
        # Should be the same!
        assert pdc.n == original_shape

        # Pairs should be updated too
        shape = pdc.dists_squared.shape
        n_choose_2 = pdc.n * (pdc.n-1) / 2
        assert shape == (n_choose_2,)

    def test_get_idxs_within_cutoff_distance(self):

        self.make_particles()

        cutoff_distance = 0.3
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_distance=cutoff_distance)

        # Go through all particles
        count_no_cutoff_dists = 0
        for idx in range(0,pdc.n):
            idxs_within_cutoff = pdc.get_particle_idxs_within_cutoff_distance_to_particle_with_idx(idx)
            count_no_cutoff_dists += len(idxs_within_cutoff)

        # These should match, although we double counted => factor 2
        assert 2*pdc.no_cutoff_dists == count_no_cutoff_dists