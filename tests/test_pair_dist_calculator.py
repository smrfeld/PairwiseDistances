# Add the path to the module
import sys
sys.path.append('../')

from pairwiseDistances import *

import numpy as np
import copy
import uuid

class Test:

    def make_particles(self):

        # Make some particles
        self.dim = 3
        self.n = 100
        self.posns = np.random.rand(self.n,self.dim)

    def make_labels(self):

        # Make labels for all the particles
        self.labels = []
        for i in range(0,self.n):
            self.labels.append(uuid.uuid4())
        self.labels = np.array(self.labels)

    def test_dists_shape(self):

        self.make_particles()

        pdc = PairDistCalculator(self.posns, self.dim)

        # Check shape
        shape = pdc.dists_squared.shape
        n_choose_2 = pdc.n * (pdc.n-1) / 2
        assert shape == (n_choose_2,)

    def test_cutoff(self):

        self.make_particles()

        cutoff_dist = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist)

        # Check length
        length = len(pdc.idxs_first_particle_within_cutoff)
        n_choose_2 = pdc.n * (pdc.n-1) / 2
        assert length < n_choose_2

        # Check dim
        assert len(pdc.cutoff_dists_squared) < n_choose_2

    def test_add_particle(self):

        self.make_particles()

        cutoff_dist = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist)

        original_shape = copy.copy(self.n)

        posn_add = np.random.rand(3)
        idx = 10
        pdc.add_particle(posn_add, idx=idx)

        # Check shape
        # Should be +1
        assert pdc.n == original_shape + 1

        # Pairs should be updated too
        shape = pdc.dists_squared.shape
        n_choose_2 = pdc.n * (pdc.n-1) / 2
        assert shape == (n_choose_2,)


    def test_add_particle_with_label(self):

        self.make_particles()
        self.make_labels()

        cutoff_dist = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist, track_labels=True, labels=self.labels)

        original_shape = copy.copy(self.n)

        posn_add = np.random.rand(3)
        label = uuid.uuid4()
        idx = 10
        pdc.add_particle(posn_add, idx=idx, label=label, check_labels_unique=False)

        # Check shape
        assert len(pdc.labels) == self.n + 1

    def test_add_particle_with_existing_label(self):

        self.make_particles()
        self.make_labels()

        cutoff_dist = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist, track_labels=True, labels=self.labels)

        original_shape = copy.copy(self.n)

        posn_add = np.random.rand(3)
        label = self.labels[0]
        idx = 10
        try:
            pdc.add_particle(posn_add, idx=idx, label=label, check_labels_unique=True)

            # Should fail
            assert False

        except:

            # Should pass
            assert True

    def test_get_idx_from_particle_label(self):

        self.make_particles()
        self.make_labels()

        cutoff_dist = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist, track_labels=True, labels=self.labels)

        idx = 53
        label = self.labels[idx]
        idx_test = pdc.get_particle_idx_from_label(label)

        assert idx == idx_test

    def test_remove_particle(self):

        self.make_particles()

        cutoff_dist = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist)

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

        cutoff_dist = 0.1
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist)

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

    def test_get_idxs_within_cutoff_dist(self):

        self.make_particles()

        cutoff_dist = 0.3
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist)

        # Go through all particles
        count_no_pairs_within_cutoff = 0
        for idx in range(0,pdc.n):
            idxs_within_cutoff = pdc.get_particle_idxs_within_cutoff_dist_to_particle_with_idx(idx)
            count_no_pairs_within_cutoff += len(idxs_within_cutoff)

        # These should match, although we double counted => factor 2
        assert 2*pdc.no_pairs_within_cutoff == count_no_pairs_within_cutoff

    def test_centers(self):

        self.make_particles()

        cutoff_dist = 0.3
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist, calculate_track_centers=True)

        # Shape of centers
        n_choose_2 = self.n * (self.n-1) / 2
        assert pdc.centers.shape == (n_choose_2, self.dim)
        assert len(pdc.cutoff_centers) < n_choose_2

        # Add particle
        posn_add = np.random.rand(3)
        idx = 10
        pdc.add_particle(posn_add, idx=idx)

        # Shape of centers
        n_choose_2_plus_1 = (self.n+1) * (self.n+1-1) / 2
        assert pdc.centers.shape == (n_choose_2_plus_1, self.dim)

        # Remove a particle
        idx = 30
        pdc.remove_particle(idx)

        # Shape of centers
        n_choose_2 = self.n * (self.n-1) / 2
        assert pdc.centers.shape == (n_choose_2, self.dim)

        # Move a particle
        posn_move = np.random.rand(3)
        idx = 12
        pdc.move_particle(idx, posn_move)

        # Shape of centers
        n_choose_2 = self.n * (self.n-1) / 2
        assert pdc.centers.shape == (n_choose_2, self.dim)

    def test_invalidate_dists(self):

        self.make_particles()

        cutoff_dist = 0.3
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist)

        # Add particle
        posn_add = np.random.rand(3)
        idx = 10
        pdc.add_particle(posn_add, idx=idx, keep_dists_valid=False)

        try:
            tmp = pdc.dists_squared # this should fail
            assert False
        except:
            assert True

        # Recompute dists
        pdc.compute_dists()

        # Remove a particle
        idx = 30
        pdc.remove_particle(idx, keep_dists_valid=False)

        try:
            tmp = pdc.dists_squared # this should fail
            assert False
        except:
            assert True

        # Recompute dists
        pdc.compute_dists()

        # Move a particle
        posn_move = np.random.rand(3)
        idx = 12
        pdc.move_particle(idx, posn_move, keep_dists_valid=False)

        try:
            tmp = pdc.dists_squared # this should fail
            assert False
        except:
            assert True

        # Recompute dists
        pdc.compute_dists()

        # Now all should be OK
        try:
            tmp = pdc.dists_squared
            assert True
        except:
            assert False

    def test_remove_all_particles(self):

        self.make_particles()

        cutoff_dist = 0.3
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist)

        pdc.remove_all_particles()

        assert pdc.n == 0

    def test_compute_dists_squared_between_particle_and_existing(self):

        self.make_particles()

        cutoff_dist = 0.3
        pdc = PairDistCalculator(self.posns, self.dim, cutoff_dist=cutoff_dist)

        posn = np.random.rand(3)
        dists_squared, idxs_within_cutoff, centers = pdc.compute_dists_squared_between_particle_and_existing(posn, calculate_centers=True)

        assert len(dists_squared) == pdc.n
        assert len(idxs_within_cutoff) <= pdc.n
        assert centers.shape == (pdc.n, pdc.dim)
