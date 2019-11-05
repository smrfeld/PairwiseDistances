# Add the path to the module
import sys
sys.path.append('../')

from pairwiseDistances import *

import numpy as np
import copy
import uuid

class TestDifferentSpecies:

    def make_particles(self):

        # Make some particles
        self.dim = 3
        self.n_A = 40
        self.n_B = 30
        self.posns_A = np.random.rand(self.n_A,self.dim)
        self.posns_B = np.random.rand(self.n_B,self.dim)

    def make_labels(self):

        # Make labels for all the particles
        self.labels_A = []
        for i in range(0,self.n_A):
            self.labels_A.append(uuid.uuid4())
        self.labels_A = np.array(self.labels_A)

        self.labels_B = []
        for i in range(0,self.n_B):
            self.labels_B.append(uuid.uuid4())
        self.labels_B = np.array(self.labels_B)

    def test_dists_shape(self):

        self.make_particles()

        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim)

        # Check shape
        shape = pdc.dists_squared.shape
        assert shape == (self.n_A * self.n_B,)

    def test_cutoff(self):

        self.make_particles()

        cutoff_dist = 0.1
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist)

        # Check length
        length = len(pdc.idxs_first_particle_of_species_A_within_cutoff)
        assert length < self.n_A * self.n_B

    def test_add_particle(self):

        self.make_particles()

        cutoff_dist = 0.1
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist)

        posn_add = np.random.rand(3)
        idx = 10
        pdc.add_particle_of_species_A(posn_add, idx=idx)

        # Check shape
        # Should be +1
        assert pdc.n_species_A == self.n_A + 1

        # Pairs should be updated too
        shape = pdc.dists_squared.shape
        assert shape == ((self.n_A+1)*self.n_B,)

        posn_add = np.random.rand(3)
        idx = 15
        pdc.add_particle_of_species_B(posn_add, idx=idx)

        # Check shape
        # Should be +1
        assert pdc.n_species_B == self.n_B + 1

        # Pairs should be updated too
        shape = pdc.dists_squared.shape
        assert shape == ((self.n_A+1)*(self.n_B+1),)


    def test_add_particle_with_label(self):

        self.make_particles()
        self.make_labels()

        cutoff_dist = 0.1
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist, track_labels=True, labels_species_A=self.labels_A, labels_species_B=self.labels_B)

        posn_add = np.random.rand(3)
        label = uuid.uuid4()
        idx = 10
        pdc.add_particle_of_species_A(posn_add, idx=idx, label=label, check_labels_unique=False)

        # Check shape
        assert len(pdc.labels_species_A) == self.n_A + 1

        posn_add = np.random.rand(3)
        label = uuid.uuid4()
        idx = 13
        pdc.add_particle_of_species_B(posn_add, idx=idx, label=label, check_labels_unique=False)

        # Check shape
        assert len(pdc.labels_species_B) == self.n_B + 1


    def test_add_particle_with_existing_label(self):

        self.make_particles()
        self.make_labels()

        cutoff_dist = 0.1
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist, track_labels=True, labels_species_A=self.labels_A, labels_species_B=self.labels_B)

        posn_add = np.random.rand(3)
        idx = 10
        try:
            label = self.labels_A[0]
            pdc.add_particle_of_species_A(posn_add, idx=idx, label=label, check_labels_unique=True)

            # Should fail
            assert False

        except:

            # Should pass
            assert True

        try:
            label = self.labels_B[0]
            pdc.add_particle_of_species_B(posn_add, idx=idx, label=label, check_labels_unique=True)

            # Should fail
            assert False

        except:

            # Should pass
            assert True

    def test_get_idx_from_particle_label(self):

        self.make_particles()
        self.make_labels()

        cutoff_dist = 0.1
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist, track_labels=True, labels_species_A=self.labels_A, labels_species_B=self.labels_B)

        idx = 13
        label = self.labels_A[idx]
        idx_test = pdc.get_particle_idx_of_species_A_from_label(label)

        assert idx == idx_test

        idx = 13
        label = self.labels_B[idx]
        idx_test = pdc.get_particle_idx_of_species_B_from_label(label)

        assert idx == idx_test

    def test_remove_particle(self):

        self.make_particles()

        cutoff_dist = 0.1
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist)

        idx = 10
        pdc.remove_particle_of_species_A(idx)

        # Check shape
        # Should be -1
        assert pdc.n_species_A == self.n_A - 1

        # Pairs should be updated too
        shape = pdc.dists_squared.shape
        assert shape == ((self.n_A-1)*self.n_B,)

        idx = 8
        pdc.remove_particle_of_species_B(idx)

        # Check shape
        # Should be -1
        assert pdc.n_species_B == self.n_B - 1

        # Pairs should be updated too
        shape = pdc.dists_squared.shape
        assert shape == ((self.n_A-1)*(self.n_B-1),)

    def test_move_particle(self):

        self.make_particles()

        cutoff_dist = 0.1
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist)

        idx = 10
        posn_new = np.random.rand(3)
        pdc.move_particle_of_species_A(idx,posn_new)

        # Check shape
        # Should be the same!
        assert pdc.n_species_A == self.n_A

        # Pairs should be updated too
        shape = pdc.dists_squared.shape
        assert shape == (self.n_A*self.n_B,)

        idx = 8
        posn_new = np.random.rand(3)
        pdc.move_particle_of_species_A(idx,posn_new)

        # Check shape
        # Should be the same!
        assert pdc.n_species_B == self.n_B

        # Pairs should be updated too
        shape = pdc.dists_squared.shape
        assert shape == (self.n_A*self.n_B,)

    def test_get_idxs_within_cutoff_dist(self):

        self.make_particles()

        cutoff_dist = 0.3
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist)

        # Go through all particles
        count_no_pairs_within_cutoff = 0
        for idx in range(0,pdc.n_species_A):
            idxs_within_cutoff = pdc.get_particle_idxs_of_species_B_within_cutoff_dist_to_particle_of_species_A_with_idx(idx)
            count_no_pairs_within_cutoff += len(idxs_within_cutoff)

        # These should match
        assert pdc.no_pairs_within_cutoff == count_no_pairs_within_cutoff

        # Go through all particles
        count_no_pairs_within_cutoff = 0
        for idx in range(0,pdc.n_species_B):
            idxs_within_cutoff = pdc.get_particle_idxs_of_species_A_within_cutoff_dist_to_particle_of_species_B_with_idx(idx)
            count_no_pairs_within_cutoff += len(idxs_within_cutoff)

        # These should match
        assert pdc.no_pairs_within_cutoff == count_no_pairs_within_cutoff

    def test_centers(self):

        self.make_particles()

        cutoff_dist = 0.3
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist, calculate_track_centers=True)

        # Shape of centers
        assert pdc.centers.shape == (self.n_A*self.n_B, self.dim)

        # Add particle
        posn_add = np.random.rand(3)
        idx = 10
        pdc.add_particle_of_species_A(posn_add, idx=idx)

        # Shape of centers
        assert pdc.centers.shape == ((self.n_A+1)*self.n_B, self.dim)

        # Add particle
        posn_add = np.random.rand(3)
        idx = 10
        pdc.add_particle_of_species_B(posn_add, idx=idx)

        # Shape of centers
        assert pdc.centers.shape == ((self.n_A+1)*(self.n_B+1), self.dim)

        # Remove a particle
        idx = 20
        pdc.remove_particle_of_species_A(idx)

        # Shape of centers
        assert pdc.centers.shape == ((self.n_A)*(self.n_B+1), self.dim)

        # Remove a particle
        idx = 20
        pdc.remove_particle_of_species_B(idx)

        # Shape of centers
        assert pdc.centers.shape == ((self.n_A)*(self.n_B), self.dim)

        # Move a particle
        posn_move = np.random.rand(3)
        idx = 12
        pdc.move_particle_of_species_A(idx, posn_move)

        # Shape of centers
        assert pdc.centers.shape == ((self.n_A)*(self.n_B), self.dim)

        # Move a particle
        posn_move = np.random.rand(3)
        idx = 12
        pdc.move_particle_of_species_B(idx, posn_move)

        # Shape of centers
        assert pdc.centers.shape == ((self.n_A)*(self.n_B), self.dim)

    def test_invalidate_dists(self):

        self.make_particles()

        cutoff_dist = 0.3
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist)

        # Add particle
        posn_add = np.random.rand(3)
        idx = 10
        pdc.add_particle_of_species_A(posn_add, idx=idx, keep_dists_valid=False)

        try:
            tmp = pdc.dists_squared # this should fail
            assert False
        except:
            assert True

        # Recompute dists
        pdc.compute_dists()

        # Remove a particle
        idx = 14
        pdc.remove_particle_of_species_B(idx, keep_dists_valid=False)

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
        pdc.move_particle_of_species_A(idx, posn_move, keep_dists_valid=False)

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
        pdc = PairDistCalculatorDifferentSpecies(self.posns_A, self.posns_B, self.dim, cutoff_dist=cutoff_dist)

        pdc.remove_all_particles()

        assert pdc.n_species_A == 0
        assert pdc.n_species_B == 0
