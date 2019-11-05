import numpy as np
from collections import Counter

class PairDistCalculatorDifferentSpecies:
    """Calculates pairwise distances for two sets of particles of different species, where only cross-species distances are computed. If the first list of particles is species A, and the second is species B, this means distances between A-B but not A-A or B-B. This corresponds to nA*nB distances, rather than (nA+nB) choose 2 = (nA+nB)*(nA+nB-1)/2 distances.

    Parameters
    ----------
    posns_species_A : np.array([[float]])
        Particle positions of the first species. First axis are particles, second are coordinates in n-dimensional space.
    posns_species_B : np.array([[float]])
        Particle positions of the first species. First axis are particles, second are coordinates in n-dimensional space.
    dim : int
        Dimensionality of each point >= 1. This needs to be specified because you can pass: posns = [].
    cutoff_dist : float
        Optional cutoff distance (the default is None).
    track_labels : bool
        Whether to track labels or not (the default is False).
    labels_species_A : np.array([?])
        If track_labels: the labels of the first species which can in principle be any type which we can search. They must be unique! (the default is np.array([])).
    labels_species_B : np.array([?])
        If track_labels: the labels of the second species which can in principle be any type which we can search. They must be unique! (the default is np.array([])).
    calculate_track_centers : bool
        Whether to calculate and track centers (x_i + x_j)/2 for each pair of particles in addition to the distances (the default is False).
    """

    def __init__(self, posns_species_A, posns_species_B, dim, cutoff_dist=None, track_labels=False, labels_species_A=np.array([]), labels_species_B=np.array([]), calculate_track_centers=False):

        # vars
        self._dim = dim
        self._posns_species_A = np.copy(posns_species_A)
        self._posns_species_B = np.copy(posns_species_B)
        self._n_species_A = len(self._posns_species_A)
        self._n_species_B = len(self._posns_species_B)
        self._cutoff_dist = cutoff_dist
        self._calculate_track_centers = calculate_track_centers

        self._track_labels = track_labels
        if self._track_labels:
            # Check length of labels to match positions
            if len(labels_species_A) != len(self._posns_species_A):
                raise ValueError("The length of the specified labels must match the length of the posns array for the first species, i.e. one label per particle.")
            if len(labels_species_B) != len(self._posns_species_B):
                raise ValueError("The length of the specified labels must match the length of the posns array for the second species, i.e. one label per particle.")

            # Check duplicates
            duplicates = [item for item, count in Counter(labels_species_A).items() if count > 1]
            if len(duplicates) != 0:
                raise ValueError("The labels array for species A contains duplicates. This is not allowed; particle labels must be unique.")
            duplicates = [item for item, count in Counter(labels_species_B).items() if count > 1]
            if len(duplicates) != 0:
                raise ValueError("The labels array for species B contains duplicates. This is not allowed; particle labels must be unique.")

            # Set
            self._labels_species_A = np.copy(labels_species_A)
            self._labels_species_B = np.copy(labels_species_B)

        # Initialize all manner of other properties for possible later use
        self._reset()

        # Compute probs
        self.compute_dists()



    # Various getters
    @property
    def are_dists_valid(self):
        """Get whether the pairwise distances are currently valid.

        Returns
        ---
        bool
            True if they are valid, otherwise False.
        """
        return self._are_dists_valid

    @property
    def posns_species_A(self):
        """Get the positions of the first species.

        Returns
        -------
        np.array([[float]])
            The particle positions of the first species.

        """
        return self._posns_species_A

    @property
    def posns_species_B(self):
        """Get the positions of the second species.

        Returns
        -------
        np.array([[float]])
            The particle positions of the second species.

        """
        return self._posns_species_B

    @property
    def dim(self):
        """Get the dimensionality of the particles.

        Returns
        -------
        int
            Dimensionality >= 1.

        """
        return self._dim

    @property
    def n_species_A(self):
        """Get the number of particles of the first species.

        Returns
        -------
        int
            The number of particles of the first species.

        """
        return self._n_species_A

    @property
    def n_species_B(self):
        """Get the number of particles of the second species.

        Returns
        -------
        int
            The number of particles of the second species.

        """
        return self._n_species_B

    @property
    def track_labels(self):
        """Get the flag whehter we are tracking labels for each particle.

        Returns
        -------
        bool
            Whether we are tracking labels for each particle.

        """
        return self._track_labels

    @property
    def labels_species_A(self):
        """Get the labels of the first species.

        Returns
        -------
        np.array([?])
            The labels for each particle of the first species, of length n.

        """
        return self._labels_species_A

    @property
    def labels_species_B(self):
        """Get the labels of the second species.

        Returns
        -------
        np.array([?])
            The labels for each particle of the second species, of length n.

        """
        return self._labels_species_B

    @property
    def dists_squared(self):
        """Get the distances squared between particles of the two different species.

        Returns
        -------
        np.array([float])
            The square distances between particles, of length n_first_species*n_second_species.

        """
        if self._are_dists_valid:
            return self._dists_squared
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def idxs_first_particle_of_species_A(self):
        """Get the indexes of the first particle in dists_squared. This is always of species A.

        Returns
        -------
        np.array([int])
            The indexes of the first particle in dists_squared, of length (n choose 2).

        """
        if self._are_dists_valid:
            return self._idxs_first_particle_of_species_A
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def idxs_second_particle_of_species_B(self):
        """Get the indexes of the second particle in dists_squared. This is always of species B.

        Returns
        -------
        np.array([int])
            The indexes of the second particle in dists_squared, of length (n choose 2).

        """
        if self._are_dists_valid:
            return self._idxs_second_particle_of_species_B
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def no_pairs(self):
        """Get the number of distances in dists_squared.

        Returns
        -------
        int
            Equivalent to (n choose 2).

        """
        if self._are_dists_valid:
            return self._no_pairs
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def centers(self):
        """Get the centers between ALL particles, if they were calculated.

        Returns
        -------
        np.array([[float]])
            The centers array of size (n choose 2) x dim if they are calculated, else empty array.

        """
        if self._are_dists_valid:
            return self._centers
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def cutoff_dist(self):
        """Get the cutoff distance.

        Returns
        -------
        float or None
            The cutoff distance, else None.

        """
        return self._cutoff_dist

    @property
    def idxs_first_particle_of_species_A_within_cutoff(self):
        """Get the indexes of the first particle for pairs of particles that are within the cutoff distance. This is always of species A.

        Returns
        -------
        np.array([int])
            The indexes of the first particle, of length <= (n choose 2).

        """
        if self._are_dists_valid:
            return self._idxs_first_particle_of_species_A_within_cutoff
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def idxs_second_particle_of_species_B_within_cutoff(self):
        """Get the indexes of the second particle for pairs of particles that are within the cutoff distance. This is always of species B.

        Returns
        -------
        np.array([int])
            The indexes of the second particle, of length <= (n choose 2).

        """
        if self._are_dists_valid:
            return self._idxs_second_particle_of_species_B_within_cutoff
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def no_pairs_within_cutoff(self):
        """Get the number of pairs of particles within the cutoff distance.

        Returns
        -------
        int
            The length, which is <= (n choose 2).

        """
        if self._are_dists_valid:
            return self._no_pairs_within_cutoff
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")



    def set_cutoff_dist(self, cutoff_dist, keep_dists_valid=True):
        """Set a new cutoff distance (recalculates all cutoff particles).

        Parameters
        ----------
        cutoff_dist : float
            The new cutoff distance, else None.
        keep_dists_valid : bool
            Whether to keep the pairwise distances valid (the default is True).

        """

        self._cutoff_dist = cutoff_dist

        if keep_dists_valid:
            if self._are_dists_valid:
                # Distances are still valid; recompute probs
                self._cut_off_dists()
            else:
                # Distances are not valid to begin with; recompute
                self.compute_dists() # also runs _cut_off_dists



    def _reset(self):

        self._are_dists_valid = False

        self._dists_squared = np.array([]).astype(float)
        self._centers = np.array([]).astype(float)
        self._idxs_first_particle_of_species_A = np.array([]).astype(int)
        self._idxs_second_particle_of_species_B = np.array([]).astype(int)
        self._no_pairs = 0

        self._idxs_first_particle_of_species_A_within_cutoff = np.array([]).astype(int)
        self._idxs_second_particle_of_species_B_within_cutoff = np.array([]).astype(int)
        self._no_pairs_within_cutoff = 0



    def compute_dists(self):

        # Check there are sufficient particles
        if self._n_species_A == 0 or self._n_species_B == 0:
            self._reset()
            self._are_dists_valid = True
            return

        self._idxs_first_particle_of_species_A = np.array([]).astype(int)
        self._idxs_second_particle_of_species_B = np.array([]).astype(int)
        for idx_species_A in range(0,self._n_species_A):
            self._idxs_first_particle_of_species_A = np.append(self._idxs_first_particle_of_species_A, np.full(self._n_species_B, idx_species_A))
            self._idxs_second_particle_of_species_B = np.append(self._idxs_second_particle_of_species_B, np.arange(0,self._n_species_B))

        # uti[0] is i, and uti[1] is j from the previous example
        dr = self._posns_species_A[self._idxs_first_particle_of_species_A] - self._posns_species_B[self._idxs_second_particle_of_species_B]            # computes differences between particle positions
        self._dists_squared = np.sum(dr*dr, axis=1)    # computes distances squared; D is a 4950 x 1 np array
        self._no_pairs = len(self._dists_squared) # n choose 2

        # Centers
        if self._calculate_track_centers:
            self._centers = 0.5 * (self._posns_species_A[self._idxs_first_particle_of_species_A] + self._posns_species_B[self._idxs_second_particle_of_species_B])

        # Cut off the distances
        self._cut_off_dists()

        # Valid
        self._are_dists_valid = True



    def _cut_off_dists(self):

        # Clip distances at std_dev_clip_mult * sigma
        if self._cutoff_dist != None:
            cutoff_dist_squared = pow(self._cutoff_dist,2)

            # Eliminate beyond max dist
            stacked = np.array([self._idxs_first_particle_of_species_A,self._idxs_second_particle_of_species_B,self._dists_squared]).T
            self._idxs_first_particle_of_species_A_within_cutoff, self._idxs_second_particle_of_species_B_within_cutoff, _ = stacked[stacked[:,2] < cutoff_dist_squared].T
            self._idxs_first_particle_of_species_A_within_cutoff = self._idxs_first_particle_of_species_A_within_cutoff.astype(int)
            self._idxs_second_particle_of_species_B_within_cutoff = self._idxs_second_particle_of_species_B_within_cutoff.astype(int)

        else:

            # Take all idxs
            self._idxs_first_particle_of_species_A_within_cutoff = np.copy(self._idxs_first_particle_of_species_A_within_cutoff)
            self._idxs_second_particle_of_species_B_within_cutoff = np.copy(self._idxs_second_particle_of_species_B_within_cutoff)

        # No idx pairs
        self._no_pairs_within_cutoff = len(self._idxs_first_particle_of_species_A_within_cutoff)



    def add_particle_of_species_A(self, posn, idx=None, label=None, check_labels_unique=False, keep_dists_valid=True):
        """Add a particle of species A, performing O(n) calculation to keep pairwise distances correct if keep_dists_valid==True.

        Parameters
        ----------
        posn : np.array([float])
            The position, of length dim.
        idx : int
            The idx of the particle in the posn list, else None to add to the back, which is the fastest (0 is the slowest since all other indexes must be altered) (the default is None).
        label : ?
            Optional label for the new particle (the default is None).
        check_labels_unique : bool
            Whether to check if labels are unique (the default is None).
        keep_dists_valid : bool
            Whether to keep pairwise dists valid, comprising an O(n) calculation (the default is None).

        """

        if self._track_labels and label == None:
            raise ValueError("In add_particle_of_species_A: no label for the new particle was provided, but all the other particles have labels. This is not allowed!")

        if idx == None:
            idx = self._n_species_A

        # Insert labels
        if self._track_labels:
            # Check unique
            if check_labels_unique:
                idxs = np.argwhere(self._labels_species_A == label)
                if len(idxs) != 0:
                    raise ValueError("The provided label: " + str(label) + " of species A already exists! Labels must be unique.")

            # Insert
            self._labels_species_A = np.insert(self._labels_species_A,idx,label,axis=0)
            if len(self._labels_species_A) == 1 and type(self._labels_species_A[0]) != type(label):
                # Fix type
                self._labels_species_A = self._labels.astype(type(label))

        # Insert position
        self._posns_species_A = np.insert(self._posns_species_A,idx,posn,axis=0)
        if len(self._posns_species_A.shape) == 1:
            self._posns_species_A = np.array([self._posns_species_A])
        self._n_species_A += 1

        if self._n_species_A == 1:
            return # Finished

        # If we are not keeping pairwise dists valid, we are done
        if not keep_dists_valid:
            self._are_dists_valid = False
            return

        # If the dists are not valid to begin, need to recompute
        if not self._are_dists_valid:
            self.compute_dists()
            return

        # Shift idxs such that they do not refer to idx
        shift_1 = np.argwhere(self._idxs_first_particle_of_species_A >= idx).flatten()
        self._idxs_first_particle_of_species_A[shift_1] += 1

        shift_1 = np.argwhere(self._idxs_first_particle_of_species_A_within_cutoff >= idx).flatten()
        self._idxs_first_particle_of_species_A_within_cutoff[shift_1] += 1

        # Idxs of particle pairs to add
        # The new particle will be the last, ie index self._n_species_A-1
        idxs_add_of_species_A = np.full(self._n_species_B, self._n_species_A-1)
        # We must consider all n_species_B other particles
        idxs_add_of_species_B = np.arange(0,self._n_species_B)

        # Add a particle
        self._add_particle(idxs_add_of_species_A, idxs_add_of_species_B)



    def add_particle_of_species_B(self, posn, idx=None, label=None, check_labels_unique=False, keep_dists_valid=True):
        """Add a particle of species B, performing O(n) calculation to keep pairwise distances correct if keep_dists_valid==True.

        Parameters
        ----------
        posn : np.array([float])
            The position, of length dim.
        idx : int
            The idx of the particle in the posn list, else None to add to the back, which is the fastest (0 is the slowest since all other indexes must be altered) (the default is None).
        label : ?
            Optional label for the new particle (the default is None).
        check_labels_unique : bool
            Whether to check if labels are unique (the default is None).
        keep_dists_valid : bool
            Whether to keep pairwise dists valid, comprising an O(n) calculation (the default is None).

        """

        if self._track_labels and label == None:
            raise ValueError("In add_particle_of_species_B: no label for the new particle was provided, but all the other particles have labels. This is not allowed!")

        if idx == None:
            idx = self._n_species_B

        # Insert labels
        if self._track_labels:
            # Check unique
            if check_labels_unique:
                idxs = np.argwhere(self._labels_species_B == label)
                if len(idxs) != 0:
                    raise ValueError("The provided label: " + str(label) + " of species B already exists! Labels must be unique.")

            # Insert
            self._labels_species_B = np.insert(self._labels_species_B,idx,label,axis=0)
            if len(self._labels_species_B) == 1 and type(self._labels_species_B[0]) != type(label):
                # Fix type
                self._labels_species_B = self._labels.astype(type(label))

        # Insert position
        self._posns_species_B = np.insert(self._posns_species_B,idx,posn,axis=0)
        if len(self._posns_species_B.shape) == 1:
            self._posns_species_B = np.array([self._posns_species_B])
        self._n_species_B += 1

        if self._n_species_B == 1:
            return # Finished

        # If we are not keeping pairwise dists valid, we are done
        if not keep_dists_valid:
            self._are_dists_valid = False
            return

        # If the dists are not valid to begin, need to recompute
        if not self._are_dists_valid:
            self.compute_dists()
            return

        # Shift idxs such that they do not refer to idx
        shift_1 = np.argwhere(self._idxs_second_particle_of_species_B >= idx).flatten()
        self._idxs_second_particle_of_species_B[shift_1] += 1

        shift_1 = np.argwhere(self._idxs_second_particle_of_species_B_within_cutoff >= idx).flatten()
        self._idxs_second_particle_of_species_B_within_cutoff[shift_1] += 1

        # Idxs of particle pairs to add
        # The new particle will be the last, ie index self._n_species_B-1
        idxs_add_of_species_B = np.full(self._n_species_A, self._n_species_B-1)
        # We must consider all n_species_B other particles
        idxs_add_of_species_A = np.arange(0,self._n_species_A)

        # Add a particle
        self._add_particle(idxs_add_of_species_A, idxs_add_of_species_B)



    def _add_particle(self, idxs_add_of_species_A, idxs_add_of_species_B):

        # Distances squared
        dr = self._posns_species_A[idxs_add_of_species_A] - self._posns_species_B[idxs_add_of_species_B]
        dists_squared_add = np.sum(dr*dr, axis=1)

        # Centers
        if self._calculate_track_centers:
            centers_add = 0.5 * (self._posns_species_A[idxs_add_of_species_A] + self._posns_species_B[idxs_add_of_species_B])
            if centers_add.shape == (self._dim,):
                self._centers = np.append(self._centers, np.array([centers_add]), axis=0)
            else:
                self._centers = np.append(self._centers, centers_add, axis=0)

        # Append to the dists
        self._idxs_first_particle_of_species_A = np.append(self._idxs_first_particle_of_species_A, idxs_add_of_species_A)
        self._idxs_second_particle_of_species_B = np.append(self._idxs_second_particle_of_species_B, idxs_add_of_species_B)
        self._dists_squared = np.append(self._dists_squared,dists_squared_add)
        self._no_pairs += len(dists_squared_add)

        # Max dist
        if self._cutoff_dist != None:
            cutoff_dist_squared = pow(self._cutoff_dist,2)

            # Filter by max dist
            stacked = np.array([idxs_add_of_species_A, idxs_add_of_species_B, dists_squared_add]).T
            idxs_add_of_species_A, idxs_add_of_species_B, dists_squared_add = stacked[stacked[:,2] < cutoff_dist_squared].T

            # Back to integers
            idxs_add_of_species_A = idxs_add_of_species_A.astype(int)
            idxs_add_of_species_B = idxs_add_of_species_B.astype(int)

        # Append
        self._idxs_first_particle_of_species_A_within_cutoff = np.append(self._idxs_first_particle_of_species_A_within_cutoff,idxs_add_of_species_A)
        self._idxs_second_particle_of_species_B_within_cutoff = np.append(self._idxs_second_particle_of_species_B_within_cutoff,idxs_add_of_species_B)

        # Number of pairs now
        self._no_pairs_within_cutoff += len(idxs_add_of_species_A)



    def get_particle_idx_of_species_A_from_label(self, label):
        """Get the index of a particle of species A from the label.

        Parameters
        ----------
        label : ?
            The label.

        Returns
        -------
        idx
            The index of the particle.

        """
        if not self._track_labels:
            raise ValueError("Attempting to access particle labels, but we are not tracking particle labels! Use idxs instead.")

        idxs = np.argwhere(self._labels_species_A == label)
        if len(idxs) > 1:
            raise ValueError("More than one particle of species A has the label: " + str(label) + ". This should not be allowed.")
        elif len(idxs) == 0:
            raise ValueError("No particles of species A with the label: " + str(label) + " exist.")

        return idxs[0][0]



    def get_particle_idx_of_species_B_from_label(self, label):
        """Get the index of a particle of species B from the label.

        Parameters
        ----------
        label : ?
            The label.

        Returns
        -------
        idx
            The index of the particle.

        """
        if not self._track_labels:
            raise ValueError("Attempting to access particle labels, but we are not tracking particle labels! Use idxs instead.")

        idxs = np.argwhere(self._labels_species_B == label)
        if len(idxs) > 1:
            raise ValueError("More than one particle of species B has the label: " + str(label) + ". This should not be allowed.")
        elif len(idxs) == 0:
            raise ValueError("No particles of species B with the label: " + str(label) + " exist.")

        return idxs[0][0]


    def remove_all_particles(self):
        """Remove all particles.

        """
        self._posns_species_A = np.array([]).astype(float)
        self._posns_species_B = np.array([]).astype(float)
        self._n_species_A = 0
        self._n_species_B = 0
        self._labels_species_A = np.array([])
        self._labels_species_B = np.array([])
        self._reset()
        self._are_dists_valid = True



    def remove_particle_of_species_A(self, idx, keep_dists_valid=True):
        """Remove a particle of species A, performing O(n) calculation to keep pairwise distances correct if keep_dists_valid==True.

        Parameters
        ----------
        idx : int
            The idx of the particle in the posn list.
        keep_dists_valid : bool
            Whether the keep the pairwise distances valid (the default is True).

        """

        if self._track_labels:
            # Delete label
            self._labels_species_A = np.delete(self._labels_species_A,idx,axis=0)

        self._posns_species_A = np.delete(self._posns_species_A,idx,axis=0)
        self._n_species_A -= 1

        if self._n_species_A == 0:
            return # Finished

        # If we are not keeping pairwise distances valid, we are done
        if not keep_dists_valid:
            self._are_dists_valid = False
            return

        # If the distances are not valid to begin, need to recompute
        if not self._are_dists_valid:
            self.compute_dists()
            return

        # Idxs to delete in the pair list
        dists_idxs_delete = np.argwhere(self._idxs_first_particle_of_species_A == idx).flatten()

        cutoff_dists_idxs_delete = np.argwhere(self._idxs_first_particle_of_species_A_within_cutoff == idx).flatten()

        self._remove_particle(dists_idxs_delete, cutoff_dists_idxs_delete)

        # Shift the idxs such that they again include idx
        shift_1 = np.argwhere(self._idxs_first_particle_of_species_A > idx).flatten()
        self._idxs_first_particle_of_species_A[shift_1] -= 1

        shift_1 = np.argwhere(self._idxs_first_particle_of_species_A_within_cutoff > idx).flatten()
        self._idxs_first_particle_of_species_A_within_cutoff[shift_1] -= 1



    def remove_particle_of_species_B(self, idx, keep_dists_valid=True):
        """Remove a particle of species B, performing O(n) calculation to keep pairwise distances correct if keep_dists_valid==True.

        Parameters
        ----------
        idx : int
            The idx of the particle in the posn list.
        keep_dists_valid : bool
            Whether the keep the pairwise distances valid (the default is True).

        """

        if self._track_labels:
            # Delete label
            self._labels_species_B = np.delete(self._labels_species_B,idx,axis=0)

        self._posns_species_B = np.delete(self._posns_species_B,idx,axis=0)
        self._n_species_B -= 1

        if self._n_species_B == 0:
            return # Finished

        # If we are not keeping pairwise distances valid, we are done
        if not keep_dists_valid:
            self._are_dists_valid = False
            return

        # If the distances are not valid to begin, need to recompute
        if not self._are_dists_valid:
            self.compute_dists()
            return

        # Idxs to delete in the pair list
        dists_idxs_delete = np.argwhere(self._idxs_second_particle_of_species_B == idx).flatten()

        cutoff_dists_idxs_delete = np.argwhere(self._idxs_second_particle_of_species_B_within_cutoff == idx).flatten()

        self._remove_particle(dists_idxs_delete, cutoff_dists_idxs_delete)

        # Shift the idxs such that they again include idx
        shift_1 = np.argwhere(self._idxs_second_particle_of_species_B > idx).flatten()
        self._idxs_second_particle_of_species_B[shift_1] -= 1

        shift_1 = np.argwhere(self._idxs_second_particle_of_species_B_within_cutoff > idx).flatten()
        self._idxs_second_particle_of_species_B_within_cutoff[shift_1] -= 1



    def _remove_particle(self, dists_idxs_delete, cutoff_dists_idxs_delete):

        # Remove all probs associated with this
        self._dists_squared = np.delete(self._dists_squared,dists_idxs_delete)
        self._idxs_first_particle_of_species_A = np.delete(self._idxs_first_particle_of_species_A,dists_idxs_delete)
        self._idxs_second_particle_of_species_B = np.delete(self._idxs_second_particle_of_species_B,dists_idxs_delete)
        self._no_pairs -= len(dists_idxs_delete)

        if self._calculate_track_centers:
            self._centers = np.delete(self._centers, dists_idxs_delete, axis=0)

        self._idxs_first_particle_of_species_A_within_cutoff = np.delete(self._idxs_first_particle_of_species_A_within_cutoff,cutoff_dists_idxs_delete)
        self._idxs_second_particle_of_species_B_within_cutoff = np.delete(self._idxs_second_particle_of_species_B_within_cutoff,cutoff_dists_idxs_delete)
        self._no_pairs_within_cutoff -= len(cutoff_dists_idxs_delete)



    def move_particle_of_species_A(self, idx, new_posn, keep_dists_valid=True):
        """Move a particle of species A, performing O(n) calculation to keep pairwise distances correct if keep_dists_valid==True.

        Parameters
        ----------
        idx : int
            The idx of the particle in the posn list.
        new_posn : np.array([float])
            The new position, of length dim.
        keep_dists_valid : bool
            Whether to keep the pairwise distances correct (the default is True).

        """

        if self._track_labels:
            label = self._labels_species_A[idx]

            # Remove and reinsert
            self.remove_particle_of_species_A(idx, keep_dists_valid=keep_dists_valid)
            self.add_particle_of_species_A(new_posn, idx=idx, label=label, check_labels_unique=False, keep_dists_valid=keep_dists_valid)

        else:

            # Remove and reinsert
            self.remove_particle_of_species_A(idx, keep_dists_valid=keep_dists_valid)
            self.add_particle_of_species_A(new_posn, idx=idx, keep_dists_valid=keep_dists_valid)



    def move_particle_of_species_B(self, idx, new_posn, keep_dists_valid=True):
        """Move a particle of species B, performing O(n) calculation to keep pairwise distances correct.

        Parameters
        ----------
        idx : int
            The idx of the particle in the posn list.
        new_posn : np.array([float])
            The new position, of length dim.
        keep_dists_valid : bool
            Whether to keep the pairwise distances correct (the default is True).

        """

        if self._track_labels:
            label = self._labels_species_B[idx]

            # Remove and reinsert
            self.remove_particle_of_species_B(idx, keep_dists_valid=keep_dists_valid)
            self.add_particle_of_species_B(new_posn, idx=idx, label=label, check_labels_unique=False, keep_dists_valid=keep_dists_valid)

        else:

            # Remove and reinsert
            self.remove_particle_of_species_B(idx, keep_dists_valid=keep_dists_valid)
            self.add_particle_of_species_B(new_posn, idx=idx, keep_dists_valid=keep_dists_valid)



    def get_particle_idxs_of_species_B_within_cutoff_dist_to_particle_of_species_A_with_idx(self, idx):
        """Get list of indexes of particles of species B that are within the cutoff distance to a given particle of species A.

        Parameters
        ----------
        idx : int
            The index of the particle.

        Returns
        -------
        np.array([int])
            List of particle indexes.

        """

        if not self._are_dists_valid:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

        return self._idxs_second_particle_of_species_B_within_cutoff[self._idxs_first_particle_of_species_A_within_cutoff == idx]



    def get_particle_idxs_of_species_A_within_cutoff_dist_to_particle_of_species_B_with_idx(self, idx):
        """Get list of indexes of particles of species A that are within the cutoff distance to a given particle of species B.

        Parameters
        ----------
        idx : int
            The index of the particle.

        Returns
        -------
        np.array([int])
            List of particle indexes.

        """

        if not self._are_dists_valid:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

        return self._idxs_first_particle_of_species_A_within_cutoff[self._idxs_second_particle_of_species_B_within_cutoff == idx]
