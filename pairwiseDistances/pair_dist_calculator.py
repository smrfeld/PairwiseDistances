import numpy as np
from collections import Counter

class PairDistCalculator:
    """Calculates pairwise distances for a set of particles.

    Parameters
    ----------
    posns : np.array([[float]])
        Particle positions. First axis are particles, second are coordinates in n-dimensional space.
    dim : int
        Dimensionality of each point >= 1. This needs to be specified because you can pass: posns = [].
    cutoff_dist : float
        Optional cutoff distance (the default is None).
    track_labels : bool
        Whether to track labels or not (the default is False).
    labels : np.array([?])
        If track_labels: the labels which can in principle be any type which we can search. They must be unique! (the default is np.array([])).
    calculate_track_centers : bool
        Whether to calculate and track centers (x_i + x_j)/2 for each pair of particles in addition to the distances (the default is False).
    """

    def __init__(self, posns, dim, cutoff_dist=None, track_labels=False, labels=np.array([]), calculate_track_centers=False):

        # vars
        self._dim = dim
        self._posns = np.copy(posns)
        self._n = len(self._posns)
        self._cutoff_dist = cutoff_dist
        self._calculate_track_centers = calculate_track_centers

        self._track_labels = track_labels
        if self._track_labels:
            # Check length of labels to match positions
            if len(labels) != len(self._posns):
                raise ValueError("The length of the specified labels must match the length of the posns array, i.e. one label per particle.")

            # Check duplicates
            duplicates = [item for item, count in Counter(labels).items() if count > 1]
            if len(duplicates) != 0:
                raise ValueError("The labels array contains duplicates. This is not allowed; particle labels must be unique.")

            # Set
            self._labels = np.copy(labels)

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
    def posns(self):
        """Get the positions.

        Returns
        -------
        np.array([[float]])
            The particle positions.

        """
        return self._posns

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
    def n(self):
        """Get the number of particles.

        Returns
        -------
        int
            The number of particles.

        """
        return self._n

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
    def labels(self):
        """Get the labels.

        Returns
        -------
        np.array([?])
            The labels for each particle, of length n.

        """
        return self._labels

    @property
    def dists_squared(self):
        """Get the distances squared between ALL particles.

        Returns
        -------
        np.array([float])
            The square distances between particles, of length (n choose 2).

        """
        if self._are_dists_valid:
            return self._dists_squared
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def idxs_first_particle(self):
        """Get the indexes of the first particle in dists_squared.

        Returns
        -------
        np.array([int])
            The indexes of the first particle in dists_squared, of length (n choose 2).

        """
        if self._are_dists_valid:
            return self._idxs_first_particle
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def idxs_second_particle(self):
        """Get the indexes of the second particle in dists_squared.

        Returns
        -------
        np.array([int])
            The indexes of the second particle in dists_squared, of length (n choose 2).

        """
        if self._are_dists_valid:
            return self._idxs_second_particle
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
            raise ValueError("Pairwise distances (and centers) are currently invalid. Run compute_dists first.")

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
    def cutoff_dists_squared(self):
        """Get the distances squared between particles within the cutoff distance.

        Returns
        -------
        np.array([float])
            The square distances between particles, of length <= (n choose 2).

        """
        if self._are_dists_valid:
            return self._cutoff_dists_squared
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def idxs_first_particle_within_cutoff(self):
        """Get the indexes of the first particle for pairs of particles that are within the cutoff distance.

        Returns
        -------
        np.array([int])
            The indexes of the first particle, of length <= (n choose 2).

        """
        if self._are_dists_valid:
            return self._idxs_first_particle_within_cutoff
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def idxs_second_particle_within_cutoff(self):
        """Get the indexes of the second particle for pairs of particles that are within the cutoff distance.

        Returns
        -------
        np.array([int])
            The indexes of the second particle, of length <= (n choose 2).

        """
        if self._are_dists_valid:
            return self._idxs_second_particle_within_cutoff
        else:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

    @property
    def cutoff_centers(self):
        """Get the centers between all particles within the cutoff radius, if they were calculated.

        Returns
        -------
        np.array([[float]])
            The centers array of size (n choose 2) x dim if they are calculated, else empty array.

        """
        if self._are_dists_valid:
            return self._cutoff_centers
        else:
            raise ValueError("Pairwise distances (and centers) are currently invalid. Run compute_dists first.")

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
        self._idxs_first_particle = np.array([]).astype(int)
        self._idxs_second_particle = np.array([]).astype(int)
        self._no_pairs = 0

        self._cutoff_dists_squared = np.array([]).astype(float)
        self._cutoff_centers = np.array([]).astype(float)
        self._idxs_first_particle_within_cutoff = np.array([]).astype(int)
        self._idxs_second_particle_within_cutoff = np.array([]).astype(int)
        self._no_pairs_within_cutoff = 0



    def compute_dists(self):

        # Check there are sufficient particles
        if self._n < 2:
            self._reset()
            self._are_dists_valid = True
            return

        # uti is a list of two (1-D) numpy arrays
        # containing the indices of the upper triangular matrix
        self._idxs_first_particle, self._idxs_second_particle = np.triu_indices(self._n,k=1)        # k=1 eliminates diagonal indices

        # uti[0] is i, and uti[1] is j from the previous example
        dr = self._posns[self._idxs_first_particle] - self._posns[self._idxs_second_particle]            # computes differences between particle positions
        self._dists_squared = np.sum(dr*dr, axis=1)    # computes distances squared; D is a 4950 x 1 np array
        self._no_pairs = len(self._dists_squared) # n choose 2

        # Centers
        if self._calculate_track_centers:
            self._centers = 0.5 * (self._posns[self._idxs_first_particle] + self._posns[self._idxs_second_particle])

        # Cut off the distances
        self._cut_off_dists()

        # Valid
        self._are_dists_valid = True



    def _cut_off_dists(self):

        # Clip distances at std_dev_clip_mult * sigma
        if self._cutoff_dist != None:
            cutoff_dist_squared = pow(self._cutoff_dist,2)

            # Eliminate beyond max dist
            if self._calculate_track_centers:
                idxs = np.argwhere(self._dists_squared < cutoff_dist_squared).flatten()
                self._idxs_first_particle_within_cutoff = self._idxs_first_particle[idxs]
                self._idxs_second_particle_within_cutoff = self._idxs_second_particle[idxs]
                self._cutoff_dists_squared = self._dists_squared[idxs]
                self._cutoff_centers = self._centers[idxs]
            else:
                stacked = np.array([self._idxs_first_particle,self._idxs_second_particle,self._dists_squared]).T
                self._idxs_first_particle_within_cutoff, self._idxs_second_particle_within_cutoff, self._cutoff_dists_squared = stacked[stacked[:,2] < cutoff_dist_squared].T
                self._idxs_first_particle_within_cutoff = self._idxs_first_particle_within_cutoff.astype(int)
                self._idxs_second_particle_within_cutoff = self._idxs_second_particle_within_cutoff.astype(int)

        else:

            # Take all idxs
            self._idxs_first_particle_within_cutoff = np.copy(self._idxs_first_particle)
            self._idxs_second_particle_within_cutoff = np.copy(self._idxs_second_particle)

        # No idx pairs
        self._no_pairs_within_cutoff = len(self._idxs_first_particle_within_cutoff)



    def add_particle(self, posn, idx=None, label=None, check_labels_unique=False, keep_dists_valid=True):
        """Add a particle, performing O(n) calculation to keep pairwise distances correct if keep_dists_valid==True.

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
            raise ValueError("In add_particle: no label for the new particle was provided, but all the other particles have labels. This is not allowed!")

        # Index
        if idx == None:
            idx = self._n

        # Insert labels
        if self._track_labels:
            # Check unique
            if check_labels_unique:
                idxs = np.argwhere(self._labels == label).flatten()
                if len(idxs) != 0:
                    raise ValueError("The provided label: " + str(label) + " already exists! Labels must be unique.")

            # Insert
            if len(self._labels) == 0:
                self._labels = np.array([]).astype(type(label))
            self._labels = np.insert(self._labels,idx,label,axis=0)
            if len(self._labels) == 1 and type(self._labels[0]) != type(label):
                # Fix type
                self._labels = self._labels.astype(type(label))

        # Insert position
        self._posns = np.insert(self._posns,idx,posn,axis=0)
        if len(self._posns.shape) == 1:
            self._posns = np.array([self._posns])
        self._n += 1

        if self._n == 1:
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
        shift_1 = np.argwhere(self._idxs_first_particle >= idx).flatten()
        self._idxs_first_particle[shift_1] += 1
        shift_2 = np.argwhere(self._idxs_second_particle >= idx).flatten()
        self._idxs_second_particle[shift_2] += 1

        shift_1 = np.argwhere(self._idxs_first_particle_within_cutoff >= idx).flatten()
        self._idxs_first_particle_within_cutoff[shift_1] += 1
        shift_2 = np.argwhere(self._idxs_second_particle_within_cutoff >= idx).flatten()
        self._idxs_second_particle_within_cutoff[shift_2] += 1

        # Idxs of particle pairs to add
        idxs_add_1 = np.full(self._n-1,idx)
        idxs_add_2 = np.delete(np.arange(self._n),idx)

        # Distances squared
        dr = self._posns[idxs_add_1] - self._posns[idxs_add_2]
        dists_squared_add = np.sum(dr*dr, axis=1)

        # Centers
        if self._calculate_track_centers:
            centers_add = 0.5 * (self._posns[idxs_add_1] + self._posns[idxs_add_2])
            if centers_add.shape == (self._dim,):
                centers_add = np.array([centers_add])

            if len(self._centers) == 0:
                self._centers = centers_add
            else:
                self._centers = np.append(self._centers, centers_add, axis=0)

        # Append to the dists
        self._idxs_first_particle = np.append(self._idxs_first_particle,idxs_add_1)
        self._idxs_second_particle = np.append(self._idxs_second_particle,idxs_add_2)
        self._dists_squared = np.append(self._dists_squared,dists_squared_add)
        self._no_pairs += len(dists_squared_add)

        # Max dist
        if self._cutoff_dist != None:
            cutoff_dist_squared = pow(self._cutoff_dist,2)

            # Filter by max dist
            if self._calculate_track_centers:
                idxs = np.argwhere(dists_squared_add < cutoff_dist_squared).flatten()
                idxs_add_1 = idxs_add_1[idxs]
                idxs_add_2 = idxs_add_2[idxs]
                dists_squared_add = dists_squared_add[idxs]
                centers_add = centers_add[idxs]
            else:
                stacked = np.array([idxs_add_1,idxs_add_2,dists_squared_add]).T
                idxs_add_1, idxs_add_2, dists_squared_add = stacked[stacked[:,2] < cutoff_dist_squared].T

            # Back to integers
            idxs_add_1 = idxs_add_1.astype(int)
            idxs_add_2 = idxs_add_2.astype(int)

        # Append
        self._cutoff_dists_squared = np.append(self._cutoff_dists_squared,dists_squared_add)
        self._idxs_first_particle_within_cutoff = np.append(self._idxs_first_particle_within_cutoff,idxs_add_1)
        self._idxs_second_particle_within_cutoff = np.append(self._idxs_second_particle_within_cutoff,idxs_add_2)
        if self._calculate_track_centers:
            if centers_add.shape == (self._dim,):
                centers_add = np.array([centers_add])

            if len(self._cutoff_centers) == 0:
                    self._cutoff_centers = centers_add
            else:
                self._cutoff_centers = np.append(self._cutoff_centers, centers_add, axis=0)

        # Number of pairs now
        self._no_pairs_within_cutoff += len(idxs_add_1)



    def get_particle_idx_from_label(self, label):
        """Get the index of a particle from the label.

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

        idxs = np.argwhere(self._labels == label).flatten()
        if len(idxs) > 1:
            raise ValueError("More than one particle has the label: " + str(label) + ". This should not be allowed.")
        elif len(idxs) == 0:
            raise ValueError("No particles with the label: " + str(label) + " exist.")

        return idxs[0]



    def remove_all_particles(self):
        """Remove all particles.

        """
        self._posns = np.array([]).astype(float)
        self._n = 0
        self._labels = np.array([])
        self._reset()
        self._are_dists_valid = True



    def remove_particle(self, idx, keep_dists_valid=True):
        """Remove a particle, performing O(n) calculation to keep pairwise distances correct if keep_dists_valid==True.

        Parameters
        ----------
        idx : int
            The idx of the particle in the posn list.
        keep_dists_valid : bool
            Whether the keep the pairwise distances valid (the default is True).

        """

        if self._track_labels:
            # Delete label
            self._labels = np.delete(self._labels,idx,axis=0)

        self._posns = np.delete(self._posns,idx,axis=0)
        self._n -= 1

        if self._n == 0:
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
        dists_idxs_delete_1 = np.argwhere(self._idxs_first_particle == idx).flatten()
        dists_idxs_delete_2 = np.argwhere(self._idxs_second_particle == idx).flatten()
        dists_idxs_delete = np.append(dists_idxs_delete_1,dists_idxs_delete_2)

        cutoff_dists_idxs_delete_1 = np.argwhere(self._idxs_first_particle_within_cutoff == idx).flatten()
        cutoff_dists_idxs_delete_2 = np.argwhere(self._idxs_second_particle_within_cutoff == idx).flatten()
        cutoff_dists_idxs_delete = np.append(cutoff_dists_idxs_delete_1,cutoff_dists_idxs_delete_2)

        # Remove all probs associated with this
        self._dists_squared = np.delete(self._dists_squared, dists_idxs_delete)
        self._idxs_first_particle = np.delete(self._idxs_first_particle, dists_idxs_delete)
        self._idxs_second_particle = np.delete(self._idxs_second_particle, dists_idxs_delete)
        self._no_pairs -= len(dists_idxs_delete)

        if self._calculate_track_centers:
            self._centers = np.delete(self._centers, dists_idxs_delete, axis=0)

        self._cutoff_dists_squared = np.delete(self._cutoff_dists_squared, cutoff_dists_idxs_delete)
        self._idxs_first_particle_within_cutoff = np.delete(self._idxs_first_particle_within_cutoff, cutoff_dists_idxs_delete)
        self._idxs_second_particle_within_cutoff = np.delete(self._idxs_second_particle_within_cutoff, cutoff_dists_idxs_delete)
        self._no_pairs_within_cutoff -= len(cutoff_dists_idxs_delete)

        if self._calculate_track_centers:
            self._cutoff_centers = np.delete(self._cutoff_centers, cutoff_dists_idxs_delete, axis=0)

        # Shift the idxs such that they again include idx
        shift_1 = np.argwhere(self._idxs_first_particle > idx).flatten()
        self._idxs_first_particle[shift_1] -= 1
        shift_2 = np.argwhere(self._idxs_second_particle > idx).flatten()
        self._idxs_second_particle[shift_2] -= 1

        shift_1 = np.argwhere(self._idxs_first_particle_within_cutoff > idx).flatten()
        self._idxs_first_particle_within_cutoff[shift_1] -= 1
        shift_2 = np.argwhere(self._idxs_second_particle_within_cutoff > idx).flatten()
        self._idxs_second_particle_within_cutoff[shift_2] -= 1



    def move_particle(self, idx, new_posn, keep_dists_valid=True):
        """Move a particle, performing O(n) calculation to keep pairwise distances correct if keep_dists_valid==True.

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
            label = self._labels[idx]

            # Remove and reinsert
            self.remove_particle(idx, keep_dists_valid=keep_dists_valid)
            self.add_particle(new_posn, idx=idx, label=label, check_labels_unique=False, keep_dists_valid=keep_dists_valid)

        else:

            # Remove and reinsert
            self.remove_particle(idx, keep_dists_valid=keep_dists_valid)
            self.add_particle(new_posn, idx=idx, keep_dists_valid=keep_dists_valid)



    def get_particles_within_cutoff_dist_to_particle_with_idx(self, idx):
        """Get list of indexes of particles that are within the cutoff distance to a given particle.

        Parameters
        ----------
        idx : int
            The index of the particle.

        Returns
        -------
        np.array([int])
            List of particle indexes.
        np.array([float])
            List of distances squared.
        np.array([[float]])
            List of centers if they exist, else empty.
        """

        if not self._are_dists_valid:
            raise ValueError("Pairwise distances are currently invalid. Run compute_dists first.")

        idxs_1 = np.argwhere(self._idxs_first_particle_within_cutoff == idx).flatten()
        other_particle_idxs_1 = self._idxs_second_particle_within_cutoff[idxs_1]
        idxs_2 = np.argwhere(self._idxs_second_particle_within_cutoff == idx).flatten()
        other_particle_idxs_2 = self._idxs_first_particle_within_cutoff[idxs_2]
        other_particle_idxs = np.append(other_particle_idxs_1,other_particle_idxs_2)

        idxs = np.append(idxs_1,idxs_2)

        if self._calculate_track_centers:
            return [other_particle_idxs, self._cutoff_dists_squared[idxs], self._centers[idxs]]
        else:
            return [other_particle_idxs, self._cutoff_dists_squared[idxs], np.array([])]



    def compute_dists_squared_between_particle_and_existing(self, posn, calculate_centers=False):
        """Compute the squared distances between a particle at a given position and all other existing particles.

        Parameters
        ----------
        posn : np.array([float])
            The position, of length dim.
        calculate_centers : bool
            Whether to also compute the centers (the default is False).

        Returns
        -------
        np.array([float])
            The squared distances between this particle and all the other particles. It is of length n and in the same order as posns.
        np.array([int])
            The idxs that are within the cutoff distance. It is of length <= n.
        np.array([[float]])
            The centers if compute_centers==True, else empty array. It is of size n x dim and in the same order as posns.
        """

        if self._n == 0:
            return [np.array([]), np.array([]), np.array([])]

        # Distances squared
        dr = self._posns - posn
        dists_squared = np.sum(dr*dr, axis=1)

        # Max dist
        idxs_within_cutoff = np.arange(0,self._n)
        if self._cutoff_dist != None:
            cutoff_dist_squared = pow(self._cutoff_dist,2)

            # Filter by max dist
            stacked = np.array([idxs_within_cutoff,dists_squared]).T
            idxs_within_cutoff, _ = stacked[stacked[:,1] < cutoff_dist_squared].T
            idxs_within_cutoff = idxs_within_cutoff.astype(int)

        # Centers
        centers = np.array([]).astype(float)
        if calculate_centers:
            centers = 0.5 * (self._posns + posn)

        return [dists_squared, idxs_within_cutoff, centers]
