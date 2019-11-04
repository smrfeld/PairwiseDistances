import numpy as np
import logging
from collections import Counter

class PairDistCalculator:
    """Calculates pairwise distances for a set of particles.

    Parameters
    ----------
    posns : np.array([[float]])
        Particle positions. First axis are particles, second are coordinates in n-dimensional space.
    dim : int
        Dimensionality of each point >= 1. This needs to be specified because you can pass: posns = [].
    cutoff_distance : float
        Optional cutoff distance (the default is None).
    track_labels : bool
        Whether to track labels or not (the default is False).
    labels : np.array([?])
        If track_labels: the labels which can in principle be any type which we can search. They must be unique! (the default is np.array([])).
    calculate_track_centers : bool
        Whether to calculate and track centers (x_i + x_j)/2 for each pair of particles in addition to the distances (the default is False).
    """

    def __init__(self, posns, dim, cutoff_distance=None, track_labels=False, labels=np.array([]), calculate_track_centers=False):

        # Setup the logger
        self._logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        # Level of logging to display
        self._logger.setLevel(logging.ERROR)

        # vars
        self._dim = dim
        self._posns = np.copy(posns)
        self._n = len(self._posns)
        self._cutoff_distance = cutoff_distance
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
        self._compute_distances()



    def set_logging_level(self, level):
        """Sets the logging level.

        Parameters
        ----------
        level : logging.level
            The logging level.

        """
        self._logger.setLevel(level)



    # Various getters
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
        return self._dists_squared

    @property
    def idxs_first_particle(self):
        """Get the indexes of the first particle in dists_squared.

        Returns
        -------
        np.array([int])
            The indexes of the first particle in dists_squared, of length (n choose 2).

        """
        return self._idxs_first_particle

    @property
    def idxs_second_particle(self):
        """Get the indexes of the second particle in dists_squared.

        Returns
        -------
        np.array([int])
            The indexes of the second particle in dists_squared, of length (n choose 2).

        """
        return self._idxs_second_particle

    @property
    def no_pairs(self):
        """Get the number of distances in dists_squared.

        Returns
        -------
        int
            Equivalent to (n choose 2).

        """
        return self._no_pairs

    @property
    def centers(self):
        """Get the centers between ALL particles, if they were calculated.

        Returns
        -------
        np.array([[float]])
            The centers array of size (n choose 2) x dim if they are calculated, else empty array.

        """
        return self._centers

    @property
    def cutoff_distance(self):
        """Get the cutoff distance.

        Returns
        -------
        float or None
            The cutoff distance, else None.

        """
        return self._cutoff_distance

    @property
    def idxs_first_particle_within_cutoff(self):
        """Get the indexes of the first particle for pairs of particles that are within the cutoff distance.

        Returns
        -------
        np.array([int])
            The indexes of the first particle, of length <= (n choose 2).

        """
        return self._idxs_first_particle_within_cutoff

    @property
    def idxs_second_particle_within_cutoff(self):
        """Get the indexes of the second particle for pairs of particles that are within the cutoff distance.

        Returns
        -------
        np.array([int])
            The indexes of the second particle, of length <= (n choose 2).

        """
        return self._idxs_second_particle_within_cutoff

    @property
    def no_pairs_within_cutoff(self):
        """Get the number of pairs of particles within the cutoff distance.

        Returns
        -------
        int
            The length, which is <= (n choose 2).

        """
        return self._no_pairs_within_cutoff



    def set_cutoff_distance(self, cutoff_distance):
        """Set a new cutoff distance (recalculates all cutoff particles).

        Parameters
        ----------
        cutoff_distance : float
            The new cutoff distance, else None.

        """

        self._cutoff_distance = cutoff_distance

        # Distances are still valid; recompute probs
        self._cut_off_distances()



    def _reset(self):

        self._dists_squared = np.array([]).astype(float)
        self._centers = np.array([]).astype(float)
        self._idxs_first_particle = np.array([]).astype(int)
        self._idxs_second_particle = np.array([]).astype(int)
        self._no_pairs = 0

        self._idxs_first_particle_within_cutoff = np.array([]).astype(int)
        self._idxs_second_particle_within_cutoff = np.array([]).astype(int)
        self._no_pairs_within_cutoff = 0



    def _compute_distances(self):

        # Check there are sufficient particles
        if self._n < 2:
            self._reset()
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
        self._cut_off_distances()



    def _cut_off_distances(self):

        # Clip distances at std_dev_clip_mult * sigma
        if self._cutoff_distance != None:
            cutoff_distance_squared = pow(self._cutoff_distance,2)

            # Eliminate beyond max dist
            stacked = np.array([self._idxs_first_particle,self._idxs_second_particle,self._dists_squared]).T
            self._idxs_first_particle_within_cutoff, self._idxs_second_particle_within_cutoff, _ = stacked[stacked[:,2] < cutoff_distance_squared].T
            self._idxs_first_particle_within_cutoff = self._idxs_first_particle_within_cutoff.astype(int)
            self._idxs_second_particle_within_cutoff = self._idxs_second_particle_within_cutoff.astype(int)

        else:

            # Take all idxs
            self._idxs_first_particle_within_cutoff = np.copy(self._idxs_first_particle)
            self._idxs_second_particle_within_cutoff = np.copy(self._idxs_second_particle)

        # No idx pairs
        self._no_pairs_within_cutoff = len(self._idxs_first_particle_within_cutoff)



    def add_particle(self, idx, posn, label=None, check_labels_unique=False):
        """Add a particle, performing O(n) calculation to keep pairwise distances correct.

        Parameters
        ----------
        idx : int
            The idx of the particle in the posn list.
        posn : np.array([float])
            The position, of length dim.
        label : ?
            Optional label for the new particle (the default is None).
        check_labels_unique : bool
            Whether to check if labels are unique (the default is None).

        """

        if self._track_labels and label == None:
            raise ValueError("In add_particle: no label for the new particle was provided, but all the other particles have labels. This is not allowed!")

        # Insert labels
        if self._track_labels:
            # Check unique
            if check_labels_unique:
                idxs = np.argwhere(self._labels == label)
                if len(idxs) != 0:
                    raise ValueError("The provided label: " + str(label) + " already exists! Labels must be unique.")

            # Insert
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

        # Shift idxs such that they do not refer to idx
        dists_idxs_all = np.arange(self._no_pairs)
        shift_1 = dists_idxs_all[self._idxs_first_particle >= idx]
        self._idxs_first_particle[shift_1] += 1
        shift_2 = dists_idxs_all[self._idxs_second_particle >= idx]
        self._idxs_second_particle[shift_2] += 1

        cutoff_dists_idxs_all = np.arange(self._no_pairs_within_cutoff)
        shift_1 = cutoff_dists_idxs_all[self._idxs_first_particle_within_cutoff >= idx]
        self._idxs_first_particle_within_cutoff[shift_1] += 1
        shift_2 = cutoff_dists_idxs_all[self._idxs_second_particle_within_cutoff >= idx]
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
                self._centers = np.append(self._centers, np.array([centers_add]), axis=0)
            else:
                self._centers = np.append(self._centers, centers_add, axis=0)

        # Append to the dists
        self._idxs_first_particle = np.append(self._idxs_first_particle,idxs_add_1)
        self._idxs_second_particle = np.append(self._idxs_second_particle,idxs_add_2)
        self._dists_squared = np.append(self._dists_squared,dists_squared_add)
        self._no_pairs += len(dists_squared_add)

        # Max dist
        if self._cutoff_distance != None:
            cutoff_distance_squared = pow(self._cutoff_distance,2)

            # Filter by max dist
            stacked = np.array([idxs_add_1,idxs_add_2,dists_squared_add]).T
            idxs_add_1, idxs_add_2, dists_squared_add = stacked[stacked[:,2] < cutoff_distance_squared].T

            # Back to integers
            idxs_add_1 = idxs_add_1.astype(int)
            idxs_add_2 = idxs_add_2.astype(int)

        # Append
        self._idxs_first_particle_within_cutoff = np.append(self._idxs_first_particle_within_cutoff,idxs_add_1)
        self._idxs_second_particle_within_cutoff = np.append(self._idxs_second_particle_within_cutoff,idxs_add_2)

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

        idxs = np.argwhere(self._labels == label)
        if len(idxs) > 1:
            raise ValueError("More than one particle has the label: " + str(label) + ". This should not be allowed.")
        elif len(idxs) == 0:
            raise ValueError("No particles with the label: " + str(label) + " exist.")

        return idxs[0][0]



    def remove_particle(self, idx):
        """Remove a particle, performing O(n) calculation to keep pairwise distances correct.

        Parameters
        ----------
        idx : int
            The idx of the particle in the posn list.

        """

        if self._track_labels:
            # Delete label
            self._labels = np.delete(self._labels,idx,axis=0)

        self._posns = np.delete(self._posns,idx,axis=0)
        self._n -= 1

        if self._n == 0:
            return # Finished

        # Idxs to delete in the pair list
        dists_idxs_all = np.arange(self._no_pairs)
        dists_idxs_delete_1 = dists_idxs_all[self._idxs_first_particle == idx]
        dists_idxs_delete_2 = dists_idxs_all[self._idxs_second_particle == idx]
        dists_idxs_delete = np.append(dists_idxs_delete_1,dists_idxs_delete_2)

        cutoff_dists_idxs_all = np.arange(self._no_pairs_within_cutoff)
        cutoff_dists_idxs_delete_1 = cutoff_dists_idxs_all[self._idxs_first_particle_within_cutoff == idx]
        cutoff_dists_idxs_delete_2 = cutoff_dists_idxs_all[self._idxs_second_particle_within_cutoff == idx]
        cutoff_dists_idxs_delete = np.append(cutoff_dists_idxs_delete_1,cutoff_dists_idxs_delete_2)

        # Remove all probs associated with this
        self._dists_squared = np.delete(self._dists_squared,dists_idxs_delete)
        self._idxs_first_particle = np.delete(self._idxs_first_particle,dists_idxs_delete)
        self._idxs_second_particle = np.delete(self._idxs_second_particle,dists_idxs_delete)
        self._no_pairs -= len(dists_idxs_delete)

        if self._calculate_track_centers:
            self._centers = np.delete(self._centers, dists_idxs_delete, axis=0)

        self._idxs_first_particle_within_cutoff = np.delete(self._idxs_first_particle_within_cutoff,cutoff_dists_idxs_delete)
        self._idxs_second_particle_within_cutoff = np.delete(self._idxs_second_particle_within_cutoff,cutoff_dists_idxs_delete)
        self._no_pairs_within_cutoff -= len(cutoff_dists_idxs_delete)

        # Shift the idxs such that they again include idx
        dists_idxs_all = np.arange(self._no_pairs)
        shift_1 = dists_idxs_all[self._idxs_first_particle > idx]
        self._idxs_first_particle[shift_1] -= 1
        shift_2 = dists_idxs_all[self._idxs_second_particle > idx]
        self._idxs_second_particle[shift_2] -= 1

        cutoff_dists_idxs_all = np.arange(self._no_pairs_within_cutoff)
        shift_1 = cutoff_dists_idxs_all[self._idxs_first_particle_within_cutoff > idx]
        self._idxs_first_particle_within_cutoff[shift_1] -= 1
        shift_2 = cutoff_dists_idxs_all[self._idxs_second_particle_within_cutoff > idx]
        self._idxs_second_particle_within_cutoff[shift_2] -= 1



    def move_particle(self, idx, new_posn):
        """Move a particle, performing O(n) calculation to keep pairwise distances correct.

        Parameters
        ----------
        idx : int
            The idx of the particle in the posn list.
        new_posn : np.array([float])
            The new position, of length dim.

        """

        if self._track_labels:
            label = self._labels[idx]

            # Remove and reinsert
            self.remove_particle(idx)
            self.add_particle(idx, new_posn, label=label, check_labels_unique=False)

        else:

            # Remove and reinsert
            self.remove_particle(idx)
            self.add_particle(idx, new_posn)



    def get_particle_idxs_within_cutoff_distance_to_particle_with_idx(self, idx):
        """Get list of indexes of particles that are within the cutoff distance to a given particle.

        Parameters
        ----------
        idx : int
            The index of the particle.

        Returns
        -------
        np.array([int])
            List of particle indexes.

        """

        other_particle_idxs_1 = self._idxs_second_particle_within_cutoff[self._idxs_first_particle_within_cutoff == idx]

        other_particle_idxs_2 = self._idxs_first_particle_within_cutoff[self._idxs_second_particle_within_cutoff == idx]

        return np.append(other_particle_idxs_1,other_particle_idxs_2)
