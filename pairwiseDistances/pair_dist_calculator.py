import numpy as np
import logging

class PairDistCalculator:
    """Calculates distances for a set of particles.

    Attributes:
    posns (np.array([[float]])): particle positions. First axis are particles, seconds are coordinates in n-dimensional space
    dim (int): dimensionality of each point. This needs to be specified because you can pass: posns = [].
    n (int): number of particles
    cutoff_distance (float): cutoff distance, else None
    labels (np.array([?])): optional labels to keep track of for each particle, else []. The type can be of your choosing

    dists_idxs_first_particle (np.array([int])): idx of the first particle. Only unique combinations together with idxs 2
    dists_idxs_second_particle (np.array([int])): idx of the second particle. Only unique combinations together with idxs 1
    dists_squared (np.array(float)): distances squared between all particles
    no_dists (int): # distances

    Private attributes:
    _logger (logger): logging
    """


    def __init__(self, posns, dim, cutoff_distance=None, labels=np.array([])):

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
        self._posns = posns
        self._n = len(self._posns)
        self._cutoff_distance = cutoff_distance
        self._labels = labels

        # Initialize all manner of other properties for possible later use
        self._reset()

        # Compute probs
        self._compute_distances()



    def set_logging_level(self, level):
        """Sets the logging level

        Args:
        level (logging): logging level
        """
        self._logger.setLevel(level)



    # Various getters
    @property
    def posns(self):
        return self._posns

    @property
    def dim(self):
        return self._dim

    @property
    def n(self):
        return self._n

    @property
    def labels(self):
        return self._labels

    @property
    def dists_squared(self):
        return self._dists_squared

    @property
    def dists_idxs_first_particle(self):
        return self._dists_idxs_first_particle

    @property
    def dists_idxs_second_particle(self):
        return self._dists_idxs_second_particle

    @property
    def no_dists(self):
        return self._no_dists

    @property
    def cutoff_distance(self):
        return self._cutoff_distance

    @property
    def cutoff_dists_squared(self):
        return self._cutoff_dists_squared

    @property
    def cutoff_dists_idxs_first_particle(self):
        return self._cutoff_dists_idxs_first_particle

    @property
    def cutoff_dists_idxs_second_particle(self):
        return self._cutoff_dists_idxs_second_particle

    @property
    def no_cutoff_dists(self):
        return self._no_cutoff_dists



    def set_cutoff_distance(self, cutoff_distance):
        """Set a new cutoff distance (or None).

        Args:
        cutoff_distance (float): the cutoff distance, else None
        """

        self._cutoff_distance = cutoff_distance

        # Distances are still valid; recompute probs
        self._cut_off_distances()



    def _reset(self):
        """Reset structures
        """
        self._dists_squared = np.array([]).astype(float)
        self._dists_idxs_first_particle = np.array([]).astype(int)
        self._dists_idxs_second_particle = np.array([]).astype(int)
        self._no_dists = 0

        self._cutoff_dists_squared = np.array([]).astype(float)
        self._cutoff_dists_idxs_first_particle = np.array([]).astype(int)
        self._cutoff_dists_idxs_second_particle = np.array([]).astype(int)
        self._no_cutoff_dists = 0



    def _compute_distances(self):
        """Compute normalized probabilities.
        """

        # Check there are sufficient particles
        if self._n < 2:
            self._reset()
            return

        # uti is a list of two (1-D) numpy arrays
        # containing the indices of the upper triangular matrix
        self._dists_idxs_first_particle, self._dists_idxs_second_particle = np.triu_indices(self._n,k=1)        # k=1 eliminates diagonal indices

        # uti[0] is i, and uti[1] is j from the previous example
        dr = self._posns[self._dists_idxs_first_particle] - self._posns[self._dists_idxs_second_particle]            # computes differences between particle positions
        self._dists_squared = np.sum(dr*dr, axis=1)    # computes distances squared; D is a 4950 x 1 np array
        self._no_dists = len(self._dists_squared) # n choose 2

        # Cut off the distances
        self._cut_off_distances()



    def _cut_off_distances(self):
        """Compute normalized probabilities given distances squared
        """

        # Clip distances at std_dev_clip_mult * sigma
        if self._cutoff_distance != None:
            cutoff_distance_squared = pow(self._cutoff_distance,2)

            # Eliminate beyond max dist
            stacked = np.array([self._dists_idxs_first_particle,self._dists_idxs_second_particle,self._dists_squared]).T
            self._cutoff_dists_idxs_first_particle, self._cutoff_dists_idxs_second_particle, self._cutoff_dists_squared = stacked[stacked[:,2] < cutoff_distance_squared].T
            self._cutoff_dists_idxs_first_particle = self._cutoff_dists_idxs_first_particle.astype(int)
            self._cutoff_dists_idxs_second_particle = self._cutoff_dists_idxs_second_particle.astype(int)

        else:

            # Take all idxs
            self._cutoff_dists_squared = np.copy(self._cutoff_dists_squared)
            self._cutoff_dists_idxs_first_particle = np.copy(self._cutoff_dists_idxs_first_particle)
            self._cutoff_dists_idxs_second_particle = np.copy(self._cutoff_dists_idxs_second_particle)

        # No idx pairs
        self._no_cutoff_dists = len(self._cutoff_dists_squared)



    def add_particle(self, idx, posn):
        """Add a particle

        Args:
        idx (int): position at which to insert the particle
        posn (np.array([float])): position in d dimensions
        """

        self._posns = np.insert(self._posns,idx,posn,axis=0)
        if len(self._posns.shape) == 1:
            self._posns = np.array([self._posns])
        self._n += 1

        if self._n == 1:
            return # Finished

        # Shift idxs such that they do not refer to idx
        dists_idxs_all = np.arange(self._no_dists)
        shift_1 = dists_idxs_all[self._dists_idxs_first_particle >= idx]
        self._dists_idxs_first_particle[shift_1] += 1
        shift_2 = dists_idxs_all[self._dists_idxs_second_particle >= idx]
        self._dists_idxs_second_particle[shift_2] += 1

        cutoff_dists_idxs_all = np.arange(self._no_cutoff_dists)
        shift_1 = cutoff_dists_idxs_all[self._cutoff_dists_idxs_first_particle >= idx]
        self._cutoff_dists_idxs_first_particle[shift_1] += 1
        shift_2 = cutoff_dists_idxs_all[self._cutoff_dists_idxs_second_particle >= idx]
        self._cutoff_dists_idxs_second_particle[shift_2] += 1

        # Idxs of particle pairs to add
        idxs_add_1 = np.full(self._n-1,idx)
        idxs_add_2 = np.delete(np.arange(self._n),idx)

        # Distances squared
        dr = self._posns[idxs_add_1] - self._posns[idxs_add_2]
        dists_squared_add = np.sum(dr*dr, axis=1)

        # Append to the dists
        self._dists_idxs_first_particle = np.append(self._dists_idxs_first_particle,idxs_add_1)
        self._dists_idxs_second_particle = np.append(self._dists_idxs_second_particle,idxs_add_2)
        self._dists_squared = np.append(self._dists_squared,dists_squared_add)
        self._no_dists += len(dists_squared_add)

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
        self._cutoff_dists_idxs_first_particle = np.append(self._cutoff_dists_idxs_first_particle,idxs_add_1)
        self._cutoff_dists_idxs_second_particle = np.append(self._cutoff_dists_idxs_second_particle,idxs_add_2)
        self._cutoff_dists_squared = np.append(self._cutoff_dists_squared,dists_squared_add)

        # Number of pairs now
        self._no_cutoff_dists += len(idxs_add_1)



    def remove_particle(self, idx):
        """Remove a particle

        Args:
        idx (int): idx of the particle to remove
        """

        self._posns = np.delete(self._posns,idx,axis=0)
        self._n -= 1

        if self._n == 0:
            return # Finished

        # Idxs to delete in the pair list
        dists_idxs_all = np.arange(self._no_dists)
        dists_idxs_delete_1 = dists_idxs_all[self._dists_idxs_first_particle == idx]
        dists_idxs_delete_2 = dists_idxs_all[self._dists_idxs_second_particle == idx]
        dists_idxs_delete = np.append(dists_idxs_delete_1,dists_idxs_delete_2)

        cutoff_dists_idxs_all = np.arange(self._no_cutoff_dists)
        cutoff_dists_idxs_delete_1 = cutoff_dists_idxs_all[self._cutoff_dists_idxs_first_particle == idx]
        cutoff_dists_idxs_delete_2 = cutoff_dists_idxs_all[self._cutoff_dists_idxs_second_particle == idx]
        cutoff_dists_idxs_delete = np.append(cutoff_dists_idxs_delete_1,cutoff_dists_idxs_delete_2)

        # Remove all probs associated with this
        self._dists_squared = np.delete(self._dists_squared,dists_idxs_delete)
        self._dists_idxs_first_particle = np.delete(self._dists_idxs_first_particle,dists_idxs_delete)
        self._dists_idxs_second_particle = np.delete(self._dists_idxs_second_particle,dists_idxs_delete)
        self._no_dists -= len(dists_idxs_delete)

        self._cutoff_dists_squared = np.delete(self._cutoff_dists_squared,cutoff_dists_idxs_delete)
        self._cutoff_dists_idxs_first_particle = np.delete(self._cutoff_dists_idxs_first_particle,cutoff_dists_idxs_delete)
        self._cutoff_dists_idxs_second_particle = np.delete(self._cutoff_dists_idxs_second_particle,cutoff_dists_idxs_delete)
        self._no_cutoff_dists -= len(cutoff_dists_idxs_delete)

        # Shift the idxs such that they again include idx
        dists_idxs_all = np.arange(self._no_dists)
        shift_1 = dists_idxs_all[self._dists_idxs_first_particle > idx]
        self._dists_idxs_first_particle[shift_1] -= 1
        shift_2 = dists_idxs_all[self._dists_idxs_second_particle > idx]
        self._dists_idxs_second_particle[shift_2] -= 1

        cutoff_dists_idxs_all = np.arange(self._no_cutoff_dists)
        shift_1 = cutoff_dists_idxs_all[self._cutoff_dists_idxs_first_particle > idx]
        self._cutoff_dists_idxs_first_particle[shift_1] -= 1
        shift_2 = cutoff_dists_idxs_all[self._cutoff_dists_idxs_second_particle > idx]
        self._cutoff_dists_idxs_second_particle[shift_2] -= 1



    def move_particle(self, idx, new_posn):
        """Move a particle

        Args:
        idx (int): idx of the particle to move
        new_posn (np.array([float])): new position in d dimensions
        """

        # Remove and reinsert
        self.remove_particle(idx)
        self.add_particle(idx, new_posn)



    def get_particle_idxs_within_cutoff_radius_to_particle_with_label(self, particle_label):

        idxs = self._labels == particle_label
        if len(idxs) == 0:
            raise ValueError("Particle label: %s does not exist!" % str(particle_label))
        elif len(idxs) > 1:
            raise ValueError("Particle label: %s is not unique!" % str(particle_label))
        else:
            return self.get_particle_idxs_within_cutoff_radius_to_particle_with_idx(particle_idx=idxs[0])



    def get_particle_idxs_within_cutoff_radius_to_particle_with_idx(self, particle_idx):

        idxs_1 = self._cutoff_dists_idxs_first_particle == particle_idx
        other_particle_idxs_1 = self._cutoff_dists_idxs_second_particle[idxs_1]

        idxs_2 = self._cutoff_dists_idxs_second_particle == particle_idx
        other_particle_idxs_2 = self._cutoff_dists_idxs_first_particle[idxs_2]

        return np.append(other_particle_idxs_1,other_particle_idxs_2)
