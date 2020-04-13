"""
Kalman filter for tracking people
Uses corners of bounding box to track
"""
import numpy as np
from scipy.linalg import inv, block_diag


class Tracker():

    """
    Kalman filter class. Tracks people with help of detections as measurements

    Attributes:
        box (list): coordinates of a bounding box
        dt (float): elapsed time interval
        F (TYPE): process matix
        H (TYPE): measurement matrix
        hits (int): number of detection matches
        id (int): person ID
        misses (int): number of detection mismatches
        P (TYPE): state covariance matrix
        Q (TYPE): process covariance matrix
        R (TYPE): measurement covariance matrix
        x (list): state
    """

    def __init__(self):
        """
        Initializes Kalman filter
        """
        # Person ID
        self.id = 0

        # Coordinates of a bounding box
        self.box = []

        # Number of detection matches
        self.hits = 0

        # Number of detection mismatches
        self.misses = 0

        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        self.x = []

        # Elapsed time interval
        self.dt = 1.0

        # Process matrix. We use constant velocity model
        self.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, self.dt, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, self.dt, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, self.dt],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        # Measurement matrix. We can only measure the coordinates
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])

        # State covariance matrix
        self.L = 100.0
        # self.P = np.diag(self.L * np.ones(8))
        self.P = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, self.L, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, self.L, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, self.L, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, self.L]])

        # Process covariance matrix
        Q_comp_mat = np.array([[self.dt**4 / 2.0, self.dt**3 / 2.0],
                               [self.dt**3 / 2.0, self.dt**2]])
        self.Q = block_diag(Q_comp_mat, Q_comp_mat,
                            Q_comp_mat, Q_comp_mat)

        # Measurement covariance matrix
        self.R_ratio = 1.0 / 16.0
        R_diag_array = self.R_ratio * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)

    def update_matrices(self, dt):
        """
        Updates matrices according to time elapsed

        Args:
            dt (float): time elapsed
        """
        self.dt = dt
        self.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, self.dt, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, self.dt, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, self.dt],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        Q_comp_mat = np.array([[self.dt**4 / 4.0, self.dt**3 / 2.0],
                               [self.dt**3 / 2.0, self.dt**2]])
        self.Q = block_diag(Q_comp_mat, Q_comp_mat,
                            Q_comp_mat, Q_comp_mat)

    def predict(self):
        """
        Calculates prior x
        """
        self.x = np.dot(self.F, self.x).astype(int)
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q

    def update(self, z):
        """
        Calculates posterior x

        Args:
            z (list): detected bounding box
        """
        S = np.dot(self.H, self.P).dot(self.H.T) + self.R
        K = np.dot(self.P, self.H.T).dot(inv(S))
        y = z - np.dot(self.H, self.x)
        self.x += np.dot(K, y).astype(int)
        self.P = self.P - np.dot(K, self.H).dot(self.P)

    def predict_state(self, dt=1.0):
        """
        Predicts next x without measurement

        Args:
            dt (float, optional): time elapsed
        """
        self.update_matrices(dt)
        self.predict()

    def process_measurement(self, z, dt=1.0):
        """
        Processes measurement and updates x

        Args:
            z (list): detected bounding box
            dt (float, optional): time elapsed
        """
        self.update_matrices(dt)
        self.predict()
        self.update(z)
