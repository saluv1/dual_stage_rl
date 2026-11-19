"""PS2-RL <-> PX4 offboard bridge nodes.

Single-threaded BLAS for every node, set here because this runs before any node module imports numpy.
Older OpenBLAS builds (e.g. numpy 1.26's) spread even a 256 x 256 matrix-vector product over every
core and keep those threads spinning between calls. At the 50 Hz control rate they never sleep: the
slow-track policy's forward pass then took most of the computer away from Gazebo and PX4, which
delivered IMU samples late (EKF2 vertical-velocity kicks, height sag) and reset the uXRCE-DDS time
sync (2026-11-18). Values already set in the environment are kept.
"""
import os as _os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
