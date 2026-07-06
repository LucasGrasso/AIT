from enum import Enum


class Readout(str, Enum):
    """How the ODE model maps a trajectory to an output.

    MEANFIELD: expected state over the halting distribution, x̄(T) = ∫ h·x dt
               (uniform average ∫ x dt / T for a fixed-time NODE).
    ENDPOINT:  final state of the trajectory, x(T).
    """

    MEANFIELD = "meanfield"
    ENDPOINT = "endpoint"
