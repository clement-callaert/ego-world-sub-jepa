"""Ego-World JEPA: factorized latent world models for planning."""

from .encoders import EgoMLP, WorldViT
from .model import EgoWorldConfig, EgoWorldJEPA
from .mpc_policy import LatentMPCPolicy
from .planning import (
    CEMPlanner,
    HermiteMPPIPlanner,
    MPPIPlanner,
    build_planner,
    clamp_node_velocity,
    hermite_basis,
    hermite_interpolate,
)
from .predictor import Predictor
from .probing import linear_probe
from .sigreg import cov_decorrelation_loss, latent_diagnostics, sigreg
from .utils import Normalizer

__all__ = [
    "EgoMLP",
    "WorldViT",
    "Predictor",
    "EgoWorldConfig",
    "EgoWorldJEPA",
    "sigreg",
    "cov_decorrelation_loss",
    "latent_diagnostics",
    "linear_probe",
    "CEMPlanner",
    "MPPIPlanner",
    "HermiteMPPIPlanner",
    "build_planner",
    "hermite_basis",
    "hermite_interpolate",
    "clamp_node_velocity",
    "LatentMPCPolicy",
    "Normalizer",
]

__version__ = "0.1.0"
