"""spatial_reg_ward package

This package exposes the SpatialRegWard clustering class.
"""
from .spatialregward import SpatialRegWard
from .toy_data import make_half_moon_toy_data

__all__ = ["SpatialRegWard", "make_half_moon_toy_data"]
