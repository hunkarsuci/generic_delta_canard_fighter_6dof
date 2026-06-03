"""
Physical and numerical constants used by the 6DOF aircraft simulator.
All contants use SI units unless explicitly stated otherwise. 
"""

from __future__ import annotations

GRAVITY_MPS2: float = 9.80665
EPSILON: float = 1.0e-9

SEA_LEVEL_TEMPERATURE_K: float = 288.15
SEA_LEVEL_PRESSURE_PA: float = 101_325.0
SEA_LEVEL_DENSITY_KG_M3: float = 1.225

AIR_GAS_CONSTANT_J_KG_K: float = 287.05287
AIR_GAMMA: float = 1.4