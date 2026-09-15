"""
Status: Energy = Implemented, Carbon = Experimental
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marl.environment.resource import ResourceState

class EnergyModel:
    """Energy and optional carbon emission model."""
    
    def __init__(self, carbon_intensity: float = 0.5):
        # carbon_intensity: kg CO2 per kWh
        self.carbon_intensity = carbon_intensity

    def compute_energy(self, resource: 'ResourceState', execution_duration: float) -> float:
        """Energy in Joules. energy = power_rate(W) * duration(s)"""
        return resource.power_rate * execution_duration

    def compute_carbon(self, energy_joules: float) -> float:
        """Carbon in kg CO2. Convert J to kWh first."""
        energy_kwh = energy_joules / 3_600_000.0
        return energy_kwh * self.carbon_intensity

    def compute_idle_energy(self, resource: 'ResourceState', duration: float, idle_fraction: float = 0.3) -> float:
        """Idle power = idle_fraction * power_rate"""
        return resource.power_rate * idle_fraction * duration
