from ising_potts_model import PhaseTransitionAnalyzer, critical_temperature_ising_2d, ModelParameters
import numpy as np

# Analyze phase transition
params = ModelParameters(J=1.0, h=0.0, q=2, L=30)
analyzer = PhaseTransitionAnalyzer(params)

T_range = np.linspace(1.0, 4.0, 20)
temperatures, magnetizations, energies = analyzer.temperature_sweep(T_range)

analyzer.plot_phase_diagram(temperatures, magnetizations, energies)

print(f"Critical temperature (Onsager): {critical_temperature_ising_2d():.3f}")