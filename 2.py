from ising_potts_model import IsingPottsModel, ModelParameters

# Create Ising model at low temperature
params = ModelParameters(J=1.0, h=0.0, T=1.0, q=2, L=50)
model = IsingPottsModel(params)

# Run simulation
model.simulate(3000)

# Visualize results
model.plot_lattice("Ising Model - Low Temperature")
model.plot_observables()

print(f"Final magnetization: {model.magnetization():.3f}")
print(f"Final energy per site: {model.total_energy()/(50*50):.3f}")