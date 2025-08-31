"""
Ising-Potts Model Implementation

This module implements the Ising-Potts model, a discrete statistical model on a lattice
for describing interacting particles with a finite number of possible states.

The Ising model (q=2) describes spins s_i ∈ {-1, +1} with Hamiltonian:
H = -J Σ⟨i,j⟩ s_i s_j - h Σ_i s_i

The Potts model (q>2) generalizes this to s_i ∈ {1, 2, ..., q} with:
H = -J Σ⟨i,j⟩ δ(s_i, s_j)

where δ(a,b) is the Kronecker delta function.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ModelParameters:
    """Parameters for the Ising-Potts model"""
    J: float = 1.0          # Coupling constant
    h: float = 0.0          # External magnetic field (Ising only)
    T: float = 2.0          # Temperature
    q: int = 2              # Number of states (q=2 for Ising, q>2 for Potts)
    L: int = 50             # Lattice size (L x L)
    
    @property
    def beta(self) -> float:
        """Inverse temperature β = 1/(k_B T), assuming k_B = 1"""
        return 1.0 / self.T if self.T > 0 else float('inf')


class IsingPottsModel:
    """
    Implementation of the Ising-Potts model with Monte Carlo simulation
    """
    
    def __init__(self, params: ModelParameters):
        self.params = params
        self.lattice = self._initialize_lattice()
        self.energy_history = []
        self.magnetization_history = []
        
    def _initialize_lattice(self) -> np.ndarray:
        """Initialize the lattice with random spins/states"""
        if self.params.q == 2:
            # Ising model: spins ∈ {-1, +1}
            return np.random.choice([-1, 1], size=(self.params.L, self.params.L))
        else:
            # Potts model: states ∈ {1, 2, ..., q}
            return np.random.randint(1, self.params.q + 1, size=(self.params.L, self.params.L))
    
    def _get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get nearest neighbors with periodic boundary conditions"""
        L = self.params.L
        neighbors = [
            ((i + 1) % L, j),      # right
            ((i - 1) % L, j),      # left
            (i, (j + 1) % L),      # up
            (i, (j - 1) % L)       # down
        ]
        return neighbors
    
    def _energy_change_ising(self, i: int, j: int, new_spin: int) -> float:
        """Calculate energy change for flipping a spin in Ising model"""
        old_spin = self.lattice[i, j]
        neighbors = self._get_neighbors(i, j)
        
        neighbor_sum = sum(self.lattice[ni, nj] for ni, nj in neighbors)
        
        old_energy = -self.params.J * old_spin * neighbor_sum - self.params.h * old_spin
        new_energy = -self.params.J * new_spin * neighbor_sum - self.params.h * new_spin
        
        return new_energy - old_energy
    
    def _energy_change_potts(self, i: int, j: int, new_state: int) -> float:
        """Calculate energy change for changing state in Potts model"""
        old_state = self.lattice[i, j]
        neighbors = self._get_neighbors(i, j)
        
        old_same_neighbors = sum(1 for ni, nj in neighbors if self.lattice[ni, nj] == old_state)
        new_same_neighbors = sum(1 for ni, nj in neighbors if self.lattice[ni, nj] == new_state)
        
        old_energy = -self.params.J * old_same_neighbors
        new_energy = -self.params.J * new_same_neighbors
        
        return new_energy - old_energy
    
    def total_energy(self) -> float:
        """Calculate total energy of the system"""
        if self.params.q == 2:
            return self._total_energy_ising()
        else:
            return self._total_energy_potts()
    
    def _total_energy_ising(self) -> float:
        """Calculate total energy for Ising model"""
        energy = 0.0
        L = self.params.L
        
        # Interaction energy (count each pair once)
        for i in range(L):
            for j in range(L):
                # Only count right and up neighbors to avoid double counting
                right_neighbor = (i + 1) % L
                up_neighbor = (j + 1) % L
                
                energy -= self.params.J * self.lattice[i, j] * self.lattice[right_neighbor, j]
                energy -= self.params.J * self.lattice[i, j] * self.lattice[i, up_neighbor]
        
        # External field energy
        energy -= self.params.h * np.sum(self.lattice)
        
        return energy
    
    def _total_energy_potts(self) -> float:
        """Calculate total energy for Potts model"""
        energy = 0.0
        L = self.params.L
        
        # Count same-state neighbor pairs (count each pair once)
        for i in range(L):
            for j in range(L):
                right_neighbor = (i + 1) % L
                up_neighbor = (j + 1) % L
                
                if self.lattice[i, j] == self.lattice[right_neighbor, j]:
                    energy -= self.params.J
                if self.lattice[i, j] == self.lattice[i, up_neighbor]:
                    energy -= self.params.J
        
        return energy
    
    def magnetization(self) -> float:
        """Calculate magnetization (only meaningful for Ising model)"""
        if self.params.q == 2:
            return np.mean(self.lattice)
        else:
            # For Potts model, return order parameter (fraction in largest cluster)
            unique, counts = np.unique(self.lattice, return_counts=True)
            return np.max(counts) / (self.params.L ** 2)
    
    def metropolis_step(self) -> None:
        """Perform one Monte Carlo step using Metropolis algorithm"""
        L = self.params.L
        
        for _ in range(L * L):  # One sweep
            # Choose random site
            i, j = np.random.randint(0, L, 2)
            
            if self.params.q == 2:
                # Ising model: flip spin
                new_spin = -self.lattice[i, j]
                delta_E = self._energy_change_ising(i, j, new_spin)
                
                # Metropolis acceptance criterion
                if delta_E <= 0 or np.random.random() < np.exp(-self.params.beta * delta_E):
                    self.lattice[i, j] = new_spin
            else:
                # Potts model: change to random different state
                current_state = self.lattice[i, j]
                possible_states = [s for s in range(1, self.params.q + 1) if s != current_state]
                new_state = np.random.choice(possible_states)
                
                delta_E = self._energy_change_potts(i, j, new_state)
                
                # Metropolis acceptance criterion
                if delta_E <= 0 or np.random.random() < np.exp(-self.params.beta * delta_E):
                    self.lattice[i, j] = new_state
    
    def simulate(self, n_steps: int, record_interval: int = 10) -> None:
        """Run Monte Carlo simulation"""
        self.energy_history = []
        self.magnetization_history = []
        
        for step in range(n_steps):
            self.metropolis_step()
            
            if step % record_interval == 0:
                self.energy_history.append(self.total_energy())
                self.magnetization_history.append(self.magnetization())
    
    def plot_lattice(self, title: str = "Lattice Configuration") -> None:
        """Visualize the current lattice configuration"""
        plt.figure(figsize=(8, 8))
        
        if self.params.q == 2:
            # Ising model: use red/blue for -1/+1
            plt.imshow(self.lattice, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(label='Spin')
        else:
            # Potts model: use discrete colors for different states
            plt.imshow(self.lattice, cmap='Set3', vmin=1, vmax=self.params.q)
            plt.colorbar(label='State', ticks=range(1, self.params.q + 1))
        
        plt.title(f"{title}\nT={self.params.T:.2f}, J={self.params.J:.2f}, q={self.params.q}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
    
    def plot_observables(self) -> None:
        """Plot energy and magnetization evolution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Energy plot
        ax1.plot(self.energy_history)
        ax1.set_xlabel('Monte Carlo Steps')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Magnetization/Order parameter plot
        ax2.plot(self.magnetization_history)
        ax2.set_xlabel('Monte Carlo Steps')
        if self.params.q == 2:
            ax2.set_ylabel('Magnetization')
            ax2.set_title('Magnetization Evolution')
        else:
            ax2.set_ylabel('Order Parameter')
            ax2.set_title('Order Parameter Evolution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class PhaseTransitionAnalyzer:
    """Analyze phase transitions in Ising-Potts models"""
    
    def __init__(self, base_params: ModelParameters):
        self.base_params = base_params
    
    def temperature_sweep(self, T_range: np.ndarray, n_steps: int = 3000, 
                         equilibration_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform temperature sweep to study phase transition
        
        Returns:
            temperatures, average_magnetizations, average_energies
        """
        magnetizations = []
        energies = []
        
        for T in T_range:
            # Create model with current temperature
            params = ModelParameters(
                J=self.base_params.J,
                h=self.base_params.h,
                T=T,
                q=self.base_params.q,
                L=self.base_params.L
            )
            
            model = IsingPottsModel(params)
            
            # Equilibration
            for _ in range(equilibration_steps):
                model.metropolis_step()
            
            # Measurement
            model.simulate(n_steps, record_interval=1)
            
            # Calculate averages (excluding initial transient)
            start_idx = len(model.energy_history) // 4  # Skip first 25%
            avg_energy = np.mean(model.energy_history[start_idx:]) / (params.L ** 2)
            avg_magnetization = np.mean(np.abs(model.magnetization_history[start_idx:]))
            
            energies.append(avg_energy)
            magnetizations.append(avg_magnetization)
            
            print(f"T={T:.3f}: E/site={avg_energy:.3f}, |M|={avg_magnetization:.3f}")
        
        return T_range, np.array(magnetizations), np.array(energies)
    
    def plot_phase_diagram(self, T_range: np.ndarray, magnetizations: np.ndarray, 
                          energies: np.ndarray) -> None:
        """Plot phase transition diagram"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Magnetization vs Temperature
        ax1.plot(T_range, magnetizations, 'bo-', markersize=4)
        ax1.set_xlabel('Temperature T')
        ax1.set_ylabel('|Magnetization|' if self.base_params.q == 2 else 'Order Parameter')
        ax1.set_title('Order Parameter vs Temperature')
        ax1.grid(True, alpha=0.3)
        
        # Energy vs Temperature
        ax2.plot(T_range, energies, 'ro-', markersize=4)
        ax2.set_xlabel('Temperature T')
        ax2.set_ylabel('Energy per site')
        ax2.set_title('Energy vs Temperature')
        ax2.grid(True, alpha=0.3)
        
        # Add critical temperature estimate for 2D Ising
        if self.base_params.q == 2:
            T_c_analytical = 2.0 / np.log(1 + np.sqrt(2))  # Onsager solution
            ax1.axvline(T_c_analytical, color='red', linestyle='--', alpha=0.7, 
                       label=f'T_c (analytical) = {T_c_analytical:.3f}')
            ax2.axvline(T_c_analytical, color='red', linestyle='--', alpha=0.7,
                       label=f'T_c (analytical) = {T_c_analytical:.3f}')
            ax1.legend()
            ax2.legend()
        
        plt.tight_layout()
        plt.show()


def critical_temperature_ising_2d() -> float:
    """Analytical critical temperature for 2D Ising model (Onsager solution)"""
    return 2.0 / np.log(1 + np.sqrt(2))


def demonstrate_ising_model():
    """Demonstrate the Ising model at different temperatures"""
    print("=== Ising Model Demonstration ===")
    
    # Parameters
    params_low_T = ModelParameters(J=1.0, h=0.0, T=1.0, q=2, L=30)
    params_high_T = ModelParameters(J=1.0, h=0.0, T=4.0, q=2, L=30)
    
    T_c = critical_temperature_ising_2d()
    print(f"Critical temperature (analytical): {T_c:.3f}")
    
    # Low temperature simulation
    print("\n--- Low Temperature (T=1.0) ---")
    model_low = IsingPottsModel(params_low_T)
    model_low.simulate(2000)
    
    print(f"Final energy per site: {model_low.total_energy()/(30*30):.3f}")
    print(f"Final magnetization: {model_low.magnetization():.3f}")
    
    # High temperature simulation
    print("\n--- High Temperature (T=4.0) ---")
    model_high = IsingPottsModel(params_high_T)
    model_high.simulate(2000)
    
    print(f"Final energy per site: {model_high.total_energy()/(30*30):.3f}")
    print(f"Final magnetization: {model_high.magnetization():.3f}")
    
    # Critical temperature
    params_critical = ModelParameters(J=1.0, h=0.0, T=T_c, q=2, L=30)
    model_critical = IsingPottsModel(params_critical)
    model_critical.simulate(2000)
    
    # Plot configurations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Lattice configurations
    axes[0, 0].imshow(model_low.lattice, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 0].set_title(f'T = {params_low_T.T:.1f} < T_c (Ordered)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    
    axes[0, 1].imshow(model_critical.lattice, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_title(f'T = {T_c:.2f} ≈ T_c (Critical)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    
    axes[0, 2].imshow(model_high.lattice, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 2].set_title(f'T = {params_high_T.T:.1f} > T_c (Disordered)')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    
    # Energy evolution
    axes[1, 0].plot(model_low.energy_history)
    axes[1, 0].set_title('Energy Evolution (Low T)')
    axes[1, 0].set_xlabel('MC Steps')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(model_critical.energy_history)
    axes[1, 1].set_title('Energy Evolution (Critical T)')
    axes[1, 1].set_xlabel('MC Steps')
    axes[1, 1].set_ylabel('Energy')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].plot(model_high.energy_history)
    axes[1, 2].set_title('Energy Evolution (High T)')
    axes[1, 2].set_xlabel('MC Steps')
    axes[1, 2].set_ylabel('Energy')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_potts_model():
    """Demonstrate the Potts model with different q values"""
    print("\n=== Potts Model Demonstration ===")
    
    q_values = [3, 4, 8]
    fig, axes = plt.subplots(1, len(q_values), figsize=(15, 4))
    
    for idx, q in enumerate(q_values):
        params = ModelParameters(J=1.0, T=1.5, q=q, L=30)
        model = IsingPottsModel(params)
        model.simulate(2000)
        
        print(f"q={q}: Final order parameter: {model.magnetization():.3f}")
        
        # Plot configuration
        im = axes[idx].imshow(model.lattice, cmap='Set3', vmin=1, vmax=q)
        axes[idx].set_title(f'Potts Model (q={q})')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx])
        cbar.set_label('State')
        cbar.set_ticks(range(1, q + 1))
    
    plt.tight_layout()
    plt.show()


def analyze_phase_transition():
    """Analyze phase transition in 2D Ising model"""
    print("\n=== Phase Transition Analysis ===")
    
    params = ModelParameters(J=1.0, h=0.0, q=2, L=20)  # Smaller lattice for faster computation
    analyzer = PhaseTransitionAnalyzer(params)
    
    # Temperature range around critical point
    T_c = critical_temperature_ising_2d()
    T_range = np.linspace(1.0, 4.0, 15)
    
    print("Performing temperature sweep...")
    temperatures, magnetizations, energies = analyzer.temperature_sweep(T_range, n_steps=1500)
    
    # Plot phase diagram
    analyzer.plot_phase_diagram(temperatures, magnetizations, energies)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Ising-Potts Model Simulation")
    print("=" * 40)
    
    # Demonstrate Ising model
    demonstrate_ising_model()
    
    # Demonstrate Potts model
    demonstrate_potts_model()
    
    # Analyze phase transition
    analyze_phase_transition()
    
    print("\nSimulation completed!")