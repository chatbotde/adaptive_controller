import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# Print initial message
print("Starting hypersonic vehicle adaptive control simulation...")

# Basic constants
R0 = 6356766  # Earth radius (m)
g = 9.81      # Gravity (m/s^2)
m_nominal = 1000.0  # Mass (kg)
S_nominal = 0.5     # Area (m^2)

print(f"Using model parameters: Earth radius = {R0/1000:.1f} km, gravity = {g} m/s², nominal mass = {m_nominal} kg, nominal area = {S_nominal} m²")

def get_density(h):
    rho0 = 1.225  # Sea level density (kg/m^3)
    h_scale = 7000.0
    return rho0 * np.exp(-h / h_scale)

def CL0(Ma):
    return 0.1 + 0.01 * Ma

def CL_alpha(Ma):
    return 0.8 + 0.05 * Ma

def CD0(Ma):
    return 0.02 + 0.005 * Ma

def K(Ma):
    return 0.2 - 0.01 * np.min([Ma, 10])

def calculate_mach(V, h):
    T = 288.15 - 0.0065 * min(h, 11000)
    a = np.sqrt(1.4 * 287 * T)
    return V / a

class HypersonicVehicle:
    def __init__(self, mass_uncertainty=0.0, aero_uncertainty=0.0):
        self.mass = m_nominal * (1.0 + mass_uncertainty)
        self.S = S_nominal
        self.aero_uncertainty = aero_uncertainty
        print(f"Created vehicle with mass = {self.mass:.1f} kg (uncertainty: {mass_uncertainty*100:.1f}%), " +
              f"aero uncertainty: {aero_uncertainty*100:.1f}%")
        
    def dynamics(self, t, state, alpha):
        H, R, V, gamma = state
        
        rho = get_density(H)
        Ma = calculate_mach(V, H)
        q = 0.5 * rho * V**2
        
        CL_val = (CL0(Ma) + CL_alpha(Ma) * alpha) * (1.0 + self.aero_uncertainty)
        CD_val = (CD0(Ma) + K(Ma) * CL_val**2) * (1.0 + self.aero_uncertainty)
        
        L = CL_val * q * self.S
        D = CD_val * q * self.S
        
        dH = V * np.sin(gamma)
        dR = V * np.cos(gamma) / (R0 + H)
        dV = -D / self.mass - g * np.sin(gamma)
        dgamma = L / (self.mass * V) - g * np.cos(gamma) / V + V * np.cos(gamma) / (R0 + H)
        
        return [dH, dR, dV, dgamma]

class ReferenceModel:
    def __init__(self, omega_n=0.5, zeta=1):
        self.omega_n = omega_n
        self.zeta = zeta
        print(f"Created reference model with natural frequency = {omega_n} rad/s, damping ratio = {zeta}")
        
    def dynamics(self, t, state, gamma_cmd):
        gamma_ref, dgamma_ref = state
        ddgamma_ref = self.omega_n**2 * (gamma_cmd - gamma_ref) - 2 * self.zeta * self.omega_n * dgamma_ref
        return [dgamma_ref, ddgamma_ref]

class AdaptiveController:
    def __init__(self, adaptation_rate=1.0, gamma_cmd_function=None):
        self.gamma_cmd_function = gamma_cmd_function
        self.adaptation_rate = adaptation_rate
        self.current_time = 0.0
        
        self.theta1 = 1.0
        self.theta2 = 0.5
        self.ki = 1
        self.I = 0
        
        print(f"Created adaptive controller with adaptation rate = {adaptation_rate}")
        print(f"Initial adaptive parameters: theta1 = {self.theta1}, theta2 = {self.theta2}, ki = {self.ki}")
        
        self.ref_model = ReferenceModel()
        self.ref_state = [0.0, 0.0]
        
    def update_reference(self, dt):
        current_gamma_cmd = self.gamma_cmd_function(self.current_time)
        
        sol = solve_ivp(
            lambda t, y: self.ref_model.dynamics(t, y, current_gamma_cmd),
            [0, dt],
            self.ref_state,
            method='RK45'
        )
        self.ref_state = sol.y[:, -1]
        self.current_time += dt
        
        return self.ref_state[0], current_gamma_cmd
    
    def calculate_control(self, state, dt):
        _, _, V, gamma = state
        gamma_ref = self.ref_state[0]
        dgamma_ref = self.ref_state[1]
        
        e = gamma - gamma_ref
        self.I += e*dt
        
        self.theta1 += self.adaptation_rate * e * dt
        self.theta2 += self.adaptation_rate * e * dgamma_ref * dt

        alpha = -(self.theta1 * e + self.theta2 * dgamma_ref + self.ki*self.I)
        alpha = np.clip(alpha, -15 * np.pi/180, 15 * np.pi/180)
        
        return alpha

def gamma_cmd_function(t):
    return -2 * np.pi/180

sim_time = 240.0
dt = 1
t_values = np.arange(0, sim_time, dt)

H0 = 20000
R0_sim = 0
V0 = 2000
gamma0 = 0.0

initial_state = [H0, R0_sim, V0, gamma0]
print(f"Initial conditions: Altitude = {H0/1000:.1f} km, Velocity = {V0:.1f} m/s, Flight path angle = {gamma0*180/np.pi:.1f} deg")
print(f"Simulation parameters: Time = {sim_time:.1f} s, Time step = {dt:.1f} s")
print(f"Command flight path angle = {gamma_cmd_function(0)*180/np.pi:.1f} deg")

def run_simulation(mass_uncertainty=0.0, aero_uncertainty=0.0, adaptation_rate=5.0):
    print(f"\nStarting simulation run with mass uncertainty = {mass_uncertainty*100:.1f}%, " +
          f"aero uncertainty = {aero_uncertainty*100:.1f}%, adaptation rate = {adaptation_rate}")
    
    sim_start_time = time.time()
    vehicle = HypersonicVehicle(mass_uncertainty, aero_uncertainty)
    controller = AdaptiveController(adaptation_rate, gamma_cmd_function)
    
    state = initial_state.copy()
    states_history = [state.copy()]
    alpha_history = []
    gamma_ref_history = [gamma0]
    gamma_cmd_history = [gamma_cmd_function(0.0)]
    theta1_history = [controller.theta1]
    theta2_history = [controller.theta2]
    
    progress_interval = max(1, len(t_values) // 10)  # Show progress about 10 times
    
    for i, t in enumerate(t_values[:-1]):
        if i % progress_interval == 0:
            print(f"  Progress: {i/len(t_values)*100:.1f}% complete, t = {t:.1f}s, " +
                  f"altitude = {state[0]/1000:.2f} km, velocity = {state[2]:.1f} m/s, " +
                  f"gamma = {state[3]*180/np.pi:.2f} deg")
        
        gamma_ref, current_cmd = controller.update_reference(dt)
        gamma_ref_history.append(gamma_ref)
        gamma_cmd_history.append(current_cmd)
        
        alpha = controller.calculate_control(state, dt)
        alpha_history.append(alpha)
        
        theta1_history.append(controller.theta1)
        theta2_history.append(controller.theta2)
        
        sub_steps = 5  
        sub_dt = dt / sub_steps
        current_state = state.copy()
        
        for _ in range(sub_steps):
            sol = solve_ivp(
                lambda t, y: vehicle.dynamics(t, y, alpha),
                [0, sub_dt],
                current_state,
                method='RK45',
                rtol=1e-6,
                atol=1e-9
            )
            current_state = sol.y[:, -1]
        
        state = current_state
        states_history.append(state.copy())
    
    sim_end_time = time.time()
    print(f"  Simulation complete! Runtime: {sim_end_time - sim_start_time:.2f} seconds")
    print(f"  Final state: altitude = {state[0]/1000:.2f} km, velocity = {state[2]:.1f} m/s, " +
          f"gamma = {state[3]*180/np.pi:.2f} deg")
    print(f"  Final adaptive parameters: theta1 = {controller.theta1:.4f}, theta2 = {controller.theta2:.4f}")
    
    return np.array(states_history), np.array(alpha_history), np.array(gamma_ref_history), np.array(gamma_cmd_history), np.array(theta1_history), np.array(theta2_history)

adaptation_rate = 5.0

print("\n==================== TESTING MASS UNCERTAINTIES ====================")
# Test mass uncertainties
mass_uncertainties = [0.0, 0.2, 0.4]
results_mass = []

for mass_unc in mass_uncertainties:
    result = run_simulation(mass_uncertainty=mass_unc, aero_uncertainty=0.0, adaptation_rate=adaptation_rate)
    results_mass.append(result)

print("Generating mass uncertainty plots...")
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
for i, mass_unc in enumerate(mass_uncertainties):
    plt.plot(t_values, results_mass[i][0][:, 3] * 180/np.pi, label=f'Mass Unc = {mass_unc*100:.0f}%')
plt.plot(t_values, results_mass[0][2] * 180/np.pi, 'k--', label='Reference')
plt.plot(t_values, results_mass[0][3] * 180/np.pi, 'r--', label='Command')
plt.xlabel('Time (s)')
plt.ylabel('Flight Path Angle (deg)')
plt.legend()
plt.grid(True)
plt.title('Flight Path Angle Tracking (Different Mass Uncertainties)')

plt.subplot(2, 3, 2)
for i, mass_unc in enumerate(mass_uncertainties):
    plt.plot(t_values, (results_mass[i][0][:, 3] - results_mass[i][2]) * 180/np.pi, label=f'Mass Unc = {mass_unc*100:.0f}%')
plt.xlabel('Time (s)')
plt.ylabel('Tracking Error (deg)')
plt.legend()
plt.grid(True)
plt.title('Flight Path Angle Tracking Error')

plt.subplot(2, 3, 3)
for i, mass_unc in enumerate(mass_uncertainties):
    plt.plot(t_values[:-1], results_mass[i][1] * 180/np.pi, label=f'Mass Unc = {mass_unc*100:.0f}%')
plt.xlabel('Time (s)')
plt.ylabel('Angle of Attack (deg)')
plt.legend()
plt.grid(True)
plt.title('Control Input (Angle of Attack)')

plt.subplot(2, 3, 4)
for i, mass_unc in enumerate(mass_uncertainties):
    plt.plot(t_values, results_mass[i][4], label=f'Mass Unc = {mass_unc*100:.0f}%')
plt.xlabel('Time (s)')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)
plt.title('Adaptive Parameter θ1')

plt.subplot(2, 3, 5)
for i, mass_unc in enumerate(mass_uncertainties):
    plt.plot(t_values, results_mass[i][5], label=f'Mass Unc = {mass_unc*100:.0f}%')
plt.xlabel('Time (s)')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)
plt.title('Adaptive Parameter θ2')

plt.subplot(2, 3, 6)
for i, mass_unc in enumerate(mass_uncertainties):
    plt.plot(t_values, results_mass[i][0][:, 0]/1000, label=f'Mass Unc = {mass_unc*100:.0f}%')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (km)')
plt.legend()
plt.grid(True)
plt.title('Altitude')

plt.tight_layout()
print("Saving mass uncertainty plot to 'hypersonic_adaptive_control_mass_uncertainties.png'")
plt.savefig('hypersonic_adaptive_control_mass_uncertainties.png', dpi=300)

print("\n==================== TESTING AERODYNAMIC UNCERTAINTIES ====================")
# Test aero uncertainties
aero_uncertainties = [-0.15, 0.0, 0.15]
results_aero = []

for aero_unc in aero_uncertainties:
    result = run_simulation(mass_uncertainty=0.0, aero_uncertainty=aero_unc, adaptation_rate=adaptation_rate)
    results_aero.append(result)

print("Generating aerodynamic uncertainty plots...")
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
for i, aero_unc in enumerate(aero_uncertainties):
    plt.plot(t_values, results_aero[i][0][:, 3] * 180/np.pi, label=f'Aero Unc = {aero_unc*100:.0f}%')
plt.plot(t_values, results_aero[2][2] * 180/np.pi, 'k--', label='Reference')
plt.plot(t_values, results_aero[2][3] * 180/np.pi, 'r--', label='Command')
plt.xlabel('Time (s)')
plt.ylabel('Flight Path Angle (deg)')
plt.legend()
plt.grid(True)
plt.title('Flight Path Angle Tracking (Different Aerodynamic Uncertainties)')

plt.subplot(2, 3, 2)
for i, aero_unc in enumerate(aero_uncertainties):
    plt.plot(t_values, (results_aero[i][0][:, 3] - results_aero[i][2]) * 180/np.pi, label=f'Aero Unc = {aero_unc*100:.0f}%')
plt.xlabel('Time (s)')
plt.ylabel('Tracking Error (deg)')
plt.legend()
plt.grid(True)
plt.title('Flight Path Angle Tracking Error')

plt.subplot(2, 3, 3)
for i, aero_unc in enumerate(aero_uncertainties):
    plt.plot(t_values[:-1], results_aero[i][1] * 180/np.pi, label=f'Aero Unc = {aero_unc*100:.0f}%')
plt.xlabel('Time (s)')
plt.ylabel('Angle of Attack (deg)')
plt.legend()
plt.grid(True)
plt.title('Control Input (Angle of Attack)')

plt.subplot(2, 3, 4)
for i, aero_unc in enumerate(aero_uncertainties):
    plt.plot(t_values, results_aero[i][4], label=f'Aero Unc = {aero_unc*100:.0f}%')
plt.xlabel('Time (s)')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)
plt.title('Adaptive Parameter θ1')

plt.subplot(2, 3, 5)
for i, aero_unc in enumerate(aero_uncertainties):
    plt.plot(t_values, results_aero[i][5], label=f'Aero Unc = {aero_unc*100:.0f}%')
plt.xlabel('Time (s)')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)
plt.title('Adaptive Parameter θ2')

plt.subplot(2, 3, 6)
for i, aero_unc in enumerate(aero_uncertainties):
    plt.plot(t_values, results_aero[i][0][:, 2], label=f'Aero Unc = {aero_unc*100:.0f}%')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.title('Velocity')

plt.tight_layout()
print("Saving aerodynamic uncertainty plot to 'hypersonic_adaptive_control_aero_uncertainties.png'")
plt.savefig('hypersonic_adaptive_control_aero_uncertainties.png', dpi=300)

print("\n==================== TESTING COMBINED UNCERTAINTIES ====================")
# Test combined uncertainties
combined_cases = [
    {"mass_unc": 0.0, "aero_unc": 0.0, "label": "Nominal"},
    {"mass_unc": 0.4, "aero_unc": -0.15, "label": "Mass +40%, Aero -15%"},
    {"mass_unc": -0.4, "aero_unc": 0.15, "label": "Mass -40%, Aero +15%"},
]

results_combined = []

for case in combined_cases:
    result = run_simulation(
        mass_uncertainty=case["mass_unc"], 
        aero_uncertainty=case["aero_unc"], 
        adaptation_rate=adaptation_rate
    )
    results_combined.append(result)

print("Generating combined uncertainty plots...")
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
for i, case in enumerate(combined_cases):
    plt.plot(t_values, results_combined[i][0][:, 3] * 180/np.pi, label=case["label"])
plt.plot(t_values, results_combined[0][2] * 180/np.pi, 'k--', label='Reference')
plt.plot(t_values, results_combined[0][3] * 180/np.pi, 'r--', label='Command')
plt.xlabel('Time (s)')
plt.ylabel('Flight Path Angle (deg)')
plt.legend()
plt.grid(True)
plt.title('Flight Path Angle Tracking (Combined Uncertainties)')

plt.subplot(2, 3, 2)
for i, case in enumerate(combined_cases):
    plt.plot(t_values, (results_combined[i][0][:, 3] - results_combined[i][2]) * 180/np.pi, label=case["label"])
plt.xlabel('Time (s)')
plt.ylabel('Tracking Error (deg)')
plt.legend()
plt.grid(True)
plt.title('Flight Path Angle Tracking Error')

plt.subplot(2, 3, 3)
for i, case in enumerate(combined_cases):
    plt.plot(t_values[:-1], results_combined[i][1] * 180/np.pi, label=case["label"])
plt.xlabel('Time (s)')
plt.ylabel('Angle of Attack (deg)')
plt.legend()
plt.grid(True)
plt.title('Control Input (Angle of Attack)')

plt.subplot(2, 3, 4)
for i, case in enumerate(combined_cases):
    plt.plot(t_values, results_combined[i][4], label=case["label"])
plt.xlabel('Time (s)')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)
plt.title('Adaptive Parameter θ1')

plt.subplot(2, 3, 5)
for i, case in enumerate(combined_cases):
    plt.plot(t_values, results_combined[i][5], label=case["label"])
plt.xlabel('Time (s)')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)
plt.title('Adaptive Parameter θ2')

plt.subplot(2, 3, 6)
for i, case in enumerate(combined_cases):
    plt.plot(t_values, results_combined[i][0][:, 0]/1000, label=case["label"])
plt.xlabel('Time (s)')
plt.ylabel('Altitude (km)')
plt.legend()
plt.grid(True)
plt.title('Altitude')

plt.tight_layout()
print("Saving combined uncertainty plot to 'hypersonic_adaptive_control_combined_uncertainties.png'")
plt.savefig('hypersonic_adaptive_control_combined_uncertainties.png', dpi=300)

print("\n==================== TESTING ADAPTATION RATES ====================")
# Test adaptation rates
adaptation_rates = [1.0, 5.0, 10.0]
results_rates = []

for rate in adaptation_rates:
    result = run_simulation(mass_uncertainty=0.0, aero_uncertainty=0.0, adaptation_rate=rate)
    results_rates.append(result)

print("Generating adaptation rate plots...")
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
for i, rate in enumerate(adaptation_rates):
    plt.plot(t_values, results_rates[i][0][:, 3] * 180/np.pi, label=f'Rate = {rate}')
plt.plot(t_values, results_rates[0][2] * 180/np.pi, 'k--', label='Reference')
plt.plot(t_values, results_rates[0][3] * 180/np.pi, 'r--', label='Command')
plt.xlabel('Time (s)')
plt.ylabel('Flight Path Angle (deg)')
plt.legend()
plt.grid(True)
plt.title('Flight Path Angle Tracking (Different Adaptation Rates)')

plt.subplot(2, 3, 2)
for i, rate in enumerate(adaptation_rates):
    plt.plot(t_values, (results_rates[i][0][:, 3] - results_rates[i][2]) * 180/np.pi, label=f'Rate = {rate}')
plt.xlabel('Time (s)')
plt.ylabel('Tracking Error (deg)')
plt.legend()
plt.grid(True)
plt.title('Flight Path Angle Tracking Error')

plt.subplot(2, 3, 3)
for i, rate in enumerate(adaptation_rates):
    plt.plot(t_values[:-1], results_rates[i][1] * 180/np.pi, label=f'Rate = {rate}')
plt.xlabel('Time (s)')
plt.ylabel('Angle of Attack (deg)')
plt.legend()
plt.grid(True)
plt.title('Control Input (Angle of Attack)')

plt.subplot(2, 3, 4)
for i, rate in enumerate(adaptation_rates):
    plt.plot(t_values, results_rates[i][4], label=f'Rate = {rate}')
plt.xlabel('Time (s)')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)
plt.title('Adaptive Parameter θ1')

plt.subplot(2, 3, 5)
for i, rate in enumerate(adaptation_rates):
    plt.plot(t_values, results_rates[i][5], label=f'Rate = {rate}')
plt.xlabel('Time (s)')
plt.ylabel('Parameter Value')
plt.legend()
plt.grid(True)
plt.title('Adaptive Parameter θ2')

plt.subplot(2, 3, 6)
for i, rate in enumerate(adaptation_rates):
    plt.plot(t_values, results_rates[i][0][:, 0]/1000, label=f'Rate = {rate}')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (km)')
plt.legend()
plt.grid(True)
plt.title('Altitude')

plt.tight_layout()
print("Saving adaptation rate plot to 'hypersonic_adaptive_control_adaptation_rates.png'")
plt.savefig('hypersonic_adaptive_control_adaptation_rates.png', dpi=300)

print("\nAll simulations and plots complete!")
print("Plot files created:")
print("- hypersonic_adaptive_control_mass_uncertainties.png")
print("- hypersonic_adaptive_control_aero_uncertainties.png")
print("- hypersonic_adaptive_control_combined_uncertainties.png")
print("- hypersonic_adaptive_control_adaptation_rates.png")

# Show plots if in an interactive environment
try:
    plt.show()
    print("Plots displayed. Close plot windows to continue.")
except:
    print("Non-interactive environment detected. Plots saved to files but not displayed.")