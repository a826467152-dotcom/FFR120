import numpy as np
import time
import matplotlib.pyplot as plt

# ==========================================
# 1. Mesh Generation
# ==========================================
def generate_grid_cloth(width_steps, height_steps, spacing=0.04, height_offset=1.2):
    num_particles = width_steps * height_steps
    V = np.zeros((num_particles, 3))
    
    # Cloth centered at (0,0) in XY, height in Z
    for y in range(height_steps):
        for x in range(width_steps):
            idx = y * width_steps + x
            vx = (x * spacing) - (width_steps * spacing / 2.0)
            vy = (y * spacing) - (height_steps * spacing / 2.0)
            V[idx] = [vx, vy, height_offset]

    edges = []
    def add_edge(u, v): edges.append([u, v])

    for y in range(height_steps):
        for x in range(width_steps):
            idx = y * width_steps + x
            if x < width_steps - 1: add_edge(idx, idx + 1)
            if y < height_steps - 1: add_edge(idx, idx + width_steps)
            if x < width_steps - 1 and y < height_steps - 1: add_edge(idx, idx + width_steps + 1)
            if x > 0 and y < height_steps - 1: add_edge(idx, idx + width_steps - 1)
            if x < width_steps - 2: add_edge(idx, idx + 2)
            if y < height_steps - 2: add_edge(idx, idx + 2 * width_steps)

    E = np.array(edges, dtype=np.int32)
    
    corners = {
        'tl': 0, 'tr': width_steps - 1, 
        'bl': (height_steps - 1) * width_steps, 'br': num_particles - 1
    }
    
    return V, E, corners

# ==========================================
# 2. Liu et al. Solver (Implicit Euler)
# ==========================================
class FastMassSpringSolver:
    def __init__(self, V, E, corners, stiffness=800.0, damping=2.0, dt=1.0/60.0):
        self.name = "Liu et al. (Implicit)"
        self.num_particles = V.shape[0]
        self.pos = V.copy()
        self.prev_pos = V.copy()
        self.vel = np.zeros_like(V)
        self.edges = E
        
        self.iterations = 10 
        self.gravity = np.array([0, 0, -9.8])
        self.k = stiffness
        self.dt = dt
        
        self.pin_indices = [corners['tl'], corners['tr']]
        self.all_indices = np.arange(self.num_particles)
        self.free_indices = np.setdiff1d(self.all_indices, self.pin_indices)
        
        p1 = V[E[:, 0]]; p2 = V[E[:, 1]]
        self.rest_lengths = np.linalg.norm(p1 - p2, axis=1)
        
        self.base_L = np.zeros((self.num_particles, self.num_particles))
        np.add.at(self.base_L, (E[:,0], E[:,0]), 1.0)
        np.add.at(self.base_L, (E[:,1], E[:,1]), 1.0)
        np.add.at(self.base_L, (E[:,0], E[:,1]), -1.0)
        np.add.at(self.base_L, (E[:,1], E[:,0]), -1.0)
        
        self.A_inv = None
        self.A_fp = None
        self.last_k = -1.0; self.last_dt = -1.0
        
        # Damping
        dt_approx = 1.0/60.0
        self.drag_factor = max(0.0, 1.0 - damping * dt_approx)
        self.update_system_matrix()

    def update_system_matrix(self):
        if abs(self.k - self.last_k) < 1e-5 and abs(self.dt - self.last_dt) < 1e-7: return
        h = self.dt; h2 = h * h
        L_ff = self.base_L[np.ix_(self.free_indices, self.free_indices)]
        L_fp = self.base_L[np.ix_(self.free_indices, self.pin_indices)]
        I_reduced = np.eye(len(self.free_indices))
        A_ff = I_reduced + (h2 * self.k) * L_ff
        self.A_fp = (h2 * self.k) * L_fp
        self.A_inv = np.linalg.inv(A_ff)
        self.last_k = self.k; self.last_dt = self.dt

    def update(self, dt):
        self.dt = dt
        self.update_system_matrix()
        h = self.dt; h2 = h*h
        
        pred_vel = self.vel * self.drag_factor
        y = self.pos + h * pred_vel + h2 * self.gravity
        y[self.pin_indices] = self.pos[self.pin_indices]
        
        x = y.copy()
        pin_pos = self.pos[self.pin_indices]
        boundary_correction = self.A_fp @ pin_pos
        
        for _ in range(self.iterations):
            p1 = x[self.edges[:, 0]]; p2 = x[self.edges[:, 1]]
            diff = p1 - p2
            curr_dist = np.linalg.norm(diff, axis=1)
            curr_dist[curr_dist < 1e-9] = 1e-9
            d = diff * (self.rest_lengths / curr_dist)[:, None]
            
            Jd = np.zeros_like(x)
            force_term = self.k * d
            np.add.at(Jd, self.edges[:, 0], force_term)
            np.add.at(Jd, self.edges[:, 1], -force_term)
            
            RHS_full = y + h2 * Jd
            x_free = self.A_inv @ (RHS_full[self.free_indices] - boundary_correction)
            x[self.free_indices] = x_free
            
        self.prev_pos = self.pos.copy()
        self.pos = x
        self.vel = (self.pos - self.prev_pos) / h
        
        floor_mask = self.pos[:, 2] < 0
        self.pos[floor_mask, 2] = 0; self.vel[floor_mask] = 0

# ==========================================
# 3. Explicit RK Solver
# ==========================================
class ExplicitRKSolver:
    def __init__(self, V, E, corners, mode="RK4", stiffness=800.0, damping=2.0):
        self.name = f"Explicit {mode}"
        self.mode = mode 
        self.num_particles = V.shape[0]
        self.pos = V.copy()
        self.vel = np.zeros_like(V)
        self.mass = 1.0
        self.inv_mass = np.ones(self.num_particles) / self.mass
        self.edges = E
        
        p1 = V[E[:, 0]]; p2 = V[E[:, 1]]
        self.rest_lengths = np.linalg.norm(p1 - p2, axis=1)
        
        self.gravity = np.array([0, 0, -9.8])
        self.k_stiffness = stiffness
        self.k_damping = damping
        
        self.pin_indices = [corners['tl'], corners['tr']]
        self.inv_mass[self.pin_indices] = 0.0

    def compute_forces(self, pos, vel):
        forces = np.zeros_like(pos)
        forces += self.gravity * self.mass
        
        idx1, idx2 = self.edges[:, 0], self.edges[:, 1]
        p1, p2 = pos[idx1], pos[idx2]
        delta = p1 - p2
        dist = np.linalg.norm(delta, axis=1)
        dist[dist < 1e-9] = 1e-9 
        
        displacement = dist - self.rest_lengths
        force_mag = -self.k_stiffness * displacement
        force_vec = (delta / dist[:, None]) * force_mag[:, None]
        
        np.add.at(forces, idx1, force_vec)
        np.add.at(forces, idx2, -force_vec)
        
        forces -= self.k_damping * vel
        forces[self.pin_indices] = 0.0
        return forces

    def get_derivative(self, state):
        n = self.num_particles
        pos = state[:n]
        vel = state[n:]
        forces = self.compute_forces(pos, vel)
        acc = forces * self.inv_mass[:, None]
        return np.concatenate([vel, acc])

    def update(self, dt):
        state = np.concatenate([self.pos, self.vel])
        
        if self.mode == "RK2":
            k1 = self.get_derivative(state)
            k2 = self.get_derivative(state + dt * k1)
            new_state = state + dt * 0.5 * (k1 + k2)
        elif self.mode == "RK4":
            k1 = self.get_derivative(state)
            k2 = self.get_derivative(state + 0.5 * dt * k1)
            k3 = self.get_derivative(state + 0.5 * dt * k2)
            k4 = self.get_derivative(state + dt * k3)
            new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
        n = self.num_particles
        self.pos = new_state[:n]; self.vel = new_state[n:]
        
        floor_mask = self.pos[:, 2] < 0
        if np.any(floor_mask):
            self.pos[floor_mask, 2] = 0; self.vel[floor_mask] *= 0.5; self.vel[floor_mask, 2] = 0

# ==========================================
# 4. Simulation & Plotting
# ==========================================

def compute_energy(solver, k):
    v2 = np.sum(solver.vel**2, axis=1)
    ke = 0.5 * 1.0 * np.sum(v2)
    pe_g = np.sum(1.0 * 9.8 * solver.pos[:, 2])
    p1 = solver.pos[solver.edges[:, 0]]
    p2 = solver.pos[solver.edges[:, 1]]
    dist = np.linalg.norm(p1 - p2, axis=1)
    pe_s = 0.5 * k * np.sum((dist - solver.rest_lengths)**2)
    return ke, pe_g + pe_s

def run_headless():
    print("Initializing Simulation...")
    
    # Simulation Parameters
    width, height = 25, 25 
    stiffness_k = 800.0  
    damping_d = 2.0       
    sim_dt = 1.0/150.0    
    
    # Run duration
    simulation_seconds = 5.0 
    total_frames = int(simulation_seconds / sim_dt)
    
    # Generate Cloth Data (Shared)
    V, E, corners = generate_grid_cloth(width, height)
    
    # --- Define Configurations (5 Scenarios) ---
    # We store: (Solver Instance, Substeps, Display Name, Base Method Name)
    configs = []
    
    # 1. Liu (Implicit) - 1 step
    s_liu = FastMassSpringSolver(V, E, corners, stiffness=stiffness_k, damping=damping_d)
    configs.append({'solver': s_liu, 'substeps': 1, 'name': 'Liu (Implicit)', 'group': 'Liu'})

    # 2. RK2 - 1 step
    s_rk2_1 = ExplicitRKSolver(V, E, corners, mode="RK2", stiffness=stiffness_k, damping=damping_d)
    configs.append({'solver': s_rk2_1, 'substeps': 1, 'name': 'RK2 (1 sub)', 'group': 'RK2'})
    
    # 3. RK4 - 1 step
    s_rk4_1 = ExplicitRKSolver(V, E, corners, mode="RK4", stiffness=stiffness_k, damping=damping_d)
    configs.append({'solver': s_rk4_1, 'substeps': 1, 'name': 'RK4 (1 sub)', 'group': 'RK4'})

    # 4. RK2 - 5 steps
    s_rk2_5 = ExplicitRKSolver(V, E, corners, mode="RK2", stiffness=stiffness_k, damping=damping_d)
    configs.append({'solver': s_rk2_5, 'substeps': 5, 'name': 'RK2 (5 sub)', 'group': 'RK2'})

    # 5. RK4 - 5 steps
    s_rk4_5 = ExplicitRKSolver(V, E, corners, mode="RK4", stiffness=stiffness_k, damping=damping_d)
    configs.append({'solver': s_rk4_5, 'substeps': 5, 'name': 'RK4 (5 sub)', 'group': 'RK4'})
    
    # Reference Solver (Ground Truth)
    ref_solver = ExplicitRKSolver(V, E, corners, mode="RK4", stiffness=stiffness_k, damping=damping_d)
    
    # Data Storage
    history = {c['name']: {'time': [], 'rmse': [], 'energy': [], 'energy_ratio': [], 'calc_time': []} for c in configs}
    stats = {c['name']: {'settled_time': None, 'total_calc_time': 0} for c in configs}
    
    print(f"Running {total_frames} frames ({simulation_seconds}s physics time) at dt={sim_dt:.5f}s...")
    
    sim_time = 0.0
    
    for frame in range(total_frames):
        if frame % 100 == 0:
            print(f"Processing frame {frame}/{total_frames}...")

        # 1. Update Reference (20x oversampling for Ground Truth)
        dt_ref = sim_dt / 20.0
        for _ in range(20):
            ref_solver.update(dt_ref)
        
        _, ref_total_e = compute_energy(ref_solver, stiffness_k)
        if ref_total_e < 1e-9: ref_total_e = 1e-9 
            
        # 2. Update All Configurations
        for cfg in configs:
            solver = cfg['solver']
            substeps = cfg['substeps']
            name = cfg['name']
            
            t_start = time.perf_counter()
            
            # --- The Loop Update ---
            dt_sub = sim_dt / substeps
            for _ in range(substeps):
                solver.update(dt_sub)
                    
            t_end = time.perf_counter()
            dt_calc_ms = (t_end - t_start) * 1000.0
            
            # Metrics
            diff = solver.pos - ref_solver.pos
            rmse = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            ke, total_e = compute_energy(solver, stiffness_k)
            
            ratio = total_e / ref_total_e
            
            # Store Data
            history[name]['time'].append(sim_time)
            history[name]['rmse'].append(rmse)
            history[name]['energy'].append(total_e)
            history[name]['energy_ratio'].append(ratio)
            history[name]['calc_time'].append(dt_calc_ms)
            
            stats[name]['total_calc_time'] += dt_calc_ms

            # Stability Check
            if ke < 0.5:
                if stats[name]['settled_time'] is None:
                    stats[name]['settled_time'] = sim_time
            else:
                stats[name]['settled_time'] = None 

        sim_time += sim_dt

    print("\nSimulation Complete. Generating Plots...")
    
    # --- PLOTTING ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cloth Solver Comparison (k={stiffness_k}, dt={sim_dt*1000:.1f}ms)", fontsize=16)
    
    # 1. RMSE vs Time
    ax1 = axs[0, 0]
    for name, data in history.items():
        # Add small epsilon to avoid log(0)
        ax1.semilogy(data['time'], np.array(data['rmse']) + 1e-9, label=name, linewidth=2)
    ax1.set_title("RMSE Error (Log Scale)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RMSE (m)")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend()
    
    # 2. Energy Ratio vs Time
    ax2 = axs[0, 1]
    for name, data in history.items():
        ax2.plot(data['time'], data['energy_ratio'], label=name, linewidth=2)
    ax2.set_title("Energy Ratio (E_solver / E_ref)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Ratio")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.5) 
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    # [cite_start]3. Calculation Time (Grouped Bar Chart) [cite: 181]
    # We want groups: Liu, RK2, RK4
    ax3 = axs[1, 0]
    
    groups = ['Liu', 'RK2', 'RK4']
    
    # Extract avg times
    avg_times = {}
    for name in history:
        avg_times[name] = np.mean(history[name]['calc_time'])
        
    # Prepare data for plotting
    # Format: [Liu_val, RK2_val, RK4_val]
    times_1sub = [
        avg_times['Liu (Implicit)'], 
        avg_times['RK2 (1 sub)'], 
        avg_times['RK4 (1 sub)']
    ]
    
    times_5sub = [
        0, # Liu doesn't have 5 sub
        avg_times['RK2 (5 sub)'],
        avg_times['RK4 (5 sub)']
    ]
    
    x = np.arange(len(groups))
    width = 0.35  # width of the bars

    # Plot Group 1 (1 Substep)
    rects1 = ax3.bar(x - width/2, times_1sub, width, label='1 Substep', color='tab:blue')
    # Plot Group 2 (5 Substeps)
    rects2 = ax3.bar(x + width/2, times_5sub, width, label='5 Substeps', color='tab:orange')

    ax3.set_title("Avg Calculation Time per Frame")
    ax3.set_ylabel("Time (ms)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(groups)
    ax3.legend()
    
    # Add labels on top of bars
    ax3.bar_label(rects1, fmt='%.2f', padding=3)
    ax3.bar_label(rects2, fmt='%.2f', padding=3)
    
    # 4. Settling Time
    ax4 = axs[1, 1]
    settled_times = []
    names = list(history.keys())
    for n in names:
        t = stats[n]['settled_time']
        settled_times.append(t if t is not None else simulation_seconds)
    
    # Using simple distinct colors for the 5 bars here
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars2 = ax4.bar(names, settled_times, color=colors)
    ax4.set_title("Time to Stabilize (KE < 0.5J)")
    ax4.set_ylabel("Time (s)")
    ax4.tick_params(axis='x', rotation=45) # Rotate labels to fit
    
    for i, rect in enumerate(bars2):
        height = rect.get_height()
        t_val = stats[names[i]]['settled_time']
        label = 'Not Settled' if t_val is None else f'{height:.2f}s'
        ax4.text(rect.get_x() + rect.get_width()/2.0, height, label, ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    run_headless()