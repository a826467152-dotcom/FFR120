import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# [SIMULATION CLASSES - UNCHANGED]
# ==========================================
def generate_grid_cloth(width_steps, height_steps, spacing=0.04, model_type=1):
    num_particles = width_steps * height_steps
    V = np.zeros((num_particles, 3))
    for y in range(height_steps):
        for x in range(width_steps):
            idx = y * width_steps + x
            V[idx] = [(x - width_steps/2)*spacing, (y - height_steps/2)*spacing, 0]

    edges = []
    def add_edge(u, v): edges.append([u, v])

    for y in range(height_steps):
        for x in range(width_steps):
            idx = y * width_steps + x
            if x < width_steps - 1: add_edge(idx, idx + 1)
            if y < height_steps - 1: add_edge(idx, idx + width_steps)
            if model_type >= 2:
                if x < width_steps - 1 and y < height_steps - 1: add_edge(idx, idx + width_steps + 1)
                if x > 0 and y < height_steps - 1: add_edge(idx, idx + width_steps - 1)
            if model_type >= 3:
                if x < width_steps - 2: add_edge(idx, idx + 2)
                if y < height_steps - 2: add_edge(idx, idx + 2 * width_steps)

    E = np.array(edges, dtype=np.int32)
    wall_indices = [y * width_steps for y in range(height_steps)]
    stick_indices = [y * width_steps + (width_steps - 1) for y in range(height_steps)]
    return V, E, np.array(wall_indices), np.array(stick_indices)

class FastSolver:
    def __init__(self, V, E, wall_idx, stick_idx, k=100.0):
        self.pos = V.copy()
        self.edges = E
        self.k = k
        self.dt = 1.0/30.0 
        self.gravity = np.array([0, 0, 0]) 
        self.wall_idx = wall_idx
        self.stick_idx = stick_idx
        self.pin_indices = np.concatenate([wall_idx, stick_idx])
        self.free_indices = np.setdiff1d(np.arange(len(V)), self.pin_indices)
        p1 = V[E[:, 0]]; p2 = V[E[:, 1]]
        self.rest_lengths = np.linalg.norm(p1 - p2, axis=1)
        n = len(V)
        self.base_L = np.zeros((n, n))
        np.add.at(self.base_L, (E[:,0], E[:,0]), 1.0)
        np.add.at(self.base_L, (E[:,1], E[:,1]), 1.0)
        np.add.at(self.base_L, (E[:,0], E[:,1]), -1.0)
        np.add.at(self.base_L, (E[:,1], E[:,0]), -1.0)
        self.update_stiffness(k)

    def update_stiffness(self, new_k):
        self.k = new_k
        h2 = self.dt ** 2
        L_ff = self.base_L[np.ix_(self.free_indices, self.free_indices)]
        self.A_ff = np.eye(len(self.free_indices)) + (h2 * self.k) * L_ff
        L_fp = self.base_L[np.ix_(self.free_indices, self.pin_indices)]
        self.A_fp = (h2 * self.k) * L_fp
        self.A_inv = np.linalg.inv(self.A_ff)

    def set_boundary_twist(self, angle_deg):
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        stick_pos = self.pos[self.stick_idx]
        center = np.mean(stick_pos, axis=0)
        rel = stick_pos - center
        new_y = rel[:, 1] * c - rel[:, 2] * s
        new_z = rel[:, 1] * s + rel[:, 2] * c
        self.pos[self.stick_idx, 1] = center[1] + new_y
        self.pos[self.stick_idx, 2] = center[2] + new_z

    def solve_static_equilibrium(self, steps=100, local_iters=10):
        h2 = self.dt ** 2
        boundary_term = self.A_fp @ self.pos[self.pin_indices]
        gravity_displacement = h2 * self.gravity
        force_history = []  # To store avg force at each step

        for _ in range(steps):
            y = self.pos.copy() 
            for _ in range(local_iters):
                p1 = self.pos[self.edges[:, 0]]
                p2 = self.pos[self.edges[:, 1]]
                diff = p1 - p2
                curr_len = np.linalg.norm(diff, axis=1)
                curr_len[curr_len < 1e-6] = 1e-6 
                d = diff * (self.rest_lengths / curr_len)[:, None]
                Jd = np.zeros_like(self.pos)
                force_mag = self.k * d
                np.add.at(Jd, self.edges[:, 0], force_mag)
                np.add.at(Jd, self.edges[:, 1], -force_mag)
                rhs = y + gravity_displacement + h2 * Jd
                rhs_free = rhs[self.free_indices] - boundary_term
                self.pos[self.free_indices] = self.A_inv @ rhs_free
            
            # Record average force after each step
            current_avg_force, _, _ = self.get_avg_force()
            force_history.append(current_avg_force)
            
        return force_history

    def get_avg_force(self):
        p1 = self.pos[self.edges[:, 0]]
        p2 = self.pos[self.edges[:, 1]]
        current_len = np.linalg.norm(p1 - p2, axis=1)
        forces = self.k * np.abs(current_len - self.rest_lengths)
        return np.mean(forces), forces, self.rest_lengths

# ==========================================
# 3. Calibration & SPACED VISUALIZATION
# ==========================================
def run_calibration_experiment():
    W, H = 15, 35
    twist_angle = 60
    base_k_model1 = 2000.0
    tolerance = 1e-3
    spacing = 0.04 
    
    print(f"=== Starting Calibration (Target Error < {tolerance}) ===")
    
    V, E, wall, stick = generate_grid_cloth(W, H, model_type=1)
    ref_solver = FastSolver(V, E, wall, stick, k=base_k_model1)
    ref_solver.set_boundary_twist(twist_angle)
    model1_history = ref_solver.solve_static_equilibrium(steps=100, local_iters=10) 
    target_force, _, _ = ref_solver.get_avg_force()
    print(f"    -> Target Force: {target_force:.5f} N")
    
    results = { 'Model 1': ref_solver }
    history_data = { 'Model 1': model1_history }
    
    for m in [2, 3]:
        model_name = f"Model {m}"
        print(f"\n[2] Calibrating {model_name}...")
        low_k = 1.0; high_k = base_k_model1 * 1.5 
        best_solver = None
        
        for i in range(50):
            mid_k = (low_k + high_k) / 2.0
            V_tmp, E_tmp, w_tmp, s_tmp = generate_grid_cloth(W, H, model_type=m)
            sim = FastSolver(V_tmp, E_tmp, w_tmp, s_tmp, k=mid_k)
            sim.set_boundary_twist(twist_angle)
            force_hist = sim.solve_static_equilibrium(steps=100, local_iters=10) 
            current_force, _, _ = sim.get_avg_force()
            diff = current_force - target_force
            rel_error = abs(diff) / target_force
            
            if rel_error < tolerance:
                print(f"    >>> CONVERGED! k={mid_k:.2f}, F={current_force:.3f}")
                best_solver = sim
                history_data[model_name] = force_hist
                break
            
            if diff > 0: high_k = mid_k
            else: low_k = mid_k
        results[model_name] = best_solver

    # [3] PLOTTING (SPACED OUT)
    print("\n[3] Plotting Final State...")
    fig = plt.figure(figsize=(18, 7)) # Increased height for more vertical room
    
    valid_solvers = [s for s in results.values() if s is not None]
    if not valid_solvers: return

    all_forces = []
    for s in valid_solvers:
        _, f, _ = s.get_avg_force()
        all_forces.extend(f)
    vmax = np.percentile(all_forces, 99)

    sorted_keys = sorted(results.keys())
    
    for i, key in enumerate(sorted_keys):
        s = results[key]
        if s is None: continue

        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        avg_f, forces, rest_lens = s.get_avg_force()
        
        p1 = s.pos[s.edges[:, 0]]
        p2 = s.pos[s.edges[:, 1]]
        
        for j in range(len(forces)):
            L0 = rest_lens[j]
            is_structural = (L0 < spacing * 1.2) 
            if not is_structural and forces[j] < (0.10 * vmax): 
                continue 
            
            c = plt.cm.jet(min(forces[j]/vmax, 1.0))
            
            if is_structural:
                lw = 1.5
                alph = 1.0
            else:
                lw = 0.5
                alph = 0.15 
            
            ax.plot([p1[j,0], p2[j,0]], [p1[j,1], p2[j,1]], [p1[j,2], p2[j,2]], 
                    color=c, linewidth=lw, alpha=alph)
        
        ax.scatter(s.pos[s.wall_idx,0], s.pos[s.wall_idx,1], s.pos[s.wall_idx,2], c='k', s=5)
        ax.scatter(s.pos[s.stick_idx,0], s.pos[s.stick_idx,1], s.pos[s.stick_idx,2], c='r', s=5)

        # FIXED TITLE SPACING
        ax.set_title(f"{key}\nk={s.k:.1f}", fontweight='bold', fontsize=11, pad=20)
        
        ax.set_xlabel('X'); ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 0.5])
        ax.view_init(elev=25, azim=-70)
        
        # MOVED TEXT BOX LOWER
        ax.text2D(0.5, -0.15, f"Avg F: {avg_f:.3f}N", 
                  transform=ax.transAxes, ha='center', 
                  bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))

    # ADJUST OVERALL LAYOUT TO PREVENT CROWDING
    # Increase top margin so Suptitle doesn't hit subplots
    plt.subplots_adjust(left=0.05, right=0.9, top=0.80, bottom=0.1, wspace=0.15)
    
    cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.6])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Spring Force (N)')

    plt.suptitle(f"Stiffness Calibration (Tolerance < 1e-3)", fontsize=16, y=0.95)
    plt.show()

    # [4] Plotting Force History
    print("\n[4] Plotting Force History...")
    plt.figure(figsize=(10, 6))
    for key in sorted_keys:
        if key in history_data:
            plt.plot(history_data[key], label=key)
            
    plt.xlabel("Simulation Steps")
    plt.ylabel("Average Force (N)")
    plt.title("Average Force Convergence Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_calibration_experiment()