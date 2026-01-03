import numpy as np
import polyscope as ps
import time
import os
from datetime import datetime
from PIL import Image

# ==========================================
# 1. Texture / Image Handling
# ==========================================
def generate_texture_colors(width_steps, height_steps, image_path):
    """
    Loads an image, handles transparency, resizes it to the grid, 
    and returns the RGB colors for every particle.
    """
    num_particles = width_steps * height_steps
    
    # Check if file exists
    if os.path.exists(image_path):
        print(f"Loading texture from: {image_path}")
        try:
            # 1. Load Image
            img = Image.open(image_path)
            
            # 2. Handle Transparency (RGBA -> RGB with White Background)
            if img.mode == 'RGBA':
                # Create a white background
                background = Image.new("RGB", img.size, (255, 255, 255))
                # Paste the image on top using the alpha channel as a mask
                background.paste(img, mask=img.split()[3]) 
                img = background
            else:
                img = img.convert("RGB")
            
            # 3. Resize image to match grid density exactly
            # LANCZOS filter ensures the text stays sharp when resizing
            img = img.resize((width_steps, height_steps), Image.Resampling.LANCZOS)
            
            # 4. Convert to Numpy Array and normalize to 0.0 - 1.0
            data = np.array(img).astype(float) / 255.0
            
            # 5. Flatten to (num_particles, 3)
            # The grid generation loop is Y then X, which matches image data structure
            colors = data.reshape(-1, 3)
            return colors
            
        except Exception as e:
            print(f"Error loading image: {e}")
            print("Falling back to default pattern.")
    else:
        print(f"ERROR: Image file not found at: {image_path}")

    # FALLBACK: Procedural Checkerboard
    print("Generating procedural checkerboard...")
    colors = np.zeros((num_particles, 3))
    for y in range(height_steps):
        for x in range(width_steps):
            idx = y * width_steps + x
            if (x // 4 + y // 4) % 2 == 0:
                colors[idx] = [0.9, 0.9, 0.9] 
            else:
                colors[idx] = [0.2, 0.2, 0.8] 
    return colors

# ==========================================
# 2. Mesh Generation
# ==========================================
def generate_grid_cloth(width_steps, height_steps, spacing, height_offset):
    print(f"Generating {width_steps}x{height_steps} cloth...")
    
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
    def add_edge(u, v):
        edges.append([u, v])

    for y in range(height_steps):
        for x in range(width_steps):
            idx = y * width_steps + x
            
            # Structural
            if x < width_steps - 1: add_edge(idx, idx + 1)
            if y < height_steps - 1: add_edge(idx, idx + width_steps)
            
            # Shear
            if x < width_steps - 1 and y < height_steps - 1: add_edge(idx, idx + width_steps + 1)
            if x > 0 and y < height_steps - 1: add_edge(idx, idx + width_steps - 1)
            
            # Bending
            if x < width_steps - 2: add_edge(idx, idx + 2)
            if y < height_steps - 2: add_edge(idx, idx + 2 * width_steps)

    E = np.array(edges, dtype=np.int32)

    # Generate Faces for rendering
    faces = []
    for y in range(height_steps - 1):
        for x in range(width_steps - 1):
            p00 = y * width_steps + x
            p10 = y * width_steps + (x + 1)
            p01 = (y + 1) * width_steps + x
            p11 = (y + 1) * width_steps + (x + 1)
            faces.append([p00, p01, p10])
            faces.append([p10, p01, p11])
            
    F = np.array(faces, dtype=np.int32)
    
    # Identify corners
    tl = 0
    tr = width_steps - 1
    bl = (height_steps - 1) * width_steps
    br = num_particles - 1
    corners = {'tl': tl, 'tr': tr, 'bl': bl, 'br': br}
    
    return V, F, E, corners

# ==========================================
# 3. XPBD Physics Engine
# ==========================================
class ClothSolver:
    def __init__(self, V, E, corners):
        self.num_particles = V.shape[0]
        self.pos = V.copy()       
        self.prev_pos = V.copy()    
        self.vel = np.zeros_like(V) 
        self.inv_mass = np.ones(self.num_particles) 
        
        # Constraints
        self.edges = E
        
        # Pre-compute valence for Jacobi averaging
        self.valences = np.zeros(self.num_particles, dtype=np.float32)
        idx1 = E[:, 0]
        idx2 = E[:, 1]
        np.add.at(self.valences, idx1, 1)
        np.add.at(self.valences, idx2, 1)
        self.valences[self.valences == 0] = 1.0 
        
        # Calculate initial rest lengths
        p1 = V[E[:, 0]]
        p2 = V[E[:, 1]]
        self.rest_lengths = np.linalg.norm(p1 - p2, axis=1)
        
        # Physics Parameters
        self.substeps = 10          # Slightly reduced for performance at high res
        self.dt = 1.0 / 60.0        
        self.gravity = np.array([0, 0, -9.8])
        self.compliance = 0.000001  
        self.damping = 0.995        
        
        # Collision Parameters
        self.ball_active = True
        self.ball_center = np.array([0.0, 0.0, 0.0])
        self.ball_radius = 0.4
        self.collision_thickness = 0.02 
        self.friction = 0.6
        self.restitution = 0.9      
        
        # Pins
        self.pin_indices = []
        self.pin_positions = []
        
        # For Visualization
        self.collision_mask = np.zeros(self.num_particles, dtype=bool)

    def set_pins(self, indices):
        self.pin_indices = indices
        if len(indices) > 0:
            self.pin_positions = self.pos[indices].copy()
            self.inv_mass[indices] = 0.0
        
        mask = np.ones(self.num_particles, dtype=bool)
        if len(indices) > 0:
            mask[indices] = False
        self.inv_mass[mask] = 1.0

    def solve_distance_constraints(self, dt_sub):
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]
        p1 = self.pos[idx1]
        p2 = self.pos[idx2]
        
        diff = p1 - p2
        dist = np.linalg.norm(diff, axis=1)
        dist[dist < 1e-9] = 1e-9
        
        alpha = self.compliance / (dt_sub * dt_sub)
        w1 = self.inv_mass[idx1]
        w2 = self.inv_mass[idx2]
        w_sum = w1 + w2
        
        n = diff / dist[:, None]
        C = dist - self.rest_lengths
        delta_lambda = -C / (w_sum + alpha + 1e-6) 
        correction = n * delta_lambda[:, None]
        
        grad1 = w1[:, None] * correction
        grad2 = -w2[:, None] * correction
        
        grad1 /= self.valences[idx1][:, None]
        grad2 /= self.valences[idx2][:, None]
        
        np.add.at(self.pos, idx1, grad1)
        np.add.at(self.pos, idx2, grad2)

    def solve_collision(self):
        if not self.ball_active:
            self.collision_mask[:] = False
            return

        diff = self.pos - self.ball_center
        dist = np.linalg.norm(diff, axis=1)
        
        effective_radius = self.ball_radius + self.collision_thickness
        colliding = dist < effective_radius
        self.collision_mask = colliding
        
        if not np.any(colliding):
            return
            
        safe_dist = dist[colliding]
        safe_dist[safe_dist < 1e-9] = 1e-9
        
        normals = diff[colliding] / safe_dist[:, None]
        contact_points = self.ball_center + normals * effective_radius
        
        incoming_vec = self.pos[colliding] - self.prev_pos[colliding]
        v_dot_n = np.einsum('ij,ij->i', incoming_vec, normals)
        
        incoming_normal = normals * v_dot_n[:, None]
        incoming_tangent = incoming_vec - incoming_normal
        new_tangent = incoming_tangent * (1.0 - self.friction)
        
        restitution_factor = np.where(v_dot_n < 0, -self.restitution, 1.0)
        new_normal = incoming_normal * restitution_factor[:, None]
        
        self.pos[colliding] = contact_points
        self.prev_pos[colliding] = contact_points - (new_tangent + new_normal)

    def update(self):
        dt_sub = self.dt / self.substeps
        
        for _ in range(self.substeps):
            self.vel += self.gravity * dt_sub
            self.prev_pos = self.pos.copy()
            self.pos += self.vel * dt_sub
            
            self.solve_distance_constraints(dt_sub)
            
            if len(self.pin_indices) > 0:
                self.pos[self.pin_indices] = self.pin_positions
            
            self.solve_collision()
            
            self.vel = (self.pos - self.prev_pos) / dt_sub
            self.vel *= (1.0 - (1.0 - self.damping) / self.substeps)

# ==========================================
# 4. GUI & Main
# ==========================================

# Settings State
state = {
    'running': True,
    'solver': None,
    'mesh_object': None,
    'corners': None,
    'drop_height': 1.2, 
    'camera_dist': 3.0,
    'texture_colors': None,
    # High Resolution Settings
    'width': 80,       # Increased from 35 to 80 for sharper image
    'height': 80,
    'spacing': 0.018   # Decreased spacing to keep cloth size normal (~1.44m)
}

def callback():
    if state['running']:
        state['solver'].update()
        
        # Update mesh in polyscope
        state['mesh_object'].update_vertex_positions(state['solver'].pos)
        
        # Visualize Collision + Texture
        current_colors = state['texture_colors'].copy()
        
        mask = state['solver'].collision_mask
        if np.any(mask):
            # Blend Red into the texture where colliding
            current_colors[mask] = current_colors[mask] * 0.5 + np.array([0.5, 0.0, 0.0])
            
        state['mesh_object'].add_color_quantity("Cloth Texture", current_colors, enabled=True)

def apply_axonometric_view():
    ps.set_view_projection_mode("orthographic")
    view_vec = np.array([1.0, -1.0, 1.0])
    view_vec /= np.linalg.norm(view_vec)
    target = np.array([0.0, 0.0, 0.6])
    eye = target + view_vec * state['camera_dist']
    ps.look_at(eye, target, np.array([0.0, 0.0, 1.0]))

def main():
    # 1. PATH FIX: Find where THIS script is running
    script_folder = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_folder, "chalmers_logo.png")
    
    # 2. Setup Data (High Res)
    V, F, E, corners = generate_grid_cloth(state['width'], state['height'], state['spacing'], state['drop_height'])
    
    # 3. Load Texture
    print(f"--- Looking for image at: {image_path} ---")
    state['texture_colors'] = generate_texture_colors(state['width'], state['height'], image_path)
    
    # 4. Init Solver
    solver = ClothSolver(V, E, corners)
    
    # Start pinned - ALL 4 CORNERS
    all_pins = [corners['tl'], corners['tr'], corners['bl'], corners['br']]
    solver.set_pins(all_pins)
    
    state['solver'] = solver
    state['corners'] = corners
    
    # Create screenshots directory
    screenshot_dir = os.path.join(script_folder, "screenshots")
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    
    # 5. Init Polyscope
    ps.init()
    ps.set_program_name("Chalmers Cloth Sim")
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Register Ball
    ps.register_point_cloud("Ball", np.array([solver.ball_center]), 
                            radius=solver.ball_radius, 
                            point_render_mode='sphere')
                            
    # Register Cloth
    state['mesh_object'] = ps.register_surface_mesh("Cloth", V, F)
    state['mesh_object'].set_smooth_shade(True)
    state['mesh_object'].add_color_quantity("Cloth Texture", state['texture_colors'], enabled=True)

    # 6. GUI Callback
    def gui_callback():
        s = state['solver']
        
        ps.imgui.Text("Simulation Control")
        if ps.imgui.Button("Run/Pause"):
            state['running'] = not state['running']
        
        if ps.imgui.Button("Reset (Pin All 4)"):
            # Reset logic must also use High Res settings
            V_new, _, _, _ = generate_grid_cloth(state['width'], state['height'], state['spacing'], state['drop_height'])
            s.pos = V_new.copy()
            s.prev_pos = V_new.copy()
            s.vel = np.zeros_like(V_new)
            
            # Recalculate corners for new size
            tl, tr = 0, state['width'] - 1
            bl, br = (state['height'] - 1) * state['width'], (state['width'] * state['height']) - 1
            state['corners'] = {'tl': tl, 'tr': tr, 'bl': bl, 'br': br}
            
            s.set_pins([tl, tr, bl, br])
            
            # Reload texture in case sizing changed
            state['texture_colors'] = generate_texture_colors(state['width'], state['height'], image_path)
            state['mesh_object'].update_vertex_positions(s.pos)

        ps.imgui.Separator()
        ps.imgui.Text("Actions")
        
        if ps.imgui.Button("Release Bottom Corners"):
            c = state['corners']
            s.set_pins([c['tl'], c['tr']])
            
        if ps.imgui.Button("Drop All (Release 4)"):
            s.set_pins([])

        if ps.imgui.Button("Pin All 4 (Current Pos)"):
             c = state['corners']
             s.set_pins([c['tl'], c['tr'], c['bl'], c['br']])

        ps.imgui.Separator()
        ps.imgui.Text("Parameters")
        
        # Camera
        changed_cam, state['camera_dist'] = ps.imgui.SliderFloat("Cam Dist", state['camera_dist'], 1.0, 8.0)
        if changed_cam: apply_axonometric_view()
        if ps.imgui.Button("Set Axonometric"): apply_axonometric_view()

        ps.imgui.Separator()
        
        # Physics Params
        _, state['drop_height'] = ps.imgui.SliderFloat("Drop Height", state['drop_height'], 0.5, 2.5)
        _, s.ball_radius = ps.imgui.SliderFloat("Ball Radius", s.ball_radius, 0.1, 0.5)
        
        # Update ball visual if changed
        ps.register_point_cloud("Ball", np.array([s.ball_center]), 
                            radius=s.ball_radius, point_render_mode='sphere')

        _, s.friction = ps.imgui.SliderFloat("Friction", s.friction, 0.0, 1.0)
        _, s.substeps = ps.imgui.SliderInt("Substeps", s.substeps, 1, 20)
        _, s.collision_thickness = ps.imgui.SliderFloat("Thickness", s.collision_thickness, 0.1, 0.3)

        stiffness = 1.0 / (s.compliance * 1e7 + 0.001)
        changed, new_stiff = ps.imgui.SliderFloat("Stiffness", stiffness, 0.0, 5.0)
        if changed:
            if new_stiff < 0.001: new_stiff = 0.001
            s.compliance = (1.0/new_stiff - 0.001) / 1e7
            
        if ps.imgui.Button("Screenshot"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(screenshot_dir, f"sim_{timestamp}.png")
            ps.screenshot(fname)
            print(f"Saved: {fname}")

        callback()

    ps.set_user_callback(gui_callback)
    ps.show()

if __name__ == "__main__":
    main()