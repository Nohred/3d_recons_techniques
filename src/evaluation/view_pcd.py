import sys, time
import numpy as np
import open3d as o3d
import matplotlib
from plyfile import PlyData   # pip install plyfile
SH_C0 = 0.28209479177387814
matplotlib.use("Agg")
from matplotlib import cm

def load(path):
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        pcd = o3d.io.read_triangle_mesh(path).sample_points_uniformly(200_000)

    # 3DGS: reconstruir RGB desde f_dc_*
    if path.lower().endswith(".ply"):
        try:
            v = PlyData.read(path)["vertex"]
            if "f_dc_0" in v.data.dtype.names:
                f = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
                rgb = np.clip(0.5 + SH_C0 * f, 0, 1)
                pcd.points = o3d.utility.Vector3dVector(np.stack([v["x"], v["y"], v["z"]], 1))
                pcd.colors = o3d.utility.Vector3dVector(rgb)
                print(f"{path}: colores reconstruidos desde 3DGS (f_dc)")
        except Exception as e:
            print("plyfile falló:", e)

    # Diagnóstico y normalización
    if pcd.has_colors():
        c = np.asarray(pcd.colors)
        print(f"{path}: color min={c.min():.3f} max={c.max():.3f} mean={c.mean():.3f}")
        if c.max() > 1.0:
            pcd.colors = o3d.utility.Vector3dVector(np.clip(c / 255.0, 0, 1))
        elif c.max() < 1e-6:                      # todo negro => color inútil
            print("  -> colores vacíos, uso color por altura")
            pcd.colors = o3d.utility.Vector3dVector(height_colors(pcd))
    return pcd

def height_colors(pcd, axis=2):
    z = np.asarray(pcd.points)[:, axis]
    t = (z - z.min()) / (np.ptp(z) + 1e-12)
    return cm.turbo(t)[:, :3]

paths = sys.argv[1:]
pcds = [load(p) for p in paths]
palette = [[.9,.2,.2],[.2,.5,.9],[.2,.8,.3],[.9,.7,.1]]
orig_colors = [np.asarray(p.colors).copy() if p.has_colors() else None for p in pcds]
for i, p in enumerate(pcds):
    print(f"{paths[i]}: {len(p.points):,} pts | bbox extent = {p.get_axis_aligned_bounding_box().get_extent()}")
    if orig_colors[i] is None:
        p.paint_uniform_color(palette[i % 4]); orig_colors[i] = np.asarray(p.colors).copy()

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Point cloud viewer  (H = ayuda de Open3D)", 1280, 800)
for p in pcds: vis.add_geometry(p)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2 * np.linalg.norm(pcds[0].get_axis_aligned_bounding_box().get_extent()))
bboxes = [p.get_axis_aligned_bounding_box() for p in pcds]
for b in bboxes: b.color = (0, 0, 0)
state = dict(axes=False, bbox=False, dark=True, spin=False, size=2.0, frame=0)
opt = vis.get_render_option()
opt.light_on = False      # evita que las normales oscurezcan los puntos
opt.point_size = state["size"]; opt.background_color = np.array([.05,.05,.05])

def toggle_axes(v):
    (v.remove_geometry if state["axes"] else v.add_geometry)(axes, reset_bounding_box=False)
    state["axes"] ^= True; return False
def toggle_bbox(v):
    for b in bboxes: (v.remove_geometry if state["bbox"] else v.add_geometry)(b, reset_bounding_box=False)
    state["bbox"] ^= True; return False
def toggle_bg(v):
    state["dark"] ^= True
    opt.background_color = np.array([.05]*3 if state["dark"] else [1.]*3); return False
def toggle_spin(v): state["spin"] ^= True; return False
def color_height(v):
    for p in pcds: p.colors = o3d.utility.Vector3dVector(height_colors(p)); v.update_geometry(p)
    return False
def color_orig(v):
    for p, c in zip(pcds, orig_colors): p.colors = o3d.utility.Vector3dVector(c); v.update_geometry(p)
    return False
def color_normals(v):
    for p in pcds:
        if not p.has_normals(): p.estimate_normals()
        p.colors = o3d.utility.Vector3dVector((np.asarray(p.normals) + 1) / 2); v.update_geometry(p)
    return False
def screenshot(v):
    name = f"shot_{int(time.time())}.png"; v.capture_screen_image(name); print("guardado", name); return False
def size(delta):
    def f(v):
        state["size"] = float(np.clip(state["size"] + delta, 1, 20)); opt.point_size = state["size"]; return False
    return f
def spin(v):
    if state["spin"]: v.get_view_control().rotate(5.0, 0.0)
    return False
def view(front, up):
    def f(v):
        c = v.get_view_control(); c.set_front(front); c.set_up(up); c.set_lookat(pcds[0].get_center()); return False
    return f

K = vis.register_key_callback
K(ord("X"), toggle_axes); K(ord("O"), toggle_bbox); K(ord("V"), toggle_bg)
K(ord("T"), toggle_spin); K(ord("1"), color_orig); K(ord("2"), color_height)
K(ord("3"), color_normals); K(ord("P"), screenshot)
K(ord("="), size(+1)); K(ord("-"), size(-1))
K(ord("F"), view([0,-1,0],[0,0,1]))      # frontal
K(ord("G"), view([1,0,0],[0,0,1]))       # lateral
K(ord("U"), view([0,0,1],[0,1,0]))       # cenital
vis.register_animation_callback(spin)

print("""
MOUSE: izq=rotar | rueda=zoom | Ctrl+izq o Shift+izq=pan | der=pan
TECLAS: 1 color original | 2 color por altura | 3 color por normales
        X ejes | O bounding box | V fondo claro/oscuro | T turntable
        F/G/U vistas frontal/lateral/cenital | +/- tamaño de punto
        P screenshot | H ayuda de Open3D | Q salir
""")
vis.run(); vis.destroy_window()
