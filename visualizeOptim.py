import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as psim
import utility
import time


with open('wts.txt', 'r') as f:
    mesh_file = f.readline()
    cp_i = [int(i) for i in f.readline().split(", ")]
    num_iter = int(f.readline())
vertices, faces = igl.read_triangle_mesh(mesh_file[:-1])
ps.init()
ps_mesh = ps.register_surface_mesh("my mesh", vertices, faces)
ps.register_point_cloud("control points", vertices[cp_i], radius=0.01, color=[0, 0, 0], point_render_mode='sphere')
wts = np.genfromtxt("wts.txt", skip_header=3)
colors = utility.get_Colours(len(cp_i))
i = 0


def callback():
    global i

    if i < num_iter:
        C = np.matmul(wts[i*vertices.shape[0]:(i+1)*vertices.shape[0], :], colors)
        ps_mesh.add_color_quantity("quasi harmonics colors", C, enabled=True)
        time.sleep(0.2)
        i += 1

    if psim.Button("Restart"):
        i = 0


ps.set_user_callback(callback)
ps.show()
ps.clear_user_callback()