import sys
import igl
import polyscope as ps
import polyscope.imgui as psim
import optimizeWeights
import numpy as np


def main():
    mesh_file = "./data/decimated-knight.off" if len(sys.argv) == 1 else sys.argv[1]
    vertices, faces = igl.read_triangle_mesh(mesh_file)

    ps.init()
    ps_mesh = ps.register_surface_mesh("my mesh", vertices, faces)
    colours = np.array([[0, 0.5, 0.5]]*len(vertices))
    control_points_i = []
    flag = False
    point_cloud = None
    ncp = 3

    def callback():
        nonlocal flag, point_cloud, ncp

        changed, ncp = psim.InputInt("# control points: ", ncp, step=1, step_fast=10)

        if psim.Button("Random points"):
            control_points_i.clear()
            control_points_i.extend(np.random.randint(1, vertices.shape[0], ncp).tolist())
            point_cloud = ps.register_point_cloud("my points", vertices[control_points_i],
                                    radius=0.005, color=[0, 0, 0], point_render_mode='sphere')
            c = optimizeWeights.runOptimization(vertices, faces, control_points_i)[::]
            colours[::] = c
            ps_mesh.add_color_quantity("quasi harmonics colors", colours, enabled=True)

        if psim.Button("Custom points"):
            control_points_i.clear()
            flag = True

        if flag:
            selected = ps.get_selection()
            if selected[0] != '' and selected[1] < len(vertices):
                control_points_i.append(selected[1])
                ps.set_selection('', 0)
            psim.TextUnformatted("{} of {} custom points selected".format(len(control_points_i), ncp))
            psim.Separator()
            if len(control_points_i) == ncp:
                point_cloud = ps.register_point_cloud("my points", vertices[control_points_i],
                                        radius=0.005, color=[0, 0, 0], point_render_mode='sphere')
                colours[::] = optimizeWeights.runOptimization(vertices, faces, control_points_i)
                ps_mesh.add_color_quantity("quasi harmonics colors", colours, enabled=True)
                flag = False

        # if psim.Button("Reset"):
        #     control_points_i.clear()
        #     point_cloud.remove()
        #     colours[::] = np.array([[0, 0.5, 0.5]]*len(vertices))

    ps.set_user_callback(callback)
    ps.show()
    ps.clear_user_callback()


if __name__ == '__main__':
    sys.exit(main())