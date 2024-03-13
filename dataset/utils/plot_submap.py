#
# Created on Tue Apr 26 2023
# The MIT License (MIT)
# Copyright (c) 2023 Yun-Jin Li (Jim)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def plot_submap(submap_path):
    submap = np.load(submap_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(submap)
    print(f"Plot Submap that has {submap.shape[0]} points")
    o3d.visualization.draw_geometries([pcd])

def plot_submap_matplot(submap_path: str, save_path: str = None, cam_centered=False, screen=False, verbose=False):
    submap = np.load(submap_path)
    if verbose:
        print(f"Plot Submap that has {submap.shape[0]} points")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(submap[:, 0], submap[:, 1], submap[:, 2], s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if not cam_centered:
        ax.set_xlim(0, 50)
        ax.set_ylim(-25, 25)
    else:
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
    # ax.set_zlim(-25, 25)
    ax.view_init(elev=-150, azim=30)
    if save_path is None:
        pass
    else:
        plt.savefig(save_path)
    if screen:
        plt.show()
    plt.close(fig)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot a submap')
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--matplot', action='store_true')
    parser.add_argument('--cam_centered', action='store_true')
    args = parser.parse_args()
    if args.matplot:
        plot_submap_matplot(args.path, args.cam_centered)
    else:
        plot_submap(args.path)