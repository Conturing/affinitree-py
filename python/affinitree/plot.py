#   Copyright 2025 affinitree developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from typing import Callable, Iterable, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import QhullError

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.typing import ColorType
from mpl_toolkits import mplot3d
import seaborn as sns

from affinitree import *


class LedgerDiscrete:
    
    def __init__(self, cmap : Union[str, Sequence[ColorType], None] = 'tab10', names=None, color_idx=None, absolute_mapping=True, num=None, title=None, position='right'):
        self.cmap = cmap
        if names is None:
            self.names = dict()
        else:
            self.names = names
        self.color_idx = color_idx
        self.absolute_mapping = absolute_mapping
        self.num = num
        self.title = title
        self.position = position
            
    def fit(self, dd: AffTree):
        values = set()

        for terminal in dd.terminals():
            values.add(int(terminal.val.bias.sum().item()))
        
        self.vmax = int(max(values))
        self.vmin = int(min(values))
        
        if self.num is not None:
            pass
        elif self.color_idx is not None:
            self.num = max(self.color_idx.items(), key=lambda x: x[0])
        elif self.absolute_mapping:
            self.num = self.vmax + 1
        else:
            self.num = len(values)
        
        self.color_palette = sns.color_palette(self.cmap, self.num)
        self.values = values
    
    def map_color(self, node: AffNode) -> Tuple:
        idx = node.val.bias.sum().item()
        if self.color_idx is not None:
            return self.color_palette[self.color_idx[idx]]
        elif self.absolute_mapping:
            return self.color_palette[int(idx)]
    
    def create_legend(self, fig, ax):
        patches = []
        for idx in self.values:
            if self.color_idx is not None:
                patches.append(mpatches.Patch(
                    color=self.color_palette[self.color_idx[idx]], 
                    label=self.names.get(idx, f'{idx}')
                ))
            elif self.absolute_mapping:
                patches.append(mpatches.Patch(
                    color=self.color_palette[idx], 
                    label=self.names.get(idx, f'{idx}')
                ))
        if self.position == 'top':
            ax.legend(handles=patches, bbox_to_anchor=(0, 1), loc='lower left', title=self.title, ncols=len(patches), fontsize='small')
        else:
            ax.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper right', title=self.title)


class LedgerContinuous:
    
    def __init__(self, cmap='flare', title=None, position='right'):
        self.cmap = cmap
        self.title = title
        self.position = position
            
    def fit(self, dd: AffTree):
        values = set()

        for terminal in dd.terminals():
            values.add((np.linalg.norm(terminal.val.mat, ord='fro') + 0.2 * np.linalg.norm(terminal.val.bias, ord=2)).sum())
        
        self.vmin = min(values)
        self.vmax = max(values)
        self.norm = matplotlib.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        
        if isinstance(self.cmap, matplotlib.colors.Colormap):
            self.color_palette = self.cmap
        else:
            self.color_palette = sns.color_palette(self.cmap, as_cmap=True)
        self.values = values
    
    def map_color(self, node: AffNode) -> Tuple:
        val = (np.linalg.norm(node.val.mat, ord='fro') + 0.2 * np.linalg.norm(node.val.bias, ord=2)).sum()
        color = self.color_palette(self.norm(val))
        return (color[0], color[1], color[2])
    
    def create_legend(self, fig, ax):
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
             ax=ax, orientation='vertical', label=self.title)


def compute_polytope_vertices(poly: Polytope) -> npt.NDArray:
    """ Calculates the extreme points (vertices) of the given 2D polytope
     and returns them in counter-clockwise order. """

    cheby_poly, cost = poly.chebyshev_center()
        
    try:
        res = cheby_poly.solve(cost)
    except ValueError as e:
        raise ValueError('cannot determine inner point of given polytope:', e)
    
    center = res[:-1]
    radius = res[-1]

    if np.isinf(center).any():
        raise ValueError('cannot determine inner point of given polytope')

    A, b = poly.to_Axbleqz()

    try:
        hs = HalfspaceIntersection(np.hstack([A, b.reshape(-1, 1)]), center)
    except QhullError as e:
        raise ValueError('cannot determine vertices of polytope, Qhull returned with error:', e)

    x, y = hs.intersections[:, 0], hs.intersections[:, 1]
    order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
    
    return hs.intersections[order].copy()


def plot_preimage_partition(dd: AffTree, ledger,
                            intervals: Sequence[Tuple[float, float]], precondition=None,
                            edge_color=None, linewidth=None,
                            projection: Callable[[Polytope], Polytope] = None,
                            fig: plt.Figure = None,
                            ax: plt.Axes = None):
    """ Plots the preimage partition of the specified AffTree. """

    if edge_color is None:
        edge_color = (0.15, 0.15, 0.33)

    if linewidth is None:
        linewidth = 0.5

    input_space = Polytope.hyperrectangle(2, intervals)
    if precondition is not None:
        input_space = input_space & precondition

    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
        ax.set_xlim(*intervals[0])
        ax.set_ylim(*intervals[1])

    ledger.create_legend(fig, ax)

    patches = []
    patch_colors = []

    for _, node, poly in dd.polyhedra():
        
        if node.is_decision():
            continue
        
        poly = input_space & poly
        if projection is not None:
            poly = projection(poly)
        
        try:
            polygon = compute_polytope_vertices(poly)
        except ValueError as e:
            print(f'Warning: got an error when determing extreme points of node {node.id}', e)
            continue
        
        patches += [Polygon(polygon)]
        
        color = ledger.map_color(node)
        patch_colors += [color]

    # plot polygons with colors
    polygon_art = PatchCollection(patches)
    polygon_art.set_facecolors(patch_colors)
    polygon_art.set_edgecolor(edge_color)
    polygon_art.set_linewidth(linewidth)
    ax.add_collection(polygon_art)

    return ax


def plot_image(dd: AffTree, ledger, intervals: Sequence[Tuple[float, float]], precondition=None,
               colors: Iterable = None, edge_color=None, linewidth=None,
               projection_in: Callable[[Polytope], Polytope] = None,
               projection_out: Callable[[npt.NDArray], npt.NDArray] = None,
               fig: plt.Figure = None,
               ax: plt.Axes = None):
          
    if edge_color is None:
        edge_color = (0.15, 0.15, 0.33)

    if linewidth is None:
        linewidth = 0.5

    input_space = Polytope.hyperrectangle(2, intervals[0:2])
    if precondition is not None:
        input_space = input_space & precondition

    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        ax.set_xlim(*intervals[0])
        ax.set_ylim(*intervals[1])
        ax.set_zlim3d(*intervals[2])

    ledger.create_legend(fig, ax)

    patches = []
    patch_colors = []

    for _, node, poly in dd.polyhedra():
        
        if node.is_decision():
            continue
        
        if node.val.outdim() > 1 and projection_out is None:
            raise ValueError("Projection needed for multi-dimensional output")
        
        poly = input_space & poly
        if projection_in is not None:
            poly = projection_in(poly)
        
        try:
            polygon = compute_polytope_vertices(poly)
        except ValueError as e:
            print(f'Warning: got an error when determing extreme points of node {node.id}', e)
            continue
        
        def z_value(p):
            val = np.array([node.val.apply(x) for x in p])
            if projection_out is not None:
                val = projection_out(val)
            return val.reshape(-1, 1)
        
        patches += [np.hstack([polygon, z_value(polygon)])]
        
        color = ledger.map_color(node)
        patch_colors += [color]

    # plot polygons with colors
    polygon_art = mplot3d.art3d.Poly3DCollection(patches)
    polygon_art.set_facecolors(patch_colors)
    polygon_art.set_edgecolor(edge_color)
    polygon_art.set_linewidth(linewidth)
    ax.add_collection(polygon_art)

    return ax
