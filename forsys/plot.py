import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import forsys as fs

import forsys.virtual_edges as ve
import forsys.frames as fframes
import forsys.edge as fedges
import forsys.time_series as ftimes
import forsys.myosin as fmyo
from typing import Union, Tuple
from copy import deepcopy
import PIL
import tifffile as tif


def plot_with_stress_custom(frame: fframes.Frame, step: str, folder: str, stress_dictionary: dict, 
                            earr: list, versors: Union[bool, list]=False, maxForce: float=None, 
                            minForce: float=None, normalized: Union[bool, str]=False, mirror_y: bool=False, **kwargs) -> None:
    """Generate a plot on a given mesh using a custom force dictionary.

   :param frame: Frame to plot
    :type frame: fframes.Frame
    :param step: Step at which to create the plot. It is used as the file name
    :type step: int
    :param folder: Folder to save the plot to
    :type folder: str
    :param stress_dictionary: Custom dictionary of stresses
    :type stress_dictionary: dict
    :param earr: List of big edges in the system
    :type earr: list
    :param versors: Versors to plot if necessary. Useful when external stresses are used, defaults to False
    :type versors: Union[bool, list], optional
    :param maxForce: Max force to use in the normalization, defaults to None
    :type maxForce: float, optional
    :param minForce: Min force in the normalization, defaults to None
    :type minForce: float, optional
    :param normalized: "max" normalizes by the maximum force; "normal" normalizes to a \
        normal distribution centered in zero, defaults to False
    :type normalized: Union[bool, str], optional
    :param mirror_y: Plot mirroring the y component, defaults to False
    :type mirror_y: bool, optional
    """
    plt.close()
    if not os.path.exists(folder):
        os.makedirs(folder)
    jet = plt.get_cmap('jet')

    _, ax = plt.subplots(1,1)

    if not os.path.exists(folder):
        os.makedirs(folder)

    if maxForce == None:
        maxForce = max(stress_dictionary.values())
    if minForce == None:
        minForce = min(stress_dictionary.values())
    
    allValues = np.array(list(stress_dictionary.values())).flatten()
    aveForce = np.mean(allValues)
    stdForce = np.std(allValues)

    already_plotted = []

    for name, force in stress_dictionary.items():
        if not isinstance(name, str):
            try:
                current = earr[name]
            except IndexError:
                print(f"IndexError: trying big edge {name} in an earr of length {len(earr)}")
                pass
            for j in range(0, len(current)):
                eBelong = frame.vertices[current[j]].ownEdges
                for e in eBelong:
                    edge = frame.edges[e]
                    try:
                        if len(set(edge.get_vertices_id()) & set([current[j], current[j+1]])) == 2:
                            if normalized == "normal":
                                forceValToPlot = (force - aveForce) / stdForce
                            elif normalized == "max":
                                forceValToPlot = force / maxForce
                            else:
                                forceValToPlot = force/maxForce

                            color = jet(forceValToPlot)
                            plt.plot(   (edge.v1.x, edge.v2.x),
                                        (edge.v1.y, edge.v2.y),
                                        color=color, linewidth=3)
                            already_plotted.append(edge.id)
                    except IndexError:
                        pass
                    
    for edge in frame.edges.values():
        if edge.id not in already_plotted:
            plt.plot(   (edge.v1.x, edge.v2.x),
                        (edge.v1.y, edge.v2.y),
                        color="black", linewidth=0.5, alpha=0.6)

    plt.axis('off')
    if mirror_y:
        plt.gca().invert_yaxis()
    if kwargs.get("colorbar", False):
        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm)

    name = os.path.join(folder, str(step) + ".png")

    plt.tight_layout()
    plt.savefig(name, dpi=500)

    plt.close()


def plot_inference(frame: fframes.Frame, pressure: bool = False,
                   ground_truth: bool = False, maxForce: float = None,
                   minForce: float = None,
                   normalized: Union[bool, str] = False,
                   ax: mpl.axes.Axes = None,
                   **kwargs) -> Tuple:
    """Generate a plot of a tissue using the inferred tensions and pressures and return the plot object

    :param frame: Frame to plot
    :type frame: fframes.Frame
    :param pressure: If True pressures are plotted in the inner space of the cells.
    :type pressure: bool
    :param ground_truth: False plots inferred stresses, True will plot the ground truth value assigned to each edge
    :param maxForce: Max force to use in the normalization, defaults to None
    :type maxForce: float, optional
    :param minForce: Min force in the normalization, defaults to None
    :type minForce: float, optional
    :param normalized: "max" normalizes by the maximum force; "normal" normalizes to a \
        normal distribution centered in zero, defaults to False
    :type normalized: Union[bool, str], optional
    :return: Matplotlib plot object
    :rtype: Tuple
    """
    if not ax:
        _, ax = plt.subplots(1, 1)
    if ground_truth:
        perturbation = 0.0
        all_forces = [edge.gt + np.random.uniform(-edge.gt*perturbation, edge.gt*perturbation) for edge in frame.edges.values() if edge.gt != 0]
    else:
        all_forces = [edge.tension for edge in frame.edges.values() if edge.tension != 0]

    if maxForce is None:
        maxForce = max(all_forces)
    if minForce is None:
        minForce = min(all_forces)

    aveForce = np.mean(all_forces)
    stdForce = np.std(all_forces)

    for _, edge in frame.edges.items():
        if edge.tension != 0:
            observable = edge.tension if not ground_truth else edge.gt
        else:
            observable = edge.tension if not ground_truth else edge.gt
        if observable <= 0:
            ax.plot((edge.v1.x, edge.v2.x),
                    (edge.v1.y, edge.v2.y),
                    color="black", linewidth=0.5, alpha=0.6)
        else:
            if normalized == "normal":
                forceValToPlot = (observable - aveForce) / stdForce
            elif normalized == "max":
                forceValToPlot = observable / maxForce
                min_stress = 0
                max_stress = 1
            elif normalized == "relative":
                forceValToPlot = (observable + minForce) / maxForce
            elif normalized == "absolute":
                forceValToPlot = observable / kwargs.get("max_stress", 2)
                min_stress = 0
                max_stress = 2
            else:
                forceValToPlot = observable/maxForce
                min_stress = 0
                max_stress = 1

            cmap_stress_name = kwargs.get("cmap_stress", "magma")
            cmap_pressure_name = kwargs.get("cmap_pressure", "cividis")
            cmap_stress = plt.get_cmap(cmap_stress_name)
            # cmap_pressure = plt.get_cmap(cmap_pressure_name)

            color_stress = cmap_stress(forceValToPlot)
            stress_cb = plt.cm.ScalarMappable(cmap=cmap_stress_name,
                                              norm=plt.Normalize(vmin=min_stress,
                                                                 vmax=max_stress))
            ax.plot((edge.v1.x, edge.v2.x),
                    (edge.v1.y, edge.v2.y),
                    color=color_stress, linewidth=2)

    if pressure:
        pressures = frame.get_pressures()["pressure"]
        pressures_cb = plt.cm.ScalarMappable(cmap=cmap_pressure_name,
                                             norm=plt.Normalize(vmin=pressures.min(),
                                                                vmax=pressures.max()))
        for _, cell in frame.cells.items():
            color_to_fill = pressures_cb.to_rgba(float(cell.pressure))
            color_to_fill = mpl.colors.to_hex(color_to_fill)
            all_xs = [v.x for v in cell.get_cell_vertices()]
            all_ys = [v.y for v in cell.get_cell_vertices()]
            ax.fill(all_xs, all_ys, color_to_fill, alpha=0.4)

    plt.axis('off')
    if kwargs.get("mirror_y", False):
        plt.gca().invert_yaxis()
    if kwargs.get("mirror_x", False):
        plt.gca().invert_xaxis()

    if kwargs.get("colorbar", False):
        cb_stress = plt.colorbar(stress_cb, ax=ax)
        cb_stress.set_label("Stress [a.u.]")
        cb_stress.set_ticks([])
        if pressure:
            cb_pressure = plt.colorbar(pressures_cb, ax=ax)
            cb_pressure.set_label("Pressure [a.u.]")
            cb_pressure.set_ticks([])

    if kwargs.get("aspect_ratio", None):
        plt.gca().set_aspect(kwargs.get("aspect_ratio"))
    plt.tight_layout()
    return _, ax


def plot_difference(frame, folder: str, step: str, **kwargs) -> None:
    """Plot the difference between ground truth and inferred values in a heatmap.
    Both the inferred value and the ground truth are required

    :param frame: Frame to use
    :type frame: fframes.Frame
    :param folder: Folder to save plot
    :type folder: str
    :param step: File name of the plot
    :type step: str
    """
    plt.close()
    jet = plt.get_cmap('jet')
    _, ax = plt.subplots(1, 1)

    if not os.path.exists(folder):
        os.makedirs(folder)

    all_myosin = [edge.gt for edge in frame.edges.values()]
    all_myosin_max = max(all_myosin)

    for _, edge in frame.edges.items():
        difference_to_plot = abs(edge.tension - edge.gt) / all_myosin_max
        if difference_to_plot == 0:
            plt.plot(   (edge.v1.x, edge.v2.x),
                        (edge.v1.y, edge.v2.y),
                        color="black", linewidth=0.5, alpha=0.6)      

        else:
            color = jet(difference_to_plot)
            plt.plot(   (edge.v1.x, edge.v2.x),
                        (edge.v1.y, edge.v2.y),
                        color=color, linewidth=3)
    plt.axis('off')
    if kwargs.get("mirror_y", False):
        plt.gca().invert_yaxis()

    if kwargs.get("colorbar", False):
        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
        plt.colorbar(sm)

    name = os.path.join(folder, str(step) + ".png")
    plt.tight_layout()
    plt.savefig(name, dpi=500)
    plt.close()

def plot_force(freq: list, folder: str ='') -> None:
    """Plot histogram of the stresses.

    :param freq: List of stress values
    :type freq: list
    :param folder: Folder to save the log, defaults to ''
    :type folder: str, optional
    """
    plt.hist(freq, bins=25, density=True)
    plt.xlabel("Forces [ a. u ]")
    plt.ylabel("Frequency")
    plt.savefig(str(folder)+"log/forcesHist.png", dpi=500)
    plt.close()

def plot_mesh(frame: fframes.Frame,
              ax: mpl.axes.Axes = None,
              **kwargs) -> Tuple:
            #   xlim: list=[], ylim: list=[], 
            #   mirror_y: bool = False,
            #   mirror_x: bool = False) -> None:
    """Generate a plot of the mesh. Useful for visualizing IDs of cells, edges and vertices in the system.

    :param frame: Frame to plot
    :type frame: fframes.Frame
    :param xlim: Array of [x_min, x_max] to "zoom" in, defaults to []
    :type xlim: list, optional
    :param ylim: Array of [y_min, y_max] to "zoom" in, defaults to []
    :type ylim: list, optional
    :param mirror_y: If True the tissue is plotted as a mirror image in the Y axis, defaults to False
    :type mirror_y: bool, optional
    """
    if not ax:
        _, ax = plt.subplots(1, 1)
    if kwargs.get("plot_vertices", False) and not kwargs.get("plot_tjs", False):
        for v in frame.vertices.values():
            plt.scatter(v.x, v.y, s=2, color="black")
            if kwargs.get("plot_vertices_id", False):
                plt.annotate(str(v.id), [v.x, v.y], fontsize=2)

    if kwargs.get("plot_tjs", False):
        for v in frame.vertices.values():
            if len(v.ownEdges) >= 3:
                plt.scatter(v.x, v.y, s=2, color="black")
                if kwargs.get("plot_vertices_id", False):
                    plt.annotate(str(v.id), [v.x, v.y], fontsize=2)
    
    if kwargs.get("plot_versors", False):
        for v in frame.vertices.values():
            if len(v.ownEdges) >= 3:
                vertex_big_edges = [frame.big_edges[beid] for beid in v.own_big_edges]
                vertex_big_edges_versors = [big_edge.get_vector_from_vertex(v.id,
                                                                            fit_method="dlite") for big_edge in vertex_big_edges]
                # for versor in ve.get_versors(frame.vertices, frame.edges, v.id):
                for vector in vertex_big_edges_versors:
                    size = np.linalg.norm(vector)
                    versor = vector / size
                    plt.arrow(v.x,
                              v.y,
                              versor[0] * 10,
                              versor[1] * 10,
                              color="green",
                              width=4,
                              alpha=0.7,
                              zorder=1000)

    for e in frame.edges.values():
        if kwargs.get("plot_edges", False):
            plt.plot([e.v1.x, e.v2.x],
                     [e.v1.y, e.v2.y],
                     color="black",
                     linewidth=0.5,
                     alpha=0.6)

            if kwargs.get("plot_edges_id", False):
                plt.annotate(str(e.id), [(e.v1.x +  e.v2.x)/2 , (e.v1.y + e.v2.y)/2], fontweight="bold", fontsize=1)

    for c in frame.cells.values():
        cm = c.get_cm()
        cxs = [v.x for v in c.vertices]
        cys = [v.y for v in c.vertices]

        color_palette = kwargs.get("color_palette", "gray")
        if type(color_palette) is str:
            cell_colors = {k_id: color_palette for k_id in frame.cells.keys()}
        elif type(color_palette) is dict:
            cell_colors = color_palette
        else:
            raise ValueError("Color palette must be a string or a dictionary")
        # TODO add that if the id is not in the dictionary, it is colored in gray
        try:
            plt.fill(cxs, cys, alpha=0.25, color=cell_colors[c.id])
        except KeyError:
            plt.fill(cxs, cys, alpha=0.25, color="gray")
        if kwargs.get("plot_cells_id", False):
            plt.annotate(str(c.id), [cm[0], cm[1]], fontsize=5)

    xlim = kwargs.get("xlim", [])
    ylim = kwargs.get("ylim", [])
    if len(xlim) > 0:
        plt.xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        plt.ylim(ylim[0], ylim[1])
    if kwargs.get("mirror_y", False):
        plt.gca().invert_yaxis()
    if kwargs.get("mirror_x", False):
        plt.gca().invert_xaxis()

    plt.axis("off")
    return ax


def plot_equilibrium(mesh, step, folder, what="acceleration", normalized=False, cutoff=None):
    # According to how much acceleration is present, paint the tissue
    jet = plt.get_cmap('jet')

    # xcoords = [v.x for v in vertices.values()]
    # ycoords = [v.y for v in vertices.values()]

    # for v in vertices.values():
    #     plt.scatter(v.x, v.y, s=5,color="black")
        # plt.annotate(str(v.id), [v.x, v.y], fontsize=3)

    # for ev in mesh.mesh[step]['earr']:
    #     # calculate acceleration in both vertices of the edge
    #     acc0 = mesh.calculate_acceleration(ev[0], step)
    #     acc1 = mesh.calculate_acceleration(ev[-1], step)
    #     accelerations[mesh.mesh[step]['earr'].index(ev)] = np.linalg.norm(acc0) + np.linalg.norm(acc1)
    accelerations = mesh.whole_tissue_acceleration(step)
    maxAccel = np.nanmax(list(accelerations.values()))
    for ev in mesh.time_series[step].earr:
        for j in range(0, len(ev)):
            eBelong = mesh.time_series[step].vertices[ev[j]].ownEdges
            for eid in eBelong:
                edge = mesh.time_series[step].edges[eid]
                try:
                    if len(set(edge.get_vertices_id()) & set([ev[j], ev[j+1]])) == 2:
                        acceleration = accelerations[mesh.time_series[step].earr.index(ev)]
                        if normalized == "max":
                            color = jet(acceleration/maxAccel)
                        elif normalized == "cutoff":
                            color = "blue" if acceleration < cutoff else "red"
                        else:
                            color = jet(accelerations[mesh.time_series[step].earr.index(ev)])
                        
                        if np.isnan(acceleration):
                             plt.plot([edge.v1.x, edge.v2.x], [edge.v1.y, edge.v2.y], 
                                    color="gray", alpha=0.5, 
                                    linewidth=3)
                        else:
                            plt.plot([edge.v1.x, edge.v2.x], [edge.v1.y, edge.v2.y], 
                                    color=color, 
                                    linewidth=3)
                except IndexError:
                    pass
    
    if normalized == "max":
        vmin = 0
        vmax = maxAccel
    else:
        vmin = 0
        vmax = 1
    
    if normalized != "cutoff":
        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        plt.colorbar(sm)
    plt.tight_layout()

    # plt.xlim()

    plt.savefig(os.path.join(folder, str(step) + ".png"), dpi=500)
    plt.clf()


def plot_time_connections(mesh: ftimes.TimeSeries, initial_time: int, final_time: int, folder: str='') -> None:
    """Plot time connections among vertices in all the frames between initial and final time

    :param mesh: TimeSeries object with mapping information
    :type mesh: ftimes.TimeSeries
    :param initial_time: Initial time
    :type initial_time: int
    :param final_time: Final time
    :type final_time: int
    :param folder: Path to save the plot, defaults to ''
    :type folder: str, optional
    """
    timerange = np.arange(initial_time, final_time-1, 1)
    for t in timerange:
        if mesh.mapping[t] != None:
            real_vertices_ids1 = np.array([[x[0]]+[x[-1]] for x in mesh.time_series[t].big_edges_list])
            real_vertices_ids2 = np.array([[x[0]]+[x[-1]] for x in mesh.time_series[t+1].big_edges_list])
            for v in mesh.time_series[t].vertices.values():
                if v.id in real_vertices_ids1 or v.id in mesh.time_series[t].border_vertices:
                    plt.scatter(v.x, v.y, s=5,color="black")
                    plt.annotate(str(v.id), [v.x, v.y], fontsize=4, color="black")
                    acceleration = mesh.calculate_displacement(v.id, t)

                    plt.arrow(v.x,
                              v.y, 
                              acceleration[0] * 1,
                              acceleration[1] * 1, color="red")
            

            for v in mesh.time_series[t+1].vertices.values():
                if v.id in real_vertices_ids2 or v.id in mesh.time_series[t+1].border_vertices:
                    plt.scatter(v.x, v.y, s=5, color="green")
                    plt.annotate(str(v.id), [v.x, v.y], fontsize=4, color="green")

            for e in mesh.time_series[t+1].edges.values():
                plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="green", alpha=0.2)
            for e in mesh.time_series[t].edges.values():
                plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="black", alpha=0.2)

            plt.savefig(os.path.join(folder, str(t)+".png"), dpi=500)
            plt.clf()

def plot_time_connections_two_times(mesh: ftimes.TimeSeries, t0: int, tf: int, 
                                    fname: str, folder: str="") -> None:
    """Create a connection plot among vertices, between only two timepoints

    :param mesh: TimeSeries object with mapping information
    :type mesh: ftimes.TimeSeries
    :param t0: initial time
    :type t0: int
    :param tf: Final time
    :type tf: int
    :param fname: File name to use
    :type fname: str
    :param folder: Folder to save the plot to, defaults to ""
    :type folder: str, optional
    """
    for v0 in mesh.time_series[t0].vertices.values():
        if True:
            v1_id = mesh.get_point_id_by_map(v0.id, t0, tf)
            v1 = mesh.time_series[tf].vertices[v1_id]

            plt.scatter(v0.x, v0.y, s=5, color="black")
            plt.scatter(v1.x, v1.y, s=5, color="green")

            deltax = v1.x - v0.x
            deltay = v1.y - v0.y

            plt.arrow(v0.x, v0.y, deltax, deltay, color="red")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, str(fname)+".pdf"), dpi=500)
    plt.clf()



def plot_acceleration_heatmap(mesh: ftimes.TimeSeries, initial_time: int,
                              final_time: int, folder: str='', name: str='heatmap') -> None:
    """Generate an acceleration heatmap for each edge.

    :param mesh: TimeSeries object with mapping information
    :type mesh: ftimes.TimeSeries
    :param initial_time: Initial time
    :type initial_time: int
    :param final_time: Final time
    :type final_time: int
    :param folder: Folder to save plot to, defaults to ''
    :type folder: str, optional
    :param name: Filename for the plot, defaults to 'heatmap'
    :type name: str, optional
    """
    t0 = mesh.time_series[initial_time]
    heatmap = []
    for ii in range(0, len(t0.earr)):
        row = mesh.acceleration_per_edge(ii, initial_time, final_time)
        if not np.all(np.isnan(row)):
            heatmap.append(row) 
    
    save_heatmap(heatmap, folder, initial_time, final_time, name=name)

def get_velocity_heatmap(mesh: ftimes.TimeSeries, initial_time: int, final_time: int) -> list:
    """Generate an array with the velocity of each edge tracked through time.

    :param mesh: TimeSeries object with mapping information
    :type mesh: ftimes.TimeSeries
    :param initial_time: Initial time
    :type initial_time: int
    :param final_time: Final time
    :type final_time: int
    :return: Array where each row is the velocity of an edge and each column a different time
    :rtype: list
    """
    t0 = mesh.time_series[initial_time]
    heatmap = []
    for ii in range(0, len(t0.earr)):
        row = mesh.velocity_per_edge(ii, initial_time, final_time)
        if not np.all(np.isnan(row)):
            heatmap.append(row)
    return heatmap

def save_heatmap(heatmap: list, folder: str, initial_time: int, final_time:int, name: str='heatmap') -> None:
    """Save a given heatmap

    :param heatmap: Heatmap array
    :type heatmap: list
    :param folder: Folder to save the plot to
    :type folder: str
    :param initial_time: Initial time
    :type initial_time: int
    :param final_time: Final time
    :type final_time: int
    :param name: Name of the plot, defaults to 'heatmap'
    :type name: str, optional
    """
    plt.imshow(heatmap)
    plt.colorbar()
    plt.xticks(np.arange(0, final_time - initial_time, 5), np.arange(initial_time, final_time, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, str(name)+"_"+str(initial_time)+".png"), dpi=500)
    plt.clf()

def plot_ablated_edge(frames: dict, step: int, bedge_index: int, folder: str) -> None:
    """Plot the mesh of the system with the ablated edges in the tissue in a different color.

    :param frames: Dictionary with all the frames
    :type frames: dict
    :param step: Step to plot. Must correspond with a time in the frames dictionary
    :type step: int
    :param bedge_index: Index of the ablated big edge
    :type bedge_index: int
    :param folder: Folder to save the mesh
    :type folder: str
    """
    # TODO: integrate with  plot_mesh() as an option
    for v in frames[step].vertices.values():
        plt.scatter(v.x, v.y, s=2, color="black")
        plt.annotate(str(v.id), [v.x, v.y], fontsize=2)

    for e in frames[step].edges.values():
        plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="orange", linewidth=0.5)

    for v in frames[step].earr[bedge_index]:
        for eid in frames[step].vertices[v].ownEdges:
            e = frames[step].edges[eid]
            plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="green", linewidth=1.5)

    for c in frames[step].cells.values():
        cm = c.get_cm()
        cxs = [v.x for v in c.vertices]
        cys = [v.y for v in c.vertices]

        plt.fill(cxs, cys, alpha=0.3)
        plt.annotate(str(c.id), [cm[0], cm[1]], fontsize=5)

    plt.savefig(folder, dpi=600)
    plt.close()


def plot_residues(frame: fframes.Frame, folder: str, normalized: bool=False, maxForce:float=None, 
                    minForce: float=None, mirror_y: bool=False) -> None:
    """Plot differences between inferred values and ground truths as residue
    vectors from each pivot vertex

    :param frame: Frame with all infromation
    :type frame: fframes.Frame
    :param folder: Folder to save the plot
    :type folder: str
    :param normalized: "max" normalizes by the maximum force; "normal" normalizes to a \
        normal distribution centered in zero, defaults to False
    :type normalized: bool, optional
    :param maxForce:  Maximum force to use in the normalization , defaults to None
    :type maxForce: float, optional
    :param minForce:  Minimum force to use in the normalization, defaults to None
    :type minForce: float, optional
    :param mirror_y: Plot mirroring the y component, defaults to False
    :type mirror_y: bool, optional
    """
    plt.close()
    jet = plt.get_cmap('jet')
    _, ax = plt.subplots(1,1)

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    all_forces = [edge.tension for edge in frame.edges.values() if edge.tension != 0]

    if maxForce == None:
        maxForce = max(all_forces)
    if minForce == None:
        minForce = min(all_forces)

    allValues = all_forces
    aveForce = np.mean(allValues)
    stdForce = np.std(allValues)

    for _, edge in frame.edges.items():
        if edge.tension == 0:
            plt.plot(   (edge.v1.x, edge.v2.x),
                        (edge.v1.y, edge.v2.y),
                        color="black", linewidth=0.5, alpha=0.6)
        else:
            if normalized == "normal":
                forceValToPlot = (edge.tension - aveForce) / stdForce
            elif normalized == "max":
                forceValToPlot = edge.tension / maxForce
            else:
                forceValToPlot = edge.tension/maxForce
            
            color = jet(forceValToPlot)
            plt.plot(   (edge.v1.x, edge.v2.x),
                        (edge.v1.y, edge.v2.y),
                        color=color, linewidth=3)
    
    tj_vertices = list(set([big_edge[0] for big_edge in frame.big_edges_list]))

    for vid in tj_vertices:
        force_difference = np.array([0, 0]).astype(np.float64)
        for versor in ve.get_versors(frame.vertices, frame.edges, vid):
            try:
                calculated_force_vector = [frame.edges[versor[4]].tension * versor[0], 
                                            frame.edges[versor[4]].tension * versor[1]]
                original_force_vector = [frame.edges[versor[4]].gt * versor[0],
                                        frame.edges[versor[4]].gt * versor[1]]
            except KeyError:
                raise("Missing tension or ground truth")
            
            force_difference += np.array(calculated_force_vector) - \
                                np.array(original_force_vector)

        plt.arrow(frame.vertices[vid].x, frame.vertices[vid].y, force_difference[0],
                    force_difference[0], color="black", zorder=100)

    plt.axis('off')
    if mirror_y:
        plt.gca().invert_yaxis()
    
    name = os.path.join(folder, str(frame.time) + ".png")
    plt.tight_layout()
    plt.savefig(name, dpi=500)
    plt.close()
    
def plot_stress_tensor(frame: fframes.Frame, folder: str, fname: str, 
                       grid: int=5, radius: float=1, **kwargs) -> None:
    """Plot the principal components of the stress tensor on top of a 
    layout of the mesh

    :param frame: Frame object
    :type frame: fframes.Frame
    :param folder: Folder to save the output plot
    :type folder: str
    :param fname: Name for the plot
    :type fname: str
    :param grid: Number of grid points in each axis, defaults to 5
    :type grid: int, optional
    :param radius: Number of average cell radii to use in the integration, defaults to 1
    :type radius: float, optional
    """
    frame.calculate_stress_tensor(grid, radius)
    plt.close()
    _, ax = plt.subplots(1,1)

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for _, edge in frame.edges.items():
        plt.plot(   (edge.v1.x, edge.v2.x),
                    (edge.v1.y, edge.v2.y),
                    color="black", linewidth=0.5, alpha=0.6)

    external_scale = kwargs.get("tensor_scale", 1)
    try:
        scale_factor = (external_scale/10) * min(frame.stress_tensor[1][0][0] - frame.stress_tensor[1][0][1],
                                                frame.stress_tensor[1][1][0] - frame.stress_tensor[1][1][1])
    except IndexError:
        scale_factor = (external_scale/10) * 0.1 * min(frame.stress_tensor[1][0][0],
                                                       frame.stress_tensor[1][1][0] )

    for positions, eigensystem in frame.principal_stress.items():
        eigen_norm = np.linalg.norm(eigensystem[0])
        if eigen_norm == 0:
            rescaled_eigenvalues = [0, 0]
        else:
            rescaled_eigenvalues = eigensystem[0] / eigen_norm
        component1 = scale_factor * rescaled_eigenvalues[0] * eigensystem[1][:, 0]
        component2 = scale_factor * rescaled_eigenvalues[1] * eigensystem[1][:, 1]

        axis_1_x = [positions[0] - component1[0], positions[0], positions[0] + component1[0]]
        axis_1_y = [positions[1] - component1[1], positions[1], positions[1] + component1[1]]
        axis_2_x = [positions[0] - component2[0], positions[0], positions[0] + component2[0]]
        axis_2_y = [positions[1] - component2[1], positions[1], positions[1] + component2[1]]

        plt.plot(axis_1_x, axis_1_y, color="green", linewidth=2)
        plt.plot(axis_2_x, axis_2_y, color="green", linewidth=2)

    plt.axis('off')

    if kwargs.get("mirror_y", None):
        plt.gca().invert_yaxis()


    name = os.path.join(folder, str(fname))
    plt.tight_layout()
    plt.savefig(name, dpi=500)
    plt.close()


def plot_skeleton(frame: fframes.Frame, save_folder: str, **kwargs) -> None:

    maximize = kwargs.get("maximize", False)
    if maximize:
        factor_x, factor_y = maximize
    else:
        factor_x, factor_y = 1, 1

    new_frame = deepcopy(frame)
    for _, vertex in new_frame.vertices.items():
        vertex.x = vertex.x * factor_x
        vertex.y = vertex.y * factor_y

    for beid, big_edge in new_frame.big_edges.items():
        new_frame.big_edges[beid] = fedges.BigEdge(beid,
                                                   big_edge.vertices)
    frame = new_frame

    if kwargs.get("use_all", False):
        edges_to_use = list(frame.big_edges.values())
    else:
        edges_to_use = frame.internal_big_edges

    all_vertices = set()
    for _, big_edge in enumerate(edges_to_use):
        vertices, _ = fmyo.get_interpolation(big_edge, layers=0)
        all_vertices.update(vertices)

    all_vertices_tuple = list(zip(*all_vertices))

    all_xs = all_vertices_tuple[0]
    all_ys = all_vertices_tuple[1]
    max_x, min_x = np.max(all_xs), np.min(all_xs)
    max_y, min_y = np.max(all_ys), np.min(all_ys)

    size_x = int(max_x - min_x)
    size_y = int(max_y - min_y)

    image_array = np.zeros((size_y + 10, size_x + 10))

    for vx, vy in list(all_vertices):
        vx = int((vx - min_x)) + 5
        vy = int((vy - min_y)) + 5

        image_array[int(vy), int(vx)] = 255
    im = PIL.Image.fromarray(image_array)
    im.save(save_folder)


def plot_inference_as_tiff(forsys: fs.ForSys,
                           image_sizes: list,
                           save_folder: str,
                           **kwargs) -> None:
    """Generate a plot of a tissue using the inferred tensions and pressures
        and save it as a tiff. If multiple more than one frame exists, the
        image is saved as a stack

    :param forsys: ForSys object with all the frames already solved
    :type frame: fs.ForSys
    :param pressure: If True pressures are plotted in the inner space of the cells.
    :type pressure: bool
    """
    stack = []
    original_images = kwargs.get("original_images", [])
    for time, frame in forsys.frames.items():
        original_size = image_sizes[time]
        image_array = np.zeros((original_size[1], original_size[0]))

        if kwargs.get("use_all", True):
            edges_to_use = list(frame.big_edges.values())
        else:
            edges_to_use = frame.internal_big_edges

        # from 0 to 255
        max_tension = np.max([big_edge.tension for big_edge in edges_to_use])

        for _, big_edge in enumerate(edges_to_use):
            vertices, _ = fmyo.get_interpolation(big_edge,
                                                 layers=kwargs.get("layers",
                                                                   0))
            for vx, vy in vertices:
                if big_edge.external:
                    image_array[int(vy), int(vx)] = kwargs.get("external_color", 255)
                else:
                    image_array[int(vy), int(vx)] = (big_edge.tension / max_tension) * 255
        if original_images:
            original_image = original_images[time]
            image_array_original = np.array(original_image)
            complete_frame = np.array([image_array_original, image_array])
        else:
            complete_frame = np.array([image_array])
        stack.append(complete_frame)

    stack = np.array(stack).astype(np.uint8)
    print(f"Complete size: {stack.shape}")
    tif.imwrite(save_folder,
                stack,
                imagej=True,
                # TODO: add resolution from original image
                # resolution=(1, 1),
                metadata={
                        "mode": "composite",
                        "axes": "TCYX",
                        "Channel": {"Name": ["Microscopy", "Stress"]}
                },
                )


def plot_big_edges(frame, **kwargs) -> Tuple:
    fig, ax = plt.subplots(1, 1)

    for v in frame.vertices.values():
        if len(v.ownEdges) > 2:
            plt.scatter(v.x, v.y, s=2, color="black")
            plt.annotate(str(v.id), [v.x, v.y], fontsize=2)

    for e in frame.edges.values():
        plt.plot([e.v1.x, e.v2.x],
                 [e.v1.y, e.v2.y], color="orange", linewidth=0.5)

    for be in frame.big_edges.values():
        x_cm = np.mean(be.xs)
        y_cm = np.mean(be.ys)
        plt.annotate(str(be.big_edge_id), [x_cm , y_cm], fontweight="bold", fontsize=5, color="blue", alpha=0.4)

    for c in frame.cells.values():
        cm = c.get_cm()
        cxs = [v.x for v in c.vertices]
        cys = [v.y for v in c.vertices]

        plt.fill(cxs, cys, alpha=0.3)
        plt.annotate(str(c.id), [cm[0], cm[1]])

    kwarg_xlim = kwargs.get("xlim", [])
    kwarg_ylim = kwargs.get("ylim", [])

    if len(kwarg_xlim) > 0:
        plt.xlim(kwarg_xlim[0], kwarg_xlim[1])
    if len(kwarg_ylim) > 0:
        plt.ylim(kwarg_ylim[0], kwarg_ylim[1])
    if kwargs.get("mirror_y", False):
        plt.gca().invert_yaxis()
    if kwargs.get("mirror_x", False):
        plt.gca().invert_xaxis()

    return fig, ax
