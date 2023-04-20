import matplotlib.pyplot as plt
import os
import numpy as np
import forsys.virtual_edges as ve

def middle_point(p,q):
    return [(p.getX()+q.getX())/2,(p.getY()+q.getY())/2]

def plot_with_force_custom_fd(vertices, edges, cells, step, folder, fd, 
                        earr, versors=False, maxForce=None, 
                        minForce=None, normalized=False, mirror_y=False, **kwargs):
    plt.close()
    if not os.path.exists(folder):
        os.makedirs(folder)
    plotArrows = True
    jet = plt.get_cmap('jet')
    ## colormap = ["#77ae43", "#aec637", "#ffd200", "#fdb913", "#f26531"]
    ## colormap = ["#1D4D5F", "#14B3C8", "#A3E7DB", "#FD712D", "#EF4538"]
    # colormap = ["#0571b0", "#92c5de", "#f4a582", "#ca0020"]

    # Get external forces vectors:
    extForces = {}
    if step != "cellfit" and plotArrows:
        for name, val in fd.items():
            if isinstance(name, str):
                if int(name[1:-1]) not in extForces:
                    extForces[int(name[1:-1])] = [0, 0]
                if name[-1] == 'x':
                    extForces[int(name[1:-1])][0] = val
                else:
                    extForces[int(name[1:-1])][1] = val


    _, ax = plt.subplots(1,1)

    if not os.path.exists(folder):
        os.makedirs(folder)

    if step != "cellfit":
        # fdWithoutForce = {k: v for k, v in fd.items() if k.startswith('F')}
        fdWithoutForce = {}
        # fprime = {}
        for key, val in fd.items():
            if not isinstance(key, str):
                fdWithoutForce[key] = val
    else:
        fdWithoutForce = fd
    for vertexID, vertex in vertices.items():
        # print(vertexID, fd[edgeID])
        mFx = 0
        mFy = 0
        # plt.scatter(vertex.x,vertex.y, s=2, color="green", zorder=10)
        # origin = [vertex.x], [vertex.y]
        # ax.annotate(vertexID, xy=(vertex.getX(),vertex.getY()), size=8)
        versors = False
        if versors != False:
            try:
                for vlist in versors[vertexID]:
                    for v in vlist:
                        if v != vertexID:
                            px = vertices[v].x - vertex.x
                            py = vertices[v].y - vertex.y
                            versor = (px, py)/np.sqrt(px**2+py**2)
                            plt.arrow(vertex.x, vertex.y, versor[0],
                                        versor[1], color="red", zorder=100)
            except KeyError:
                pass

    if maxForce == None:
        maxForce = max(fdWithoutForce.values())
    if minForce == None:
        minForce = min(fdWithoutForce.values())
    
    allValues = np.array(list(fdWithoutForce.values())).flatten()
    aveForce = np.mean(allValues)
    stdForce = np.std(allValues)

    already_plotted = []

    if len(fd) > 0:
        for name, force in fd.items():
            if not isinstance(name, str):
                try:
                    current = earr[name]
                except IndexError:
                    print(f"IndexError: trying big edge {name} in an earr of length {len(earr)}")
                    pass
                # print(current)
                for j in range(0, len(current)):
                    # current[0]
                    eBelong = vertices[current[j]].ownEdges
                    for e in eBelong:
                        edge = edges[e]
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
                    
    for edge in edges.values():
        if edge.id not in already_plotted:
            plt.plot(   (edge.v1.x, edge.v2.x),
                        (edge.v1.y, edge.v2.y),
                        color="black", linewidth=0.5, alpha=0.6)


    if step != "cellfit" and plotArrows:
        for name, val in extForces.items():
            mFx = round(float(val[0]), 3)
            mFy = round(float(val[1]), 3)
            mfmodulus = np.sqrt(mFx**2 + mFy**2)
            # print(mFx, mFy)
            vToUse = vertices[int(name)]
            plt.arrow(vToUse.x, vToUse.y, mFx,
                        mFy, color=jet(abs(mfmodulus/maxForce)), zorder=100)

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


def plot_inference(frame, folder, step, pressure=False, maxForce=None, 
                        minForce=None, normalized=False, mirror_y=False, **kwargs):
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
            elif normalized == "relative":
                forceValToPlot = (edge.tension + minForce) / maxForce
            else:
                forceValToPlot = edge.tension/maxForce
            
            color = jet(forceValToPlot)
            plt.plot(   (edge.v1.x, edge.v2.x),
                        (edge.v1.y, edge.v2.y),
                        color=color, linewidth=3)
            
    if pressure:
        pressures = frame.get_pressures()["pressures"]
        pressures_cb = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=pressures.min(), vmax=pressures.max()))
        for _, cell in frame.cells.items():
            color_to_fill = pressures_cb.to_rgba(float(cell.pressure))
            all_xs = [v.x for v in cell.get_cell_vertices()]
            all_ys = [v.y for v in cell.get_cell_vertices()]
            plt.fill(all_xs, all_ys, color_to_fill, alpha=0.4)
        plt.colorbar(pressures_cb)

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


def plot_difference(frame, folder, step, **kwargs):
    plt.close()
    jet = plt.get_cmap('jet')
    _, ax = plt.subplots(1,1)

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    all_myosin = [edge.gt for edge in frame.edges.values()]
    all_myosin_max = max(all_myosin)

    # all_forces = [abs(edge.tension - edge.gt) / all_myosin_max
    #                     for edge in frame.edges.values() if edge.tension != 0]

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

def plot_force(freq, folder=''):
    plt.hist(freq, bins=25, density=True)
    plt.xlabel("Forces [ a. u ]")
    plt.ylabel("Frequency")
    plt.savefig(str(folder)+"log/forcesHist.png", dpi=500)
    plt.close()

def plot_mesh(vertices, edges, cells, name="0", folder=".", xlim=[], ylim=[], mirror_y=False):
    if not os.path.exists(folder):
        os.makedirs(folder)
    to_save = os.path.join(folder, name)
    for v in vertices.values():
        plt.scatter(v.x, v.y, s=2, color="black")
        plt.annotate(str(v.id), [v.x, v.y], fontsize=2)

    for e in edges.values():
        plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="orange", linewidth=0.5)
        plt.annotate(str(e.id), [(e.v1.x +  e.v2.x)/2 , (e.v1.y + e.v2.y)/2], fontweight="bold", fontsize=1)

    for c in cells.values():
        cm = c.get_cm()
        # print(c.id, cm)
        cxs = [v.x for v in c.vertices]
        cys = [v.y for v in c.vertices]

        plt.fill(cxs, cys, alpha=0.3)
        plt.annotate(str(c.id), [cm[0], cm[1]])
        # plt.scatter(cm[0], cm[1], marker="x", color="red")

    # # plt.show()
    if len(xlim) > 0:
        plt.xlim(xlim[0], xlim[1])
    if len(ylim) > 0:
        plt.ylim(ylim[0], ylim[1])
    # plt.axis("off")
    if mirror_y:
        plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(to_save, dpi=500)
    # plt.savefig(os.path.join(folder, str(step)+".pdf"), dpi=500)
    plt.clf()
    plt.close()

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
    # plt.savefig(os.path.join(folder, str(step) + ".pdf"), dpi=500)
    plt.clf()


def plot_time_connections(mesh, initial_time, final_time, folder=''):
    timerange = np.arange(initial_time, final_time-1, 1)
    for t in timerange:
        if mesh.mapping[t] != None:
            real_vertices_ids1 = np.array([[x[0]]+[x[-1]] for x in mesh.time_series[t].big_edges_list])
            real_vertices_ids2 = np.array([[x[0]]+[x[-1]] for x in mesh.time_series[t+1].big_edges_list])
            for v in mesh.time_series[t].vertices.values():
                if v.id in real_vertices_ids1 or v.id in mesh.time_series[t].border_vertices:
                    plt.scatter(v.x, v.y, s=5,color="black")
                    plt.annotate(str(v.id), [v.x, v.y], fontsize=4, color="black")
                    acceleration = mesh.calculate_velocity(v.id, t)

                    plt.arrow(v.x, v.y, acceleration[0], acceleration[1], color="red")
            

            for v in mesh.time_series[t+1].vertices.values():
                # if v.id in cframes[1]['external']:
                if v.id in real_vertices_ids2 or v.id in mesh.time_series[t+1].border_vertices:
                    plt.scatter(v.x, v.y, s=5, color="green")
                    plt.annotate(str(v.id), [v.x, v.y], fontsize=4, color="green")

            for e in mesh.time_series[t+1].edges.values():
                plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="green", alpha=0.2)
            for e in mesh.time_series[t].edges.values():
                plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="black", alpha=0.2)
                
            plt.savefig(os.path.join(folder, str(t)+".png"), dpi=500)
            plt.clf()

def plot_time_connections_two_times(mesh, t0, tf, fname, folder=""):
    for v0 in mesh.time_series[t0].vertices.values():
        # if v0.id in real_vertices_ids1 or v0.id in mesh.time_series[t0].border_vertices:
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



def plot_acceleration_heatmap(mesh, initial_time, final_time, folder='', name='heatmap'):
    t0 = mesh.time_series[initial_time]
    # heatmap = np.zeros((timerange, len(t0['earr'])))
    heatmap = []
    for ii in range(0, len(t0.earr)):
        row = mesh.acceleration_per_edge(ii, initial_time, final_time)
        if not np.all(np.isnan(row)):
            heatmap.append(row) 
    
    save_heatmap(heatmap, folder, initial_time, final_time, name=name)

def get_velocity_heatmap(mesh, initial_time, final_time, folder='', name='heatmap'):
    t0 = mesh.time_series[initial_time]
    # heatmap = np.zeros((timerange, len(t0['earr'])))
    heatmap = []
    for ii in range(0, len(t0.earr)):
        row = mesh.velocity_per_edge(ii, initial_time, final_time)
        if not np.all(np.isnan(row)):
            heatmap.append(row)
    return heatmap
    
    # save_heatmap(heatmap, folder, initial_time, final_time, name=name)

def save_heatmap(heatmap, folder, initial_time, final_time, name='heatmap'):
    plt.imshow(heatmap)
    plt.colorbar()
    plt.xticks(np.arange(0, final_time - initial_time, 5), np.arange(initial_time, final_time, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, str(name)+"_"+str(initial_time)+".png"), dpi=500)
    plt.clf()

def plot_ablated_edge(frames, step, bedge_index, folder):
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


def plot_residues(frame, folder, normalized=False, maxForce=None, 
                    minForce=None, mirror_y=False):
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
        # yield (vx, vy, edge.get_vertices_id(), border, edge.id)
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



        # print(f"In {[frame.vertices[vid].x, frame.vertices[vid].y]} residual vector is {force_difference}")

        plt.arrow(frame.vertices[vid].x, frame.vertices[vid].y, force_difference[0],
                    force_difference[0], color="black", zorder=100)

    plt.axis('off')
    if mirror_y:
        plt.gca().invert_yaxis()
    
    name = os.path.join(folder, str(frame.time) + ".png")
    plt.tight_layout()
    plt.savefig(name, dpi=500)
    plt.close()
    
def plot_stress_tensor(frame, folder, fname, grid=5, radius=1, **kwargs):
    # sigmas, bins_centers
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

    # _, _, bin_edges = frame.stress_tensor
    
    # for xline in bin_edges[0]:
    #     plt.axvline(xline, ls="--", color="gray", alpha=0.3)
    # for yline in bin_edges[1]:
    #     plt.axhline(yline, ls="--", color="gray", alpha=0.3)


    for positions, eigensystem in frame.principal_stress.items():
        rescaled_eigenvalues = eigensystem[0] / np.linalg.norm(eigensystem[0])
        component1 = scale_factor * rescaled_eigenvalues[0] * eigensystem[1][:, 0]
        component2 = scale_factor * rescaled_eigenvalues[1] * eigensystem[1][:, 1]
        
        axis_1_x = [positions[0] - component1[0], positions[0], positions[0] + component1[0]]
        axis_1_y = [positions[1] - component1[1], positions[1], positions[1] + component1[1]]
        axis_2_x = [positions[0] - component2[0], positions[0], positions[0] + component2[0]]
        axis_2_y = [positions[1] - component2[1], positions[1], positions[1] + component2[1]]

        plt.plot(axis_1_x, axis_1_y, color="green", linewidth=2)
        plt.plot(axis_2_x, axis_2_y, color="green", linewidth=2)



        # points_y = [positions[1], positions[1], positions[1] + component1[0]]


        # plt.arrow(positions[0], positions[1], component1[0], component1[1], color="green")
        # plt.arrow(positions[0], positions[1], component2[0], component2[1], color="green")


    # plt.axis('off')
    
    if kwargs.get("mirror_y", None):
        plt.gca().invert_yaxis()
    

    name = os.path.join(folder, str(fname))
    plt.tight_layout()
    plt.savefig(name, dpi=500)
    plt.close()
