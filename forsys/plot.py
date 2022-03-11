import matplotlib.pyplot as plt
import os
import numpy as np
# import matplotlib.cm as cm
import matplotlib as mpl
# import sysvert.forces as forces
from  forsys.exceptions import DifferentTissueException


# import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

def middle_point(p,q):
    return [(p.getX()+q.getX())/2,(p.getY()+q.getY())/2]

def plot_with_force(vertices, edges, cells, step, folder, fd, 
                        earr, versors=False, maxForce=None, 
                        minForce=None, normalized=False):
    plt.close()
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
    # step = str(step).zfill(len(str(nSteps)))
    # if not os.path.exists(folder + '/plots'):
    #     os.makedirs(folder + '/plots')
    # name = folder + "/plots" + "/step_" + step + ".png"
    # name = folder + "/step_" + step + ".png"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # eTotal = len(edges.items())
    # for edgeID,edge in edges.items():
        # v1 = vertices[edge.get_vertices_array()[0]]
        # v2 = vertices[edge.get_vertices_array()[1]]
        # plt.plot((edge.v1.x, edge.v2.x),(edge.v1.y, edge.v2.y),c="orange", linewidth=0.5, alpha=0.5)
       # ax.annotate(edgeID, xy=(middle_point(v1, v2)[0],middle_point(v1, v2)[1]), size=5)
       # plt.plot(edge.getLineX(),edge.getLineY(),c="#5E300B")

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

    if len(fd) > 0:
        # maxForce = max(fdWithoutForce.values())
        # counter = 0
        for name, force in fd.items():
            if not isinstance(name, str):
                # Draw edge in each element of big edge
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
                                # print("Edge is ", e, " with ", edge.getVerticesBelonging() )
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
                                # print("Current force is ", force, " maxforce is ", maxForce, " value is ", abs(force/maxForce))
                        except IndexError:
                            pass

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
    ax.set_aspect('equal')  # VER SI HACE FALTA
    # sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize())
    plt.colorbar(sm)
    name = os.path.join(folder, str(step) + ".png")
    # name = folder + step + ".png"
    plt.savefig(name, dpi=500)
    plt.close()
    # print("Max force used: ", maxForce)
    # Now the histogram
    # plot_force(fd.values(), folder)

def plot_force(freq, folder=''):
    plt.hist(freq, bins=25, density=True)
    plt.xlabel("Forces [ a. u ]")
    plt.ylabel("Frequency")
    plt.savefig(str(folder)+"log/forcesHist.png", dpi=300)
    plt.close()

def plot_mesh(vertices, edges, cells, step, folder=""):
    # vertices = mesh.mesh[step]['vertices']
    # edges = mesh.mesh[step]['edges']
    # cells = mesh.mesh[step]['cells']

    vids = [10, 41, 3, 50, 54, 72, 28, 89, 33, 13, 44, 98, 110, 36, 34, 25, 91, 113, 81]

    for v in vertices.values():
        if v.id in vids:
            plt.scatter(v.x, v.y, s=5,color="blue")
        # plt.annotate(str(v.id), [v.x, v.y], fontsize=3, color="red")

    # for e in edges.values():
    #     # if e.v1.id not in vertices.keys() or e.v2.id not in vertices.keys():
    #         # print("########################")
    #         # print("Vertex", e.v1, " or ", e.v2, " missing")
    #         # print("In edge: ", e)
    #         # print("########################")
    #     # earr = [5, 782, 368, 1196, 161, 989, 575, 1403, 3, 783, 369, 1197, 162, 990, 576, 1404, 2, 1414, 586, 1000, 172, 1207, 379, 793, 17]
    #     # if e.id in earr1:
    #         # plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="orange")
    #     plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="red")
    #     plt.annotate(str(e.id), [(e.v1.x +  e.v2.x)/2 , (e.v1.y + e.v2.y)/2], fontsize=3, fontweight="bold")
    #     # elif e.id in earr2:
    #     plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="blue")
    #     plt.annotate(str(e.id), [(e.v1.x +  e.v2.x)/2 , (e.v1.y + e.v2.y)/2], fontsize=3, fontweight="bold")

    for c in cells.values():
        # cm = c.get_cm()
        # print(c.id, cm)
        cxs = [v.x for v in c.vertices]
        cys = [v.y for v in c.vertices]

        plt.fill(cxs, cys, alpha=0.3)
        # plt.annotate(str(c.id), [cm[0], cm[1]])
        # plt.scatter(cm[0], cm[1], marker="x", color="red")

    # # plt.show()
    # plt.xlim(-50, 60)
    # plt.ylim(-160, -75)
    plt.savefig(os.path.join(folder, "mesh_"+str(step)+".png"), dpi=500)
    plt.clf()

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
    plt.savefig(os.path.join(folder, str(step) + ".png"), dpi=500)
    plt.clf()


def plot_time_connections(mesh, initial_time, final_time, step=1, folder=''):
    timerange = np.arange(initial_time, final_time-2, step)
    for t in timerange:
        print("Plotting ", t)
        if mesh.mapping[t] != None:
            real_vertices_ids1 = np.array([[x[0]]+[x[-1]] for x in mesh.time_series[t].earr])
            real_vertices_ids2 = np.array([[x[0]]+[x[-1]] for x in mesh.time_series[t+1].earr])
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
                plt.plot([e.v1.x, e.v2.x], [e.v1.y, e.v2.y], color="orange")
                
            plt.savefig(os.path.join(folder, "points_"+str(t)+".png"), dpi=500)
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

def plot_velocity_heatmap(mesh, initial_time, final_time, folder='', name='heatmap'):
    t0 = mesh.time_series[initial_time]
    # heatmap = np.zeros((timerange, len(t0['earr'])))
    heatmap = []
    for ii in range(0, len(t0.earr)):
        row = mesh.velocity_per_edge(ii, initial_time, final_time)
        if not np.all(np.isnan(row)):
            heatmap.append(row) 
    
    save_heatmap(heatmap, folder, initial_time, final_time, name=name)

def save_heatmap(heatmap, folder, initial_time, final_time, name='heatmap'):
    plt.imshow(heatmap)
    plt.colorbar()
    plt.xticks(np.arange(0, final_time - initial_time, 5), np.arange(initial_time, final_time, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, str(name)+"_"+str(initial_time)+".png"), dpi=500)
    plt.clf()