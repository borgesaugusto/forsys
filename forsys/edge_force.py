def edge_force(configuration, folder=''):
    index = []
    means = []
    stds = []
    edges = create_edges(configuration)
    print(edges)
    for e in edges:
        modArray = []
        # Create the vector to project from ( from v1 to vn)
        first = configuration.getVertices()[e[0]]
        last = configuration.getVertices()[e[-1]]
        vector = (last.getX() - first.getX(), last.getY() - first.getY())
        eNorm = np.sqrt(np.dot(vector, vector))

        # take the parallel and normal components of the force.
        product = 0
        for v in e:
            vobj = configuration.getVertices()[v]
            lx, ly = vertex_line_force(configuration, v, lineConstant=1)
            # ax, ay = vertex_area_force(configuration, v)
            # px, py = vertex_perimeter_force(configuration, v)
            # print("Forces: Line \t Area \t Perimeter")
            # print( (lx, ly), (ax, ay), (px, py))
            # fVector = (lx+ ax + px, ly + ay + py)
            fVector = (lx, ly)
            if fVector != (0, 0) and eNorm != 0:
                norm = np.sqrt(np.dot(fVector, fVector))
                scalarPart = norm * cosAngle(vector, fVector) / eNorm
                # Parallel vector:
                pVector = np.multiply(scalarPart, vector)
                nVector = fVector - pVector
                # Check whether the vertex is elongating or shrinking. Positive: Eloganting
                # to do this we take the scalar product and sum.
                # middle of the edgeVector:
                mp = middle_point(first, last)
                # vector from center to vertex
                v1 = (vobj.getX() - mp[0], vobj.getY() - mp[1])
                v1Norm = np.sqrt(np.dot(v1, v1))
                scPart = v1Norm * cosAngle(vector, v1) / eNorm
                if scPart != 0:
                    # Projection of the edge center to the large edge:
                    edgeVectorParallel = np.multiply(scPart, vector)
                    evpNorm = np.sqrt(np.dot(edgeVectorParallel, edgeVectorParallel))
                    v1Normalized = np.multiply(edgeVectorParallel, 1/evpNorm)
                    product += np.dot(pVector, v1Normalized)
            else:
                pVector = (0, 0)
                nVector = (0, 0)
            if len(modArray) < len(e) - 1:
                modArray.append(np.sqrt(np.dot(pVector, pVector)) * np.sign(product))
            # print(v, round(norm, 3), round(np.sqrt(np.dot(pVector, pVector)), 3))

        index.append(edges.index(e))
        means.append(np.mean(modArray))
        stds.append(np.std(modArray))
    #    print(edges.index(e), np.mean(modArray), np.std(modArray))
    df = pd.DataFrame()
    df['index'] = index
    df['mean'] = means
    df['std'] = stds
    df.to_csv(str(folder)+"log/forces.dat", sep='\t', index=False)
    # plot.plot_force(df['mean'])
    #print(df['index'], df['mean'])
    return df, edges
