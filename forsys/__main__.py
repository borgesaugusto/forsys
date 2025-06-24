
if __name__ == '__main__':
    import importlib.metadata
    import matplotlib.pyplot as plt
    import forsys as fs
    import argparse
    import os
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(description="Forsys interface command" +
                                     "line interface.")
    parser.add_argument("-f", "--folder", type=str,
                        help="Path to the folder containing the input images.")
    parser.add_argument("-d", "--dynamic", action="store_true",
                        help="Whether to use dynamic inference.\
                        Default is static")
    parser.add_argument("-m", "--method", type=str,
                        help="Method to use for inference. Options are: " +
                        "'nnls', 'lsq' and 'lsq_linear. Default is 'nnls'",
                        default="nnls")
    parser.add_argument("-vn", "--velocity_normalization", type=float,
                        help="Velocity normalization factor \
                        (only used in dynamic mode). Default is 0.1",
                        default=0.1)
    parser.add_argument("-mt", "--max_time", type=int,
                        help="Maximum time step to infer. Default is 0",
                        default=0)
    parser.add_argument("-sf", "--save_folder", type=str,
                        help="Path to the folder in which to save the output. \
                        Default is the same as the input folder.",
                        default=None)
    parser.add_argument("-cc", "--composite", action="store_true",
                        help="Whether to create composite image with the \
                        original microscopies")
    parser.add_argument("-p", "--pngs", action="store_true",
                        help="Whether to create forsys output of each frame")
    parser.add_argument("-c", "--connections", type=str,
                        help="Path to the connections file (optional)",
                        default=None)
    parser.add_argument("-ar", "--aspect_ratio", type=float,
                        help="Aspect ratio of the PNG images. Default is 1",
                        default=1)
    parser.add_argument("-y", "--pressure", action="store_true",
                        help="Plot pressure in the PNG images")
    parser.add_argument("-cb", "--colorbar", action="store_true",
                        help="Plot colorbar in the PNG images")
    parser.add_argument("-v", "--version", action="version",
                        version=f"ForSys version {importlib.metadata.version('forsys')}",
                        help="Show the version of forsys")
    parser.add_argument("-pc", "--plot_connections", action="store_true",
                        help="Plot connections without solving the system. \
                        Useful for debugging purposes.")
    parser.add_argument("-ms", "--max_size", type=int,
                        help="Maximum multiplier for cellular size. \
                        used to filter out cells that are too big. Default is \
                        no limit",
                        default=np.inf)
    parser.add_argument("-o", "--output_csv", actions="store_true",
                        help="Output the results to a CSV file.")

    args = parser.parse_args()

    if args.save_folder is not None:
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        if args.pngs and not os.path.exists(os.path.join(args.save_folder, "pngs")):
            os.makedirs(os.path.join(args.save_folder, "pngs"))
        if args.plot_connections and not os.path.exists(os.path.join(args.save_folder, "connections")):
            os.makedirs(os.path.join(args.save_folder, "connections"))
        if args.output_csv and not os.path.exists(os.path.join(args.save_folder, "csvs")):
            os.makedirs(os.path.join(args.save_folder, "csvs"))

        save_to_tiff = os.path.join(args.save_folder,
                                    "forsys_output.tif")
        save_to_png = os.path.join(args.save_folder, "pngs",
                                   "forsys_output.png")
    else:
        save_to_tiff = os.path.join(os.path.dirname(args.folder),
                                    "forsys_output.tif")
        save_to_png = os.path.join(os.path.dirname(args.folder),
                                   "forsys_output.png")

    print("This is forsys version ", importlib.metadata.version("forsys"))
    modality = "dynamic" if args.dynamic else "static"
    print(f"Running in {modality} modality.",
          f"VN = {args.velocity_normalization}",
          f"Method = {args.method}",)

    print("Inferring...")
    initial_guess = fs.auxiliar.load_initial_guess(args.connections,
                                                   min_time=0,
                                                   max_time=args.max_time)

    elements_in_folder = os.listdir(args.folder)
    # find microscopies
    sub_folders = filter(lambda x: os.path.isdir(os.path.join(args.folder,
                                                              x)),
                         elements_in_folder)
    sub_folders = sorted(list(sub_folders))

    segmentations = {}
    for sf in sub_folders:
        elements_in_sf = os.listdir(os.path.join(args.folder, sf))
        tif_elements = list(filter(lambda x: x.endswith(".tif"),
                                   elements_in_sf))
        npy_elements = list(filter(lambda x: x.endswith(".npy"),
                                   elements_in_sf))
        if "handCorrection.tif" in tif_elements:
            element_tif = os.path.join(args.folder, sf, "handCorrection.tif")
            segmentations[sf] = element_tif
        elif "handCorrection.npy" in npy_elements:
            element_npy = os.path.join(args.folder, sf, "handCorrection.npy")
            segmentations[sf] = element_npy
        elif len(tif_elements) == 1:
            segmentations[sf] = os.path.join(args.folder,
                                             sf,
                                             tif_elements[0])
        elif len(npy_elements) == 1:
            segmentations[sf] = os.path.join(args.folder,
                                             sf,
                                             npy_elements[0])
        else:
            print(f"No suitable or too many segmentations found in {sf}. Skipping.")

    if args.dynamic:
        if len(segmentations) == 1:
            print("Only one microscopy found. Inference will be static")

    frames = {}
    time = 0
    image_sizes = []
    original_images = []
    # for microscopy in microscopies_tif:
    for microscopy_name, segmentation_file in segmentations.items():
        microscopy_path = os.path.join(args.folder, f"{microscopy_name}.tif")
        if os.path.exists(microscopy_path):
            im_ori_fp = Image.open(microscopy_path)
            image_sizes.append(im_ori_fp.size)
            if args.composite:
                im_ori = np.array(im_ori_fp)
                original_images.append(im_ori)
                im_ori_fp.close()
        else:
            if microscopy_name.endswith(".npy"):
                im_ori_fp = np.load(segmentation_file)
                image_sizes.append(im_ori_fp.shape[:2])
            # im_for_size = Image.open(segmentation_file)
            # image_sizes.append(im_for_size.size)

        skeleton = fs.skeleton.Skeleton(segmentation_file, mirror_y=False)
        vertices, edges, cells = skeleton.create_lattice(max_cell_size=args.max_size)
        vertices, edges, cells, _ = fs.virtual_edges.generate_mesh(vertices,
                                                                   edges,
                                                                   cells,
                                                                   ne=5)
        frames[time] = fs.frames.Frame(time,
                                       vertices,
                                       edges,
                                       cells,
                                       time=time)
        time += 1
        if time == args.max_time:
            break

    forsys = fs.ForSys(frames, initial_guess=initial_guess,)
    print(f"Found {len(forsys.frames)} frames in the system.")

    if args.plot_connections:
        print("Plotting connections without solving the system.")
        fs.plot.plot_time_connections(forsys.mesh, 0, len(forsys.frames),
                                      folder=os.path.join(args.save_folder,
                                                          "connections"))
        exit()

    if args.dynamic:
        all_velocities = forsys.get_system_velocity_per_frame()
        ave_velocity_to_normalize = np.mean(all_velocities)
    else:
        ave_velocity_to_normalize = 1

    velocity_normalization = ((args.velocity_normalization /
                              ave_velocity_to_normalize)
                              if args.dynamic else 0)

    for time in frames.keys():
        forsys.build_force_matrix(when=time)
        forsys.solve_stress(when=time,
                            b_matrix="dynamic" if args.dynamic else "static",
                            velocity_normalization=velocity_normalization,
                            adimensional_velocity=False,
                            method=args.method,
                            use_std=False,
                            allow_negatives=False)
        forsys.build_pressure_matrix(when=time)
        forsys.solve_pressure(when=time, method="lagrange_pressure")

    print("Plotting...")
    if args.pngs:
        for time in frames.keys():
            fig, ax = plt.subplots(1, 1, figsize=(8, 3))
            fs.plot.plot_inference(forsys.frames[time],
                                   normalized="max",
                                   mirror_y=False,
                                   colorbar=args.colorbar,
                                   pressure=args.pressure,
                                   aspect_ratio=args.aspect_ratio,
                                   ax=ax)
            ax.set_title(f"Time: {time}")
            ax.axis("off")
            plt.tight_layout()

            plt.savefig(save_to_png.replace(".png", f"_{time}.png"), dpi=500)

    if args.output_csv:
        import pandas as pd
        print("Outputting results to CSV file.")
        tensions = []
        lengths = []
        positions_x = []
        positions_y = []
        be_ids = []
        curvatures = []

        cell_ids = []
        areas = []
        perimeters = []
        numbers_of_neighs = []
        cell_posx = []
        cell_posy = []
        pressures = []

        for cellid, cell in frames[time].cells.items():
            cell_ids.append(cellid)
            areas.append(abs(cell.get_area()))
            perimeters.append(cell.get_perimeter())
            numbers_of_neighs.append(len(cell.calculate_neighbors()))
            cell_posx.append(cell.get_cm()[0])
            cell_posy.append(cell.get_cm()[1])
            pressures.append(cell.pressure)

        cell_df = pd.DataFrame({
               "id": cell_ids,
               "area": areas,
               "perimeter": perimeters,
               "position_x": cell_posx,
               "position_y": cell_posy,
               "pressure": pressures,
               "number_of_neighbors": numbers_of_neighs,
        })
        cell_df.to_csv(os.path.join(args.save_folder, "csvs",
                                    f"cells_{time}.csv"),
                       index=False)

        for beid, big_edge in frames[time].big_edges.items():
            tensions.append(big_edge.tension)
            lengths.append(big_edge.get_length())
            positions_x.append(np.mean(big_edge.xs))
            positions_y.append(np.mean(big_edge.ys))
            be_ids.append(big_edge.big_edge_id)
            curvatures.append(big_edge.calculate_total_curvature())

            force_df = pd.DataFrame({
                "id": be_ids,
                "tension": tensions,
                "length": lengths,
                "position_x": positions_x,
                "position_y": positions_y,
                "curvature": curvatures,
            })

    force_df.to_csv(os.path.join(args.save_folder, "csvs",
                                 f"stress_{time}.csv"),
                    index=False)

    # Now tiff
    fs.plot.plot_inference_as_tiff(forsys,
                                   image_sizes=image_sizes,
                                   original_images=original_images,
                                   layers=2,
                                   external_color=0,
                                   save_folder=save_to_tiff)
