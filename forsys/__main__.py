
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

    args = parser.parse_args()

    if args.save_folder is not None:
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        if not os.path.exists(os.path.join(args.save_folder, "pngs")):
            os.makedirs(os.path.join(args.save_folder, "pngs"))

        save_to_tiff = os.path.join(args.save_folder,
                                    "forsys_output.tif")
        save_to_png = os.path.join(args.save_folder, "pngs",
                                   "forsys_output.png")
    else:
        save_to_tiff = os.path.join(os.path.dirname(args.folder),
                                    "forsys_output.tif")
        save_to_png = os.path.join(os.path.dirname(args.folder),
                                   "forsys_output.png")
    # DATA_FOLDER = os.path.dirname(args.folder)

    print("This is forsys version ", importlib.metadata.version("forsys"))
    print(f"Running in Dynamic modality {args.dynamic}.",
          f"VN = {args.velocity_normalization}",
          f"Method = {args.method}",)

    print("Inferring...")
    initial_guess = fs.auxiliar.load_initial_guess(args.connections,
                                                   min_time=0,
                                                   max_time=args.max_time)

    microscopies = sorted(list(filter(lambda x: x.endswith(".tif"),
                                      os.listdir(args.folder))))
    if args.dynamic:
        print("Dynamic inference")
        if len(microscopies) == 1:
            print("Only one microscopy found. Inference will be static")

    frames = {}
    time = 0
    image_sizes = []
    original_images = []
    for microscopy in microscopies:
        if args.composite:
            im_ori_fp = Image.open(os.path.join(args.folder, microscopy))
            im_ori = np.array(im_ori_fp)
            original_images.append(im_ori)
            im_ori_fp.close()

        im = Image.open(os.path.join(args.folder, microscopy))
        image_sizes.append(im.size)
        im.close()

        tiff_file = os.path.join(args.folder,
                                 microscopy[:-4],
                                 "handCorrection.tif")

        skeleton = fs.skeleton.Skeleton(tiff_file, mirror_y=False)
        vertices, edges, cells = skeleton.create_lattice()
        vertices, edges, cells, _ = fs.virtual_edges.generate_mesh(vertices,
                                                                   edges,
                                                                   cells,
                                                                   ne=5)
        frames[time] = fs.frames.Frame(time,
                                       vertices,
                                       edges,
                                       cells,
                                       time=time)
        forsys = fs.ForSys(frames, initial_guess=initial_guess,)

        if args.dynamic:
            all_velocities = forsys.get_system_velocity_per_frame()
            ave_velocity_to_normalize = np.mean(all_velocities)
        else:
            ave_velocity_to_normalize = 1

        velocity_normalization = ((args.velocity_normalization /
                                  ave_velocity_to_normalize)
                                  if args.dynamic else 0)
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
        if time == args.max_time:
            break
        time += 1

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

    # Now tiff
    fs.plot.plot_inference_as_tiff(forsys,
                                   image_sizes=image_sizes,
                                   original_images=original_images,
                                   layers=2,
                                   external_color=0,
                                   save_folder=save_to_tiff)
