[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11282555.svg)](https://doi.org/10.5281/zenodo.11282555)
[![PyPI version](https://badge.fury.io/py/forsys.svg)](https://pypi.org/project/forsys/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/forsys)](https://pypi.org/project/forsys/)
[![codecov](https://codecov.io/gh/borgesaugusto/forsys/graph/badge.svg?token=09Z4YKV8IG)](https://codecov.io/gh/borgesaugusto/forsys)
[![Documentation Status](https://readthedocs.org/projects/forsys/badge/?version=latest)](https://forsys.readthedocs.io/en/latest/?badge=latest)
# ForSys
#### ForSys: a non-invasive open-source analysis software to infer stress from microscopic time series 

---
## What is ForSys ?
ForSys can infer membrane stress and cell pressure from a skeletonized microscopy image or a time series.

##  Documentation: 
Available at
https://forsys.readthedocs.io/en/latest/

---

## Installation
To install, just run
```bash
pip install forsys
```
---

## Usage
### Static ForSys: Inferring on a single frame
After importing ForSys, the segmented image needs to be parsed into the *Skeleton* class
```python
import forsys as fsys
skeleton = fsys.skeleton.Skeleton("path/to/segmented/image.tif")
vertices, edges, cells = skeleton.create_lattice()
```
Then, the skeleton should be meshed to the desired amount of vertices per edge (in order to account for the curved edges)
in this example, the number is set to 6, using the *ne* keyworded argument.
```python
vertices, edges, cells, _ = fsys.virtual_edges.generate_mesh(vertices, edges, cells, ne=6)
```
---
Next, we create a *Frame* object with our vertices, edges and cells. Finally creating the *forsys* object
```python
frames[0] = fsys.frames.Frame(0, vertices, edges, cells, time=0)
forsys = fsys.ForSys(frames)
```

Now, we can create the matrix representation of the system for the stress or the pressures and solve it to get the 
inferred values. This is done for the stress as
```python
fsys.build_force_matrix(when=0)
fsys.solve_stress(when=0, allow_negatives=False)
```
and similarly for the pressure
```python
fsys.build_pressure_matrix(when=0)
fsys.solve_pressure(when=0, method="lagrange_pressure")
```

### Dynamical ForSys: Inferring on a time-series
The main difference in this context is the necessity to create more than one frame. We followed the same steps as before
to create generate each frame's mesh, but now create a different frame for each image in the time-series. Now, each frame
will have a *frame_number* and a *time*. The first number works as an ID of the frame and tells the order of the frames,
while the second one, time, is used to calculate the dynamical variables, such as velocity. 
```python
import forsys as fsys

for frame_id, frames in enumerate(images_path):
    real_time = 5 * frame_id # Image every 5 seconds
    skeleton = fsys.skeleton.Skeleton(frames)
    vertices, edges, cells = skeleton.create_lattice()
    vertices, edges, cells, _ = fsys.virtual_edges.generate_mesh(vertices, edges, cells, ne=6)
    frames[frame_id] = fsys.frames.Frame(frame_id,
                                         vertices,
                                         edges, 
                                         cells,
                                         time=real_time)
forsys = fsys.ForSys(frames, cm=False)
```
Then, after all frames have been created we again use the *fsys.ForSys()* class to create the mesh. If there is more than
one element in the *frames* it will automatically try to track the vertices during the series. Creation of the matrices 
and its solution is done identically, but we have to specify at which time we want to do it
```python
fsys.build_force_matrix(when=frame_number_to_solve)
fsys.solve_stress(when=frame_number_to_solve, 
                    allow_negatives=False,
                    b_matrix="velocity")

fsys.build_pressure_matrix(when=frame_number_to_solve)
fsys.solve_pressure(when=frame_number_to_solve, method="lagrange_pressure")
```
Importantly, the *b_matrix* parameter indicates ForSys to use the vertex velocity in the right hand side of the equation
(please refer to the Materials and Methods section of th paper for the details).

#### Plotting the results
For plotting the results, the main function to use is *forsys.plot.plot_inference()* and then specify whether to include
the pressure or not. If the system was already solved,
```python
fsys.plot.plot_inference(forsys.frames[0],
                       step="file_name_to_write",
                       folder="path/to/save/folder",
                       pressure=True,
                       normalized="max",
                       mirror_y=True,
                       colorbar=True)
```
By default, the membrane tension is always plotted, the pressure=True parameter plots the inferred pressures as the color 
filling the cells. 


### How to cite us
Our tool's manuscriptt is available at [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.05.28.595800v1)
To cite the tool, you can use:

**ForSys: non-invasive stress inference from time-lapse microscopy**
Augusto Borges, Jerónimo R. Miranda-Rodríguez, Alberto Sebastián Ceccarelli, Guilherme Ventura, Jakub Sedzinski, Hernán López-Schier, Osvaldo Chara.
bioRxiv 2024.05.28.595800; doi: https://doi.org/10.1101/2024.05.28.595800
