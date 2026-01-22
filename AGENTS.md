# AGENTS.md

## Overview
**ForSys** is a Python-based open-source tool for stress inference in biological tissues. It enables the estimation of membrane stresses and intracellular pressures from segmented microscopy images by translating tissue geometry into systems of linear equations. It supports three modalities: **Python scripting**, **Command Line Interface (CLI)**, and a **Graphical User Interface (GUI)**.

---

## 1. Environment Setup
Agents must ensure the following Python environment is prepared:
* **Python Version:** 3.8 or later (v3.11 recommended).
* **Installation:** `pip install forsys`.
* **Environment Manager:** Recommended use of Anaconda/Miniconda to manage the `forsys_env`.

---

## 2. Input Data & File Structure
For CLI and GUI modes, ForSys requires a strict directory hierarchy to associate images with segmentations.

### Required Directory Tree
```text
project_root/
├── data/
│   ├── sample_01.tif          # Raw experimental image (single frame)
│   ├── sample_01/             # Folder name MUST match the .tif filename
│   │   └── handCorrection.tif # 1-px wide skeleton (TIFF) or CellPose mask (.npy)
│   ├── sample_02.tif
│   └── sample_02/
│       └── handCorrection.tif
└── results/                   # Target directory for inference outputs

```

* 
**Segmentation Priority:** If both a TIFF skeleton and an NPY mask are present, ForSys defaults to the TIFF skeleton.


* 
**Fallback:** If `handCorrection` is missing, it defaults to any TIFF and then any NPY found in the subfolder.



---

## 3. Graphical User Interface (GUI) Modality

### GUI Installation (Fiji)

1. 
**Open Fiji:** Ensure Fiji is installed on the system.


2. 
**Add Update Site:** Go to `Help -> Update -> Manage update site`.


3. 
**URL:** Click "Add unlisted site" and use `https://sites.imagej.net/ForSys`.


4. 
**Apply:** Select the site, click "Apply and Close", then "Apply Changes" and restart Fiji.



### GUI Execution

1. 
**Launch:** Click `Plugins -> ForSys Plugin`.


2. **Configure:**
* 
**Conda Env:** Enter the environment name (e.g., `forsys_env`).


* 
**Input Folder:** Browse to the top-level `data/` folder.


* 
**Results Folder:** Select the destination for output files.


* 
**CSV Output:** Add `-o` to the **Extra arguments** textbox to output numerical stress and pressure CSVs.




3. 
**Run:** Click "Run ForSys".



---

## 4. Command Line Interface (CLI) Modality

Agents can run inferences via terminal commands within the Python environment.

* **Static Inference:**


`python -m forsys -f /path/to/data -m nnls -sf /path/to/results -o` 


* **Dynamic Inference:**


`python -m forsys -f /path/to/data -m nnls --dynamic -sf /path/to/results -o` 


* **Flags:**
* 
`-m`: Solver choice (e.g., `nnls`).


* `-o`: Generates quantitative CSV outputs for tensions and pressures.
* 
`-sf`: The directory where results are saved.





---

## 5. Python API Usage (Scripting)

To integrate ForSys into a custom workflow, follow this implementation pattern:

### A. Creating Frames from Microscopy

For each timepoint, you must load the skeleton and generate a mesh to create a `Frame` object.

```python
import forsys as fs
import os

frames = {}
max_time = 5  # Example frame count

for time in range(max_time):
    # Specify the image file path
    tif_path = os.path.join("data", f"t_{time}.tif")
    
    # Load skeleton and create the tissue lattice
    skeleton = fs.skeleton.Skeleton(tif_path, mirror_y=False)
    vertices, edges, cells = skeleton.create_lattice()
    
    # Generate mesh for force calculation (ne=5 is a standard subdivision)
    vertices, edges, cells, _ = fs.virtual_edges.generate_mesh(vertices, edges, cells, ne=5)
    
    # Store in a Frame object
    frames[time] = fs.frames.Frame(time, vertices, edges, cells, time=float(time))

```

### B. Running Inference

1. 
**Initialize:** `forsys_obj = fs.ForSys(frames)`.


2. 
**Solve Stress:** `forsys_obj.solve_stress(when=time_index, method="nnls")`.


3. 
**Solve Pressure:** `forsys_obj.solve_pressure(when=time_index, method="lagrange_pressure")`.


4. 
**Extract Data:** - `tensions_df = forsys_obj.frames[0].get_tensions()` 


* 
`pressures_df = forsys_obj.frames[0].get_pressures()` 





---

## 6. Troubleshooting for Agents

* 
**"Found 0 frames":** Verify that the subfolder name matches the TIFF filename exactly and contains a `handCorrection` file.


* 
**"ModuleNotFoundError":** Ensure you are in the correct Conda environment and ForSys is installed.


* 
**"ValueError: max() arg is an empty sequence":** This occurs if you attempt to access results before running the `solve_stress` or `solve_pressure` methods.



```

```