# ETSE-UV 3D Slicer Custom Modules for Ear Geometry Processing

This repository contains custom Python modules for **3D Slicer** developed for ear geometry processing, registration, mesh analysis, fiducial handling, measurement transfer, and related workflows.

The tools are associated with the work:

**Geometry Processing Tools for Parametric Representation of Ear Anatomy**

## Overview

This repository is intended to be used as a 3D Slicer custom module folder. The modules are written in Python and are designed to run inside the 3D Slicer environment.

The repository includes tools for:

- automatic mesh registration;
- fuzzy registration;
- Sumner-Amberg registration;
- trimesh-based registration;
- fiducial indexing;
- mesh ordering;
- mesh path tools;
- loop-region extraction;
- measurement transfer;
- registration metrics;
- SOFA/HRTF plotting.

## Repository Structure

The repository root contains the main 3D Slicer module files:

```text
.
│   .gitignore
│   ETSE_UV__AutoRegistration.py
│   ETSE_UV__FiducialIndexer.py
│   ETSE_UV__FuzzyRegistration.py
│   ETSE_UV__LoopRegionExtractor.py
│   ETSE_UV__MeasurementTransfer.py
│   ETSE_UV__MeshOrderer.py
│   ETSE_UV__MeshPathTools.py
│   ETSE_UV__RegistrationMetrics.py
│   ETSE_UV__SofaHrtfPlotter.py
│   ETSE_UV__SumnerAmbergRegistration.py
│   ETSE_UV__TrimeshRegistration.py
│   README.md
│
├───fuzzy_lib
│   │   fuzzyclusterreg.py
│   └   fuzzyclusterreg_gpu.py
│
└───Resources
    │   ETSE_UV__Dependencies.py
    │
    └───Icons/
```

## Important Path Requirement

Do **not** change the folder structure unless you also update the imports in the Python files.

Several modules depend on local files using imports such as:

```python
from Resources.ETSE_UV__Dependencies import ensure_packages
```

The local fuzzy registration implementation is stored in:

```text
fuzzy_lib/
```

The expected structure is:

```text
.
├── ETSE_UV__*.py
├── fuzzy_lib/
└── Resources/
    ├── ETSE_UV__Dependencies.py
    └── Icons/
```

The repository root should be added directly to 3D Slicer as an additional module path.

## Installation

### Option 1: Clone with Git

Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```

Then open 3D Slicer.

### Option 2: Download as ZIP

Download the repository as a ZIP file from GitHub, extract it, and keep the extracted folder structure unchanged.

## Loading the Modules in 3D Slicer

1. Open **3D Slicer**.

2. Go to:

   ```text
   Edit > Application Settings > Modules
   ```

3. In **Additional module paths**, add the repository folder.

4. Restart 3D Slicer.

5. The ETSE-UV modules should appear in the Slicer module list.

## Dependency Handling

Dependencies are handled inside the Slicer Python environment using the local helper:

```python
from Resources.ETSE_UV__Dependencies import ensure_packages
```

For example, some modules contain code like:

```python
import numpy as np
from Resources.ETSE_UV__Dependencies import ensure_packages

ensure_packages(
    [("trimesh", "trimesh")],
    interactive=False,
    module_name="ETSE-UV Sumner-Amberg Registration",
)

import trimesh
```

This means that required Python packages may be checked or installed when a module is loaded or executed.

The dependency helper must remain here:

```text
Resources/ETSE_UV__Dependencies.py
```

## Local Fuzzy Registration Library

The fuzzy registration code is included locally in:

```text
fuzzy_lib/
├── fuzzyclusterreg.py
├── fuzzyclusterreg_gpu.py
└── __init__.py
```

These files are required by the fuzzy registration module and should not be removed.

## Included Modules

### `ETSE_UV__AutoRegistration.py`

Custom 3D Slicer module for automatic registration workflows.

### `ETSE_UV__FiducialIndexer.py`

Custom 3D Slicer module for handling and indexing fiducial points.

### `ETSE_UV__FuzzyRegistration.py`

Custom 3D Slicer module for fuzzy registration using the local `fuzzy_lib` implementation.

### `ETSE_UV__LoopRegionExtractor.py`

Custom 3D Slicer module for extracting loop-like regions from mesh geometry.

### `ETSE_UV__MeasurementTransfer.py`

Custom 3D Slicer module for transferring measurements between meshes or registered anatomical models.

### `ETSE_UV__MeshOrderer.py`

Custom 3D Slicer module for mesh ordering operations.

### `ETSE_UV__MeshPathTools.py`

Custom 3D Slicer module for working with paths, curves, or point sequences on mesh surfaces.

### `ETSE_UV__RegistrationMetrics.py`

Custom 3D Slicer module for computing registration metrics.

### `ETSE_UV__SofaHrtfPlotter.py`

Custom 3D Slicer module for SOFA/HRTF plotting workflows.

### `ETSE_UV__SumnerAmbergRegistration.py`

Custom 3D Slicer module for Sumner-Amberg-style registration workflows.

### `ETSE_UV__TrimeshRegistration.py`

Custom 3D Slicer module for registration workflows using `trimesh`.

## Icons

Module icons are stored in:

```text
Resources/Icons/
```

Each module has a corresponding icon file. These icons are used by 3D Slicer when displaying the modules in the interface.

## Troubleshooting

### Modules do not appear in 3D Slicer

Make sure the repository root folder was added to:

```text
Edit > Application Settings > Modules > Additional module paths
```

Then restart 3D Slicer.

### Import error for `Resources.ETSE_UV__Dependencies`

Check that this file exists:

```text
Resources/ETSE_UV__Dependencies.py
```

Also check that the main module files are still located in the repository root.

## Local Fuzzy Registration Library

The fuzzy registration code is included locally in:

```text
fuzzy_lib/
├── fuzzyclusterreg.py
├── fuzzyclusterreg_gpu.py
└── __init__.py
```

This module wraps the ClusterReg implementation of:

Mingyang Zhao, Jingen Jiang, Lei Ma, Shiqing Xin, Gaofeng Meng, Dong-Ming Yan.
"Correspondence-Free Nonrigid Point Set Registration Using Unsupervised
Clustering Analysis." Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2024.

Original project: zikai1/ClusterReg  (https://github.com/zikai1/ClusterReg)

 The bundled fuzzy_lib registration backend contains code from ClusterReg.
 ClusterReg is distributed under the AGPL-3.0 license. Keep the original
 license and attribution notices when redistributing this module.

 BibTeX:
 ```bibtex
@inproceedings{zhao2024clustereg,
title={Correspondence-Free Nonrigid Point Set Registration Using Unsupervised Clustering Analysis},
author={Mingyang Zhao and Jingen Jiang and Lei Ma and Shiqing Xin and Gaofeng Meng and Dong-Ming Yan},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2024}
}
```
### External package missing

Some modules request external packages through `ensure_packages`.

If a package cannot be installed automatically, check the 3D Slicer Python console for the error message.

### Module icons do not appear

Check that the icon files exist in:

```text
Resources/Icons/
```

## Development Notes

When adding a new module, place the main `.py` file in the repository root, next to the existing `ETSE_UV__*.py` modules.

Shared helper files should be placed inside:

```text
Resources/
```

Local registration libraries or other bundled code should be kept in their existing local folders, such as:

```text
fuzzy_lib/
```

Do not commit Python cache files such as:

```text
__pycache__/
*.pyc
```

## Recommended `.gitignore`

A suitable `.gitignore` for this repository is:

```gitignore
__pycache__/
*.pyc
*.pyo
*.pyd

.vscode/
.idea/

.DS_Store
Thumbs.db

*.log
```

## Citation

The associated manuscript has been submitted to the **XXXV Spanish Conference on Computer Graphics (CEIG 2026)** and is not yet published.

Until the final publication details are available, please cite it as:

```bibtex
@misc{DeRus2026_CEIG_GeomTools,
  author = {De Rus Arance, Juan Antonio and Castorena, Carlos and Montagud, Mario and Ferri, Francesc J. and Cobos, Maximo},
  title  = {Geometry Processing Tools for Parametric Representation of Ear Anatomy},
  note   = {Manuscript submitted to the XXXV Spanish Conference on Computer Graphics (CEIG 2026), February 2026. Not yet published.},
  year   = {2026}
}
```
## License

Add the project license here.

If no license is provided, all rights are reserved by the authors.
