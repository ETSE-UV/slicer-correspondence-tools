# ETSE_UV__FuzzyRegistration.py
# ETSE_UV__FuzzyRegistration.py
# 3D Slicer scripted module
#
# This module wraps the ClusterReg implementation of:
#
# Mingyang Zhao, Jingen Jiang, Lei Ma, Shiqing Xin, Gaofeng Meng, Dong-Ming Yan.
# "Correspondence-Free Nonrigid Point Set Registration Using Unsupervised
# Clustering Analysis." Proceedings of the IEEE/CVF Conference on Computer
# Vision and Pattern Recognition (CVPR), 2024.
#
# Original project: zikai1/ClusterReg
#
# The bundled fuzzy_lib registration backend contains code from ClusterReg.
# ClusterReg is distributed under the AGPL-3.0 license. Keep the original
# license and attribution notices when redistributing this module.
#
# BibTeX:
# @inproceedings{zhao2024clustereg,
#   title={Correspondence-Free Nonrigid Point Set Registration Using Unsupervised Clustering Analysis},
#   author={Mingyang Zhao and Jingen Jiang and Lei Ma and Shiqing Xin and Gaofeng Meng and Dong-Ming Yan},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   year={2024}
# }

import os
import sys
import math
import vtk
import qt
import ctk
import slicer
import numpy as np
import time
import csv

from slicer.ScriptedLoadableModule import *
from Resources.ETSE_UV__Dependencies import ensure_packages

ensure_packages(
    [
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
    ],
    interactive=False,
    module_name="ETSE-UV Fuzzy Registration",
)

# Prefer SciPy KDTree (available in recent Slicer builds); fall back to VTK locator if needed
try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class ETSE_UV__FuzzyRegistration(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Fuzzy Registration"
        parent.categories = ["ETSE_UV"]
        parent.dependencies = []
        parent.contributors = ["ETSE UV"]
        parent.helpText = """
        <p>Register a SOURCE mesh to a TARGET mesh using the ClusterReg correspondence-free
        non-rigid point-set registration method.</p>

        <p><b>Reference:</b> Mingyang Zhao, Jingen Jiang, Lei Ma, Shiqing Xin,
        Gaofeng Meng, and Dong-Ming Yan. <i>Correspondence-Free Nonrigid Point Set
        Registration Using Unsupervised Clustering Analysis</i>. CVPR 2024.</p>

        <p><b>Pipeline:</b></p>
        <ol>
          <li>Voxel downsample the SOURCE and TARGET point sets.</li>
          <li>Normalize both point sets.</li>
          <li>Run the ClusterReg backend.</li>
          <li>Interpolate the displacement field back to the full SOURCE mesh.</li>
          <li>Denormalize the result into the TARGET frame.</li>
        </ol>

        <p>The CPU and optional GPU backends are imported from the bundled <code>fuzzy_lib</code>
        package. The bundled backend contains ClusterReg code distributed under AGPL-3.0.</p>
        """

        parent.acknowledgementText = (
            "Developed by Juan Antonio De Rus Arance at the Escola Tècnica Superior "
            "d'Enginyeria (ETSE-UV), Universitat de València, in the context of the "
            "Signal Processing & Acoustic Technology (SPAT) research group. "
            "Thanks to the 3D Slicer, SlicerMorph, VTK, NumPy, SciPy, Trimesh, and related "
            "open-source communities."
            "<p>This module uses code from ClusterReg by Zhao et al., "
            "\"Correspondence-Free Nonrigid Point Set Registration Using Unsupervised "
            "Clustering Analysis,\" CVPR 2024. "
            "ClusterReg is distributed under the AGPL-3.0 license.</p>"
        )
# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class ETSE_UV__FuzzyRegistrationWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()

        self.logic = ETSE_UV__FuzzyRegistrationLogic()

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        # Source selector
        self.srcSelector = slicer.qMRMLNodeComboBox()
        self.srcSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.srcSelector.selectNodeUponCreation = False
        self.srcSelector.addEnabled = False
        self.srcSelector.removeEnabled = False
        self.srcSelector.noneEnabled = False
        self.srcSelector.setMRMLScene(slicer.mrmlScene)
        self.srcSelector.setToolTip("Pick the SOURCE model (will be deformed).")
        form.addRow("Source model:", self.srcSelector)

        # Target selector
        self.tgtSelector = slicer.qMRMLNodeComboBox()
        self.tgtSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.tgtSelector.selectNodeUponCreation = False
        self.tgtSelector.addEnabled = False
        self.tgtSelector.removeEnabled = False
        self.tgtSelector.noneEnabled = False
        self.tgtSelector.setMRMLScene(slicer.mrmlScene)
        self.tgtSelector.setToolTip("Pick the TARGET model.")
        form.addRow("Target model:", self.tgtSelector)

        # Output option
        self.outputMode = qt.QComboBox()
        self.outputMode.addItems(["Create New Model", "Overwrite Source Model"])
        form.addRow("Output:", self.outputMode)

        # Custom name
        self.nameEdit = qt.QLineEdit()
        self.nameEdit.placeholderText = "ProjectedModel"
        form.addRow("Output name:", self.nameEdit)

        # Voxel size (downsample)
        self.voxelSpin = ctk.ctkDoubleSpinBox()
        self.voxelSpin.decimals = 3
        self.voxelSpin.singleStep = 0.01
        self.voxelSpin.minimum = 0.0
        self.voxelSpin.maximum = 100.0
        self.voxelSpin.value = 0.10  # matches typical demo.py default
        form.addRow("Voxel size (downsample):", self.voxelSpin)

        # Normals checkbox
        self.normalsCheck = qt.QCheckBox("Compute vertex normals")
        self.normalsCheck.checked = True
        form.addRow(self.normalsCheck)

        # GPU checkbox (only used if fuzzyclusterreg_gpu is importable)
        self.gpuCheck = qt.QCheckBox("Use GPU if available (requires fuzzyclusterreg_gpu)")
        self.gpuCheck.checked = False
        form.addRow(self.gpuCheck)

        # Print progress
        self.printCheck = qt.QCheckBox("Print progress")
        self.printCheck.checked = True
        form.addRow(self.printCheck)

        # Apply
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Run registration"
        self.layout.addWidget(self.applyButton)

        # Connections
        self.applyButton.clicked.connect(self.onApply)


        # ---------------------- Batch registration ----------------------
        batchBox = ctk.ctkCollapsibleButton()
        batchBox.text = "Batch registration"
        self.layout.addWidget(batchBox)
        bForm = qt.QFormLayout(batchBox)

        # Input directory
        rowIn = qt.QHBoxLayout()
        self.batchInputDirEdit = qt.QLineEdit("")
        btnIn = qt.QPushButton("Browse…")
        btnIn.clicked.connect(self.onBrowseBatchInput)
        rowIn.addWidget(self.batchInputDirEdit)
        rowIn.addWidget(btnIn)
        wIn = qt.QWidget()
        wIn.setLayout(rowIn)
        bForm.addRow("Target folder:", wIn)

        # Output directory
        rowOut = qt.QHBoxLayout()
        self.batchOutputDirEdit = qt.QLineEdit("")
        btnOut = qt.QPushButton("Browse…")
        btnOut.clicked.connect(self.onBrowseBatchOutput)
        rowOut.addWidget(self.batchOutputDirEdit)
        rowOut.addWidget(btnOut)
        wOut = qt.QWidget()
        wOut.setLayout(rowOut)
        bForm.addRow("Output folder:", wOut)

        # Run batch
        self.batchButton = qt.QPushButton("Run batch (use current Source as template)")
        self.batchButton.toolTip = (
            "Use the current Source model as template, register it against every mesh "
            "in the target folder, and save registered meshes to the output folder "
            "with the same filenames."
        )
        self.batchButton.clicked.connect(self.onBatchApply)
        bForm.addRow(self.batchButton)


        self.layout.addStretch(1)

    def onApply(self):
        srcNode = self.srcSelector.currentNode()
        tgtNode = self.tgtSelector.currentNode()
        if not srcNode or not tgtNode:
            slicer.util.errorDisplay("Please select both SOURCE and TARGET model nodes.")
            return

        outMode = self.outputMode.currentText
        outName = self.nameEdit.text.strip() or "ProjectedModel"
        voxel = float(self.voxelSpin.value)
        computeNormals = bool(self.normalsCheck.checked)
        useGPU = bool(self.gpuCheck.checked)
        verbose = bool(self.printCheck.checked)

        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            self.logic.register(
                sourceNode=srcNode,
                targetNode=tgtNode,
                outputMode=outMode,
                outputName=outName,
                voxel_size=voxel,
                compute_normals=computeNormals,
                use_gpu=useGPU,
                verbose=verbose
            )
        except Exception as e:
            slicer.app.restoreOverrideCursor()
            slicer.util.errorDisplay(f"Registration failed:\n{e}")
            raise
        finally:
            slicer.app.restoreOverrideCursor()

    def onBrowseBatchInput(self):
        d = qt.QFileDialog.getExistingDirectory(self.parent, "Select folder with target meshes")
        if d:
            self.batchInputDirEdit.setText(d)

    def onBrowseBatchOutput(self):
        d = qt.QFileDialog.getExistingDirectory(self.parent, "Select output folder for registered meshes")
        if d:
            self.batchOutputDirEdit.setText(d)

    def onBatchApply(self):
        templateNode = self.srcSelector.currentNode()
        if not templateNode:
            slicer.util.errorDisplay("Select a SOURCE model (template) in 'Source model' first.")
            return

        inputDir = self.batchInputDirEdit.text.strip()
        outputDir = self.batchOutputDirEdit.text.strip()

        if not inputDir or not os.path.isdir(inputDir):
            slicer.util.errorDisplay("Please choose a valid target folder.")
            return

        if not outputDir:
            slicer.util.errorDisplay("Please choose a valid output folder.")
            return
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir, exist_ok=True)

        voxel = float(self.voxelSpin.value)
        computeNormals = bool(self.normalsCheck.checked)
        useGPU = bool(self.gpuCheck.checked)
        verbose = bool(self.printCheck.checked)

        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            processed, saved, times, stats = self.logic.batchRegisterFolder(
                templateNode=templateNode,
                inputDir=inputDir,
                outputDir=outputDir,
                voxel_size=voxel,
                compute_normals=computeNormals,
                use_gpu=useGPU,
                verbose=verbose,
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Batch registration failed:\n{e}")
            raise
        finally:
            slicer.app.restoreOverrideCursor()

        if stats:
            msg = (
                f"Batch registration done.\n\n"
                f"Processed files: {processed}\n"
                f"Successfully saved: {saved}\n\n"
                f"Mean time: {stats['mean']:.2f} s\n"
                f"Min time:  {stats['min']:.2f} s\n"
                f"Max time:  {stats['max']:.2f} s\n"
                f"Std time:  {stats['std']:.2f} s"
            )
        else:
            msg = "Batch registration finished, but no meshes were processed or saved."

        slicer.util.infoDisplay(msg, "ETSE_UV__FuzzyRegistration batch")

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class ETSE_UV__FuzzyRegistrationLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()

        # fuzzy_lib is a subfolder next to this .py, Python can see it directly
        self._register_cpu = None
        self._register_gpu = None

        try:
            from fuzzy_lib.fuzzyclusterreg import fuzzy_cluster_reg
            self._register_cpu = fuzzy_cluster_reg
            print("[FuzzyReg] Imported CPU registrar from fuzzy_lib.fuzzyclusterreg")
        except Exception as e:
            slicer.util.warningDisplay(f"Could not import fuzzy_lib.fuzzyclusterreg (CPU): {e}")

        try:
            from fuzzy_lib.fuzzyclusterreg_gpu import fuzzy_cluster_reg_gpu
            self._register_gpu = fuzzy_cluster_reg_gpu
            print("[FuzzyReg] Imported GPU registrar from fuzzy_lib.fuzzyclusterreg_gpu")
        except Exception:
            # GPU optional
            self._register_gpu = None
            print("Could not import fuzzy_lib.fuzzyclusterreg_gpu")


    # --------------------- Public entry ---------------------
    def register(
        self,
        sourceNode: "vtkMRMLModelNode",
        targetNode: "vtkMRMLModelNode",
        outputMode: str,
        outputName: str,
        voxel_size: float = 0.10,
        compute_normals: bool = True,
        use_gpu: bool = False,
        verbose: bool = True
    ):
        if self._register_cpu is None and self._register_gpu is None:
            raise RuntimeError("fuzzyclusterreg.py not found. Place it next to this module or in fuzzy_lib.")

        src_pd = sourceNode.GetPolyData()
        tgt_pd = targetNode.GetPolyData()

        out_pd = self._compute_registration_polydata(
            src_pd=src_pd,
            tgt_pd=tgt_pd,
            voxel_size=voxel_size,
            compute_normals=compute_normals,
            use_gpu=use_gpu,
            verbose=verbose,
        )

        # Create or overwrite model node
        if outputMode == "Create New Model":
            outNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", outputName)
        else:
            outNode = sourceNode
            outNode.SetName(outputName)

        outNode.SetAndObservePolyData(out_pd)
        outNode.CreateDefaultDisplayNodes()
        if verbose:
            print(f"[FuzzyReg] Done. Output points: {out_pd.GetNumberOfPoints()}")

    
    def _compute_registration_polydata(
        self,
        src_pd: vtk.vtkPolyData,
        tgt_pd: vtk.vtkPolyData,
        voxel_size: float,
        compute_normals: bool,
        use_gpu: bool,
        verbose: bool,
    ) -> vtk.vtkPolyData:
        """Core registration: returns a vtkPolyData with TEMPLATE connectivity and deformed points."""

        # Fetch numpy arrays
        src_full = self.polydata_points_to_numpy(src_pd).astype(np.float32, copy=False)
        tgt_full = self.polydata_points_to_numpy(tgt_pd).astype(np.float32, copy=False)

        Ns = src_full.shape[0]
        Nt = tgt_full.shape[0]
        if verbose:
            print(f"[FuzzyReg] SOURCE points: {Ns} | TARGET points: {Nt}")

        # 1) Downsample BEFORE normalization
        if voxel_size > 0.0:
            src_ds, _ = self.voxel_downsample_with_indices(src_full, voxel_size)
            tgt_ds, _ = self.voxel_downsample_with_indices(tgt_full, voxel_size)
        else:
            src_ds = src_full
            tgt_ds = tgt_full

        if verbose:
            print(f"[FuzzyReg] Downsampled: SOURCE {len(src_ds)} | TARGET {len(tgt_ds)} | voxel={voxel_size}")

        # 2) Normalize like demo.py
        src_ds_n, pre_src = self.normalize_like_demo(src_ds)
        tgt_ds_n, pre_tgt = self.normalize_like_demo(tgt_ds)

        src_full_n = (src_full - pre_src["xd"]) / pre_src["xscale"]
        tgt_full_n = (tgt_full - pre_tgt["xd"]) / pre_tgt["xscale"]

        # 3) Register on downsampled normalized sets
        reg_fn = self._register_gpu if (use_gpu and self._register_gpu is not None) else self._register_cpu
        if reg_fn is None:
            reg_fn = self._register_cpu
        if verbose:
            print(f"[FuzzyReg] Using {'GPU' if reg_fn is self._register_gpu else 'CPU'} registrar")

        alpha, T_ds_n = reg_fn(
            src_ds_n.astype(np.float32, copy=False),
            tgt_ds_n.astype(np.float32, copy=False)
        )

        # 4) Displacement on ds → interpolate to full SOURCE (normalized space)
        D_ds_n = (T_ds_n - src_ds_n).astype(np.float32, copy=False)
        D_full_n = self.interpolate_displacement(src_ds_n, D_ds_n, src_full_n)

        # 5) Apply and denormalize with TARGET params (TARGET frame)
        S_full_n = src_full_n + D_full_n
        S_full_world = self.denormalize_like_demo(pre_tgt, S_full_n)

        # 6) Build new polydata with SOURCE connectivity
        out_pd = self.replace_polydata_points(src_pd, S_full_world)
        if compute_normals:
            out_pd = self.compute_normals_polydata(out_pd)

        return out_pd


    def batchRegisterFolder(
        self,
        templateNode: "vtkMRMLModelNode",
        inputDir: str,
        outputDir: str,
        voxel_size: float = 0.10,
        compute_normals: bool = True,
        use_gpu: bool = False,
        verbose: bool = True,
    ):
        """
        Batch mode:
          - templateNode: SOURCE template model (connectivity from here)
          - inputDir: folder with target meshes (.vtk, .vtp, .ply, .stl, .obj, ...)
          - outputDir: folder for registered meshes (same filenames as targets)

        Writes a CSV 'batch_stats.csv' in outputDir with:
          filename, processed, saved, time_sec, error

        Returns: (processed_count, saved_count, times_list, stats_dict)
        """
        if self._register_cpu is None and self._register_gpu is None:
            raise RuntimeError("fuzzyclusterreg.py not found. Place it in fuzzy_lib or next to this module.")

        if not os.path.isdir(inputDir):
            raise RuntimeError(f"Invalid input directory: {inputDir}")
        if not os.path.isdir(outputDir):
            os.makedirs(outputDir, exist_ok=True)

        template_pd = templateNode.GetPolyData()
        if template_pd is None or template_pd.GetNumberOfPoints() == 0:
            raise RuntimeError("Template (Source) model has no points.")

        exts = {".vtk", ".vtp", ".ply", ".stl", ".obj"}
        files = [
            f for f in os.listdir(inputDir)
            if os.path.isfile(os.path.join(inputDir, f))
            and os.path.splitext(f)[1].lower() in exts
        ]

        if verbose:
            print(f"[Batch] Template: {templateNode.GetName()} | Targets found: {len(files)}")

        processed = 0
        saved = 0
        times = []
        results = []  # per-file rows for CSV

        for fname in files:
            inPath = os.path.join(inputDir, fname)
            if verbose:
                print(f"[Batch] Processing {fname} ...")
            t0 = time.perf_counter()

            ok, tgtNode = slicer.util.loadModel(inPath, returnNode=True)
            if not ok or tgtNode is None:
                if verbose:
                    print(f"[Batch]   Failed to load {fname}, skipping.")
                results.append({
                    "filename": fname,
                    "processed": 0,
                    "saved": 0,
                    "time_sec": "",
                    "error": "load failed",
                })
                continue

            error_msg = ""
            dt = ""
            outNode = None

            try:
                tgt_pd = tgtNode.GetPolyData()
                out_pd = self._compute_registration_polydata(
                    src_pd=template_pd,
                    tgt_pd=tgt_pd,
                    voxel_size=voxel_size,
                    compute_normals=compute_normals,
                    use_gpu=use_gpu,
                    verbose=verbose,
                )

                outPath = os.path.join(outputDir, fname)
                outNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"reg_{fname}")
                outNode.SetAndObservePolyData(out_pd)
                outNode.CreateDefaultDisplayNodes()
                if not outNode.GetStorageNode():
                    outNode.AddDefaultStorageNode()
                st = outNode.GetStorageNode()
                st.SetFileName(outPath)
                if hasattr(st, "SetUseCompression"):
                    st.SetUseCompression(False)

                slicer.util.saveNode(outNode, outPath)
                t1 = time.perf_counter()
                dt = t1 - t0
                times.append(dt)
                saved += 1
                if verbose:
                    print(f"[Batch]   Saved {fname}  ({dt:.2f} s)")

            except Exception as e:
                error_msg = str(e)
                if verbose:
                    print(f"[Batch]   Error on {fname}: {e}")

            finally:
                processed += 1
                slicer.mrmlScene.RemoveNode(tgtNode)
                if outNode is not None:
                    slicer.mrmlScene.RemoveNode(outNode)
                slicer.app.processEvents()

                results.append({
                    "filename": fname,
                    "processed": 1,
                    "saved": 1 if dt != "" and error_msg == "" else 0,
                    "time_sec": f"{dt:.6f}" if dt != "" else "",
                    "error": error_msg,
                })

        stats = {}
        if times:
            arr = np.asarray(times, dtype=float)
            stats = {
                "count": int(len(times)),
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "std": float(arr.std(ddof=0)),
            }

        # ---- write CSV with per-file times + summary ----
        csv_path = os.path.join(outputDir, "batch_stats.csv")
        try:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "processed", "saved", "time_sec", "error"])
                for row in results:
                    writer.writerow([
                        row["filename"],
                        row["processed"],
                        row["saved"],
                        row["time_sec"],
                        row["error"],
                    ])
                writer.writerow([])
                writer.writerow(["SUMMARY"])
                writer.writerow(["count", stats.get("count", 0)])
                writer.writerow(["mean_time_sec", stats.get("mean", "")])
                writer.writerow(["min_time_sec", stats.get("min", "")])
                writer.writerow(["max_time_sec", stats.get("max", "")])
                writer.writerow(["std_time_sec", stats.get("std", "")])
            if verbose:
                print(f"[Batch] Wrote stats CSV: {csv_path}")
        except Exception as e:
            if verbose:
                print(f"[Batch] Could not write CSV stats: {e}")

        if verbose and stats:
            print(f"[Batch] Done. Processed={processed}, Saved={saved}, Mean time={stats.get('mean', float('nan')):.2f} s")

        return processed, saved, times, stats

    # --------------------- VTK/Numpy helpers ---------------------
    def polydata_points_to_numpy(self, polydata: vtk.vtkPolyData) -> np.ndarray:
        import vtkmodules.util.numpy_support as vtk_np
        return vtk_np.vtk_to_numpy(polydata.GetPoints().GetData())

    def replace_polydata_points(self, source_polydata: vtk.vtkPolyData, new_points_xyz: np.ndarray) -> vtk.vtkPolyData:
        """Deep-copy source polydata (to keep polys/cells), replace points."""
        import vtkmodules.util.numpy_support as vtk_np
        new_pd = vtk.vtkPolyData()
        new_pd.DeepCopy(source_polydata)

        pts = vtk.vtkPoints()
        pts.SetData(vtk_np.numpy_to_vtk(new_points_xyz.astype(np.float64, copy=False), deep=True))
        new_pd.SetPoints(pts)

        # If there were point-data arrays that no longer match, drop them:
        if new_pd.GetPointData():
            # keep only normals if present after recompute; otherwise clear
            new_pd.GetPointData().Initialize()

        return new_pd

    def compute_normals_polydata(self, polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
        nf = vtk.vtkPolyDataNormals()
        nf.SplittingOff()
        nf.ConsistencyOn()
        nf.ComputePointNormalsOn()
        nf.SetInputData(polydata)
        nf.Update()
        return nf.GetOutput()

    # --------------------- demo.py compatible normalize/denormalize ---------------------
    def normalize_like_demo(self, X: np.ndarray):
        """
        demo.normalize:
            xd = mean(X)
            Xc = X - xd
            xscale = sqrt(sum(Xc**2) / N)
            Xn = Xc / xscale
        returns Xn, {'xd': xd, 'xscale': xscale}
        """
        N = X.shape[0]
        xd = np.mean(X, axis=0, keepdims=True)
        Xc = X - xd
        xscale = float(np.sqrt(np.sum(Xc ** 2) / max(N, 1)))
        xscale = xscale if xscale > 1e-12 else 1.0
        return Xc / xscale, {"xd": xd, "xscale": xscale}

    def denormalize_like_demo(self, pre: dict, Xn: np.ndarray) -> np.ndarray:
        return Xn * pre["xscale"] + pre["xd"]

    # --------------------- Downsample + Interpolation ---------------------
    def voxel_downsample_with_indices(self, pts: np.ndarray, step: float):
        """
        Minimal voxel grid (keeps first point per voxel). Returns (points, kept_indices).
        """
        if step <= 0.0 or pts.size == 0:
            return pts, np.arange(len(pts), dtype=np.int64)

        keys = np.floor(pts / float(step) + 0.5).astype(np.int64)
        seen = {}
        kept = []
        for i, k in enumerate(map(tuple, keys)):
            if k not in seen:
                seen[k] = i
                kept.append(i)
        kept = np.array(kept, dtype=np.int64)
        return pts[kept], kept

    def interpolate_displacement(self, X_land: np.ndarray, D_land: np.ndarray, X_query: np.ndarray) -> np.ndarray:
        """
        Interpolate 3D displacement using 3-NN inverse-distance weighting (stable & smooth).
        Prefers SciPy cKDTree; falls back to VTK locator.
        """
        if X_query.shape[0] == 0:
            return np.zeros_like(X_query, dtype=np.float32)

        k = min(3, X_land.shape[0])

        if _HAS_SCIPY:
            tree = cKDTree(X_land)
            dists, idxs = tree.query(X_query, k=k, workers=-1)
            # Ensure 2D
            if k == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]
            eps = 1e-12
            w = 1.0 / (dists + eps)
            w_sum = np.sum(w, axis=1, keepdims=True)
            w = w / w_sum
            Dq = np.sum(D_land[idxs] * w[..., None], axis=1)
            return Dq.astype(np.float32, copy=False)

        # Fallback: VTK locator (slower)
        loc = vtk.vtkStaticPointLocator()
        vtkPts = vtk.vtkPoints()
        vtkPts.SetData(vtk.util.numpy_support.numpy_to_vtk(X_land.astype(np.float64), deep=True))
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtkPts)
        loc.SetDataSet(poly)
        loc.BuildLocator()

        Dq = np.zeros_like(X_query, dtype=np.float64)
        ids = vtk.vtkIdList()
        for i in range(X_query.shape[0]):
            loc.FindClosestNPoints(k, X_query[i], ids)
            idx = [ids.GetId(j) for j in range(ids.GetNumberOfIds())]
            neigh = X_land[idx]
            d = np.linalg.norm(neigh - X_query[i], axis=1)
            w = 1.0 / (d + 1e-12)
            w = w / np.sum(w)
            Dq[i] = (D_land[idx] * w[:, None]).sum(axis=0)
            if i % 10000 == 0:
                slicer.app.processEvents()
        return Dq.astype(np.float32, copy=False)
