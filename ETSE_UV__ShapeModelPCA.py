# ETSE_UV__ShapeModelPCA.py
# 3D Slicer scripted module
#
# Build a simple PCA shape model from a folder of registered meshes.
# - Loads .vtk/.vtp/.ply/.stl/.obj meshes with identical vertex order.
# - Computes mean shape and PCA components.
# - Shows mean/generated shape in the Slicer scene.
# - Provides beta sliders in standard-deviation units.
# - Shows/export explained variance metrics.
#
# This is intentionally self-contained: the ShapeModelPCA logic lives in this file.

import os
import glob
import csv
import traceback

import numpy as np
import vtk
import vtkmodules.util.numpy_support as vtk_np
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *


# -----------------------------------------------------------------------------
# Optional dependency installation/check
# -----------------------------------------------------------------------------
try:
    from Resources.ETSE_UV__Dependencies import ensure_packages
    ensure_packages(
        [
            ("sklearn", "scikit-learn"),
            ("joblib", "joblib"),
        ],
        interactive=False,
        module_name="ETSE-UV Shape Model PCA",
    )
except Exception as e:
    # Keep module loadable even if the helper is not available; imports below will
    # still raise a clear error when the user tries to build/load a model.
    print(f"[ShapeModelPCA] Dependency helper not available or failed: {e}")

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

try:
    import joblib
except Exception:
    joblib = None


# =============================================================================
# Module metadata
# =============================================================================
class ETSE_UV__ShapeModelPCA(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Shape Model PCA"
        parent.categories = ["ETSE_UV"]
        parent.dependencies = []
        parent.contributors = ["ETSE-UV"]
        parent.helpText = (
            "Build and inspect a PCA statistical shape model from registered meshes.\n\n"
            "The meshes must have one-to-one vertex correspondence: same number of "
            "points and the same point order. The first mesh supplies the output topology.\n\n"
            "Workflow:\n"
            "  1) Select a folder with registered meshes (.vtk/.vtp/.ply/.stl/.obj).\n"
            "  2) Choose the number of PCA components and preprocessing.\n"
            "  3) Build PCA model.\n"
            "  4) Move beta sliders to synthesize shapes in standard-deviation units.\n"
            "  5) Save the generated mesh or save/load the PCA model (.joblib)."
        )
        parent.acknowledgementText = "Developed for the ETSE-UV correspondence tools."


# =============================================================================
# Widget
# =============================================================================
class ETSE_UV__ShapeModelPCAWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = ETSE_UV__ShapeModelPCALogic()
        self._sliderWidgets = []
        self._updatingSliders = False

        # ------------------------------------------------------------------
        # Build / load model
        # ------------------------------------------------------------------
        buildBox = ctk.ctkCollapsibleButton()
        buildBox.text = "1) Build PCA shape model"
        buildBox.collapsed = False
        self.layout.addWidget(buildBox)
        buildForm = qt.QFormLayout(buildBox)

        self.trainingDirButton = ctk.ctkDirectoryButton()
        self.trainingDirButton.setToolTip("Folder containing registered meshes with matching vertex order.")
        self.trainingDirButton.setMaximumWidth(500)
        buildForm.addRow("Training mesh folder:", self.trainingDirButton)

        self.extEdit = qt.QLineEdit("vtk vtp ply obj stl")
        self.extEdit.setToolTip("Space-separated mesh extensions to include.")
        buildForm.addRow("Extensions:", self.extEdit)

        self.recursiveCheck = qt.QCheckBox("Search subfolders recursively")
        self.recursiveCheck.checked = True
        buildForm.addRow(self.recursiveCheck)

        self.componentsSpin = qt.QSpinBox()
        self.componentsSpin.setRange(1, 500)
        self.componentsSpin.setValue(30)
        self.componentsSpin.setToolTip("Requested number of PCA components. It will be clamped to the data rank.")
        buildForm.addRow("PCA components:", self.componentsSpin)

        self.preprocessCombo = qt.QComboBox()
        self.preprocessCombo.addItems([
            "robust",
            "center",
            "zscore",
            "maxabs",
            "global_zscore",
            "global_maxabs",
        ])
        self.preprocessCombo.setCurrentText("robust")
        self.preprocessCombo.setToolTip(
            "Preprocessing applied to flattened deviations before PCA. "
            "robust = per-feature IQR scaling, matching the notebook default."
        )
        buildForm.addRow("PCA preprocess:", self.preprocessCombo)

        self.centerMeshesCheck = qt.QCheckBox("Center each mesh by its centroid before PCA")
        self.centerMeshesCheck.checked = True
        self.centerMeshesCheck.setToolTip("Matches your notebook behavior: every mesh is centered before PCA.")
        buildForm.addRow(self.centerMeshesCheck)

        rowBuild = qt.QHBoxLayout()
        self.buildButton = qt.QPushButton("Build PCA model")
        self.loadModelButton = qt.QPushButton("Load PCA model…")
        self.saveModelButton = qt.QPushButton("Save PCA model…")
        rowBuild.addWidget(self.buildButton)
        rowBuild.addWidget(self.loadModelButton)
        rowBuild.addWidget(self.saveModelButton)
        rowBuildWidget = qt.QWidget()
        rowBuildWidget.setLayout(rowBuild)
        buildForm.addRow(rowBuildWidget)

        self.buildButton.clicked.connect(self.onBuildModel)
        self.loadModelButton.clicked.connect(self.onLoadPcaModel)
        self.saveModelButton.clicked.connect(self.onSavePcaModel)

        # ------------------------------------------------------------------
        # Model display / beta sliders
        # ------------------------------------------------------------------
        sliderBox = ctk.ctkCollapsibleButton()
        sliderBox.text = "2) Inspect PCA components"
        sliderBox.collapsed = False
        self.layout.addWidget(sliderBox)
        sliderLayout = qt.QVBoxLayout(sliderBox)

        betaOptions = qt.QGroupBox("Beta slider options")
        betaForm = qt.QFormLayout(betaOptions)

        self.sliderMinSpin = qt.QDoubleSpinBox()
        self.sliderMinSpin.setRange(-50.0, 0.0)
        self.sliderMinSpin.setDecimals(1)
        self.sliderMinSpin.setSingleStep(0.5)
        self.sliderMinSpin.setValue(-3.0)
        betaForm.addRow("Minimum std:", self.sliderMinSpin)

        self.sliderMaxSpin = qt.QDoubleSpinBox()
        self.sliderMaxSpin.setRange(0.0, 50.0)
        self.sliderMaxSpin.setDecimals(1)
        self.sliderMaxSpin.setSingleStep(0.5)
        self.sliderMaxSpin.setValue(3.0)
        betaForm.addRow("Maximum std:", self.sliderMaxSpin)

        self.autoUpdateCheck = qt.QCheckBox("Auto-update generated mesh when sliders move")
        self.autoUpdateCheck.checked = True
        betaForm.addRow(self.autoUpdateCheck)

        self.showMeanCheck = qt.QCheckBox("Show/update mean shape model")
        self.showMeanCheck.checked = True
        betaForm.addRow(self.showMeanCheck)

        sliderLayout.addWidget(betaOptions)

        self.sliderScrollArea = qt.QScrollArea()
        self.sliderScrollArea.setWidgetResizable(True)
        self.sliderScrollArea.setMinimumHeight(260)
        self.sliderWidget = qt.QWidget()
        self.sliderForm = qt.QFormLayout(self.sliderWidget)
        self.sliderScrollArea.setWidget(self.sliderWidget)
        sliderLayout.addWidget(self.sliderScrollArea)

        btnRow = qt.QHBoxLayout()
        self.updateShapeButton = qt.QPushButton("Update generated shape")
        self.resetSlidersButton = qt.QPushButton("Reset betas")
        self.saveShapeButton = qt.QPushButton("Save generated mesh…")
        btnRow.addWidget(self.updateShapeButton)
        btnRow.addWidget(self.resetSlidersButton)
        btnRow.addWidget(self.saveShapeButton)
        btnRowWidget = qt.QWidget()
        btnRowWidget.setLayout(btnRow)
        sliderLayout.addWidget(btnRowWidget)

        self.updateShapeButton.clicked.connect(self.onUpdateShape)
        self.resetSlidersButton.clicked.connect(self.onResetSliders)
        self.saveShapeButton.clicked.connect(self.onSaveGeneratedShape)

        # ------------------------------------------------------------------
        # Project an existing model into the PCA space
        # ------------------------------------------------------------------
        projectBox = ctk.ctkCollapsibleButton()
        projectBox.text = "3) Optional: project an existing model to PCA betas"
        projectBox.collapsed = True
        self.layout.addWidget(projectBox)
        projectForm = qt.QFormLayout(projectBox)

        self.projectModelSelector = slicer.qMRMLNodeComboBox()
        self.projectModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.projectModelSelector.selectNodeUponCreation = True
        self.projectModelSelector.addEnabled = False
        self.projectModelSelector.removeEnabled = False
        self.projectModelSelector.noneEnabled = True
        self.projectModelSelector.setMRMLScene(slicer.mrmlScene)
        self.projectModelSelector.setToolTip("Model with the same vertex count/order as the PCA training meshes.")
        projectForm.addRow("Input model:", self.projectModelSelector)

        self.projectButton = qt.QPushButton("Set sliders from selected model")
        projectForm.addRow(self.projectButton)
        self.projectButton.clicked.connect(self.onProjectSelectedModel)

        self.projectStatusLabel = qt.QLabel("")
        self.projectStatusLabel.setWordWrap(True)
        projectForm.addRow("Projection:", self.projectStatusLabel)

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        metricsBox = ctk.ctkCollapsibleButton()
        metricsBox.text = "4) Metrics / explained variance"
        metricsBox.collapsed = False
        self.layout.addWidget(metricsBox)
        metricsLayout = qt.QVBoxLayout(metricsBox)

        self.summaryText = qt.QPlainTextEdit()
        self.summaryText.readOnly = True
        self.summaryText.setMinimumHeight(110)
        metricsLayout.addWidget(self.summaryText)

        self.varianceTable = qt.QTableWidget(0, 5)
        self.varianceTable.setHorizontalHeaderLabels([
            "Component",
            "Explained ratio",
            "Cumulative ratio",
            "Explained variance",
            "Std dev",
        ])
        self.varianceTable.setMinimumHeight(220)
        metricsLayout.addWidget(self.varianceTable)

        self.exportMetricsButton = qt.QPushButton("Export explained variance CSV…")
        metricsLayout.addWidget(self.exportMetricsButton)
        self.exportMetricsButton.clicked.connect(self.onExportMetrics)

        self.layout.addStretch(1)
        self._setInitialState()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _setInitialState(self):
        self.summaryText.setPlainText("No PCA model loaded yet.")
        self._rebuildSliders(0)

    def _meshExtensions(self):
        return [e.strip().lower().lstrip(".") for e in self.extEdit.text.split() if e.strip()]

    def _setBusy(self, busy):
        if busy:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        else:
            slicer.app.restoreOverrideCursor()
        slicer.app.processEvents()

    def _clearSliderLayout(self):
        while self.sliderForm.rowCount() > 0:
            self.sliderForm.removeRow(0)
        self._sliderWidgets = []

    def _rebuildSliders(self, n_components):
        self._clearSliderLayout()
        if n_components <= 0:
            lab = qt.QLabel("Build or load a PCA model to create sliders.")
            lab.setWordWrap(True)
            self.sliderForm.addRow(lab)
            return

        var = self.logic.explained_variance_ratio_
        cum = self.logic.cumulative_explained_variance_
        sdev = self.logic.std_devs
        mn = float(self.sliderMinSpin.value)
        mx = float(self.sliderMaxSpin.value)
        if mx <= mn:
            mx = mn + 1.0

        for i in range(n_components):
            labelText = f"β{i}"
            if var is not None and i < len(var):
                labelText += f"  ({100.0 * float(var[i]):.2f}%, cum {100.0 * float(cum[i]):.2f}%)"
            if sdev is not None and i < len(sdev):
                labelText += f"  σ={float(sdev[i]):.4g}"

            slider = ctk.ctkSliderWidget()
            slider.minimum = mn
            slider.maximum = mx
            slider.value = 0.0
            slider.singleStep = 0.1
            slider.decimals = 2
            slider.tracking = True
            slider.setToolTip("Beta value in PCA standard-deviation units.")
            slider.valueChanged.connect(self.onSliderChanged)
            self._sliderWidgets.append(slider)
            self.sliderForm.addRow(labelText, slider)

    def _currentBeta(self):
        return np.array([float(s.value) for s in self._sliderWidgets], dtype=float)

    def _setBeta(self, beta):
        self._updatingSliders = True
        try:
            beta = np.asarray(beta, dtype=float)
            for i, s in enumerate(self._sliderWidgets):
                s.value = float(beta[i]) if i < len(beta) else 0.0
        finally:
            self._updatingSliders = False

    def _refreshOutputsAfterModelChange(self):
        self._rebuildSliders(self.logic.n_pca_components)
        self._fillMetrics()
        if self.showMeanCheck.checked:
            self.logic.show_mean_shape()
        self.onUpdateShape()

    def _fillMetrics(self):
        summary = self.logic.model_summary_text()
        self.summaryText.setPlainText(summary)

        ratios = self.logic.explained_variance_ratio_
        if ratios is None:
            self.varianceTable.setRowCount(0)
            return

        cum = self.logic.cumulative_explained_variance_
        ev = self.logic.explained_variance_
        sd = self.logic.std_devs
        n = len(ratios)
        self.varianceTable.setRowCount(n)
        for i in range(n):
            vals = [
                str(i),
                f"{float(ratios[i]):.8f}",
                f"{float(cum[i]):.8f}",
                f"{float(ev[i]):.8g}",
                f"{float(sd[i]):.8g}",
            ]
            for j, val in enumerate(vals):
                self.varianceTable.setItem(i, j, qt.QTableWidgetItem(val))
        self.varianceTable.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def onBuildModel(self):
        folder = self.trainingDirButton.directory
        if not folder or not os.path.isdir(folder):
            slicer.util.errorDisplay("Please choose a valid training mesh folder.")
            return

        if PCA is None:
            slicer.util.errorDisplay("scikit-learn is not available in this Slicer Python environment.")
            return

        try:
            self._setBusy(True)
            self.logic.build_from_folder(
                folder=folder,
                n_pca_components=int(self.componentsSpin.value),
                pca_preprocess=str(self.preprocessCombo.currentText),
                center_meshes=bool(self.centerMeshesCheck.checked),
                extensions=self._meshExtensions(),
                recursive=bool(self.recursiveCheck.checked),
                debug=True,
            )
            self._refreshOutputsAfterModelChange()
            slicer.util.infoDisplay(
                f"PCA model built from {self.logic.n_shapes} meshes.\n"
                f"Components: {self.logic.n_pca_components}\n"
                f"Total explained variance: {100.0 * self.logic.total_explained_variance():.3f}%",
                windowTitle="ETSE-UV Shape Model PCA",
            )
        except Exception as e:
            traceback.print_exc()
            slicer.util.errorDisplay(f"Could not build PCA model:\n{e}")
        finally:
            self._setBusy(False)

    def onLoadPcaModel(self):
        if joblib is None:
            slicer.util.errorDisplay("joblib is not available in this Slicer Python environment.")
            return
        path = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Load PCA shape model",
            "",
            "Joblib PCA models (*.joblib);;All files (*)",
        )
        if not path:
            return
        try:
            self._setBusy(True)
            self.logic.load_pca_model(path)
            self.componentsSpin.setValue(int(self.logic.n_pca_components))
            if self.logic.pca_preprocess:
                idx = self.preprocessCombo.findText(self.logic.pca_preprocess)
                if idx >= 0:
                    self.preprocessCombo.setCurrentIndex(idx)
            self.centerMeshesCheck.checked = bool(self.logic.center_meshes)
            self._refreshOutputsAfterModelChange()
            slicer.util.infoDisplay(f"Loaded PCA model:\n{path}")
        except Exception as e:
            traceback.print_exc()
            slicer.util.errorDisplay(f"Could not load PCA model:\n{e}")
        finally:
            self._setBusy(False)

    def onSavePcaModel(self):
        if joblib is None:
            slicer.util.errorDisplay("joblib is not available in this Slicer Python environment.")
            return
        if not self.logic.initialized:
            slicer.util.errorDisplay("Build or load a PCA model first.")
            return
        path = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save PCA shape model",
            f"shapePCA__K{self.logic.n_pca_components}__prep_{self.logic.pca_preprocess}.joblib",
            "Joblib PCA models (*.joblib);;All files (*)",
        )
        if not path:
            return
        if not str(path).lower().endswith(".joblib"):
            path += ".joblib"
        try:
            saved = self.logic.save_pca_model(path)
            slicer.util.infoDisplay(f"Saved PCA model:\n{saved}")
        except Exception as e:
            traceback.print_exc()
            slicer.util.errorDisplay(f"Could not save PCA model:\n{e}")

    def onSliderChanged(self, *args):
        if self._updatingSliders:
            return
        if bool(self.autoUpdateCheck.checked) and self.logic.initialized:
            self.onUpdateShape()

    def onUpdateShape(self):
        if not self.logic.initialized:
            return
        try:
            beta = self._currentBeta()
            self.logic.show_generated_shape(beta)
        except Exception as e:
            traceback.print_exc()
            slicer.util.errorDisplay(f"Could not update generated shape:\n{e}")

    def onResetSliders(self):
        if not self._sliderWidgets:
            return
        self._setBeta(np.zeros(len(self._sliderWidgets), dtype=float))
        self.onUpdateShape()

    def onSaveGeneratedShape(self):
        if not self.logic.generatedModelNode:
            slicer.util.errorDisplay("No generated model node exists yet. Update the generated shape first.")
            return
        path = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save generated mesh",
            "PCA_generated_shape.vtk",
            "Model files (*.vtk *.vtp *.stl *.ply *.obj);;All files (*)",
        )
        if not path:
            return
        try:
            ok = slicer.util.saveNode(self.logic.generatedModelNode, path)
            if ok:
                slicer.util.infoDisplay(f"Saved generated mesh:\n{path}")
            else:
                slicer.util.errorDisplay("Slicer could not save the generated mesh.")
        except Exception as e:
            traceback.print_exc()
            slicer.util.errorDisplay(f"Could not save generated mesh:\n{e}")

    def onProjectSelectedModel(self):
        if not self.logic.initialized:
            slicer.util.errorDisplay("Build or load a PCA model first.")
            return
        node = self.projectModelSelector.currentNode()
        if not node:
            slicer.util.errorDisplay("Select a model to project into PCA space.")
            return
        try:
            beta, rms = self.logic.project_model_to_beta(node)
            self._setBeta(beta)
            self.onUpdateShape()
            self.projectStatusLabel.setText(
                f"Set {len(beta)} beta sliders from '{node.GetName()}'. Reconstruction RMS: {rms:.6g}"
            )
        except Exception as e:
            traceback.print_exc()
            slicer.util.errorDisplay(f"Could not project selected model:\n{e}")

    def onExportMetrics(self):
        if not self.logic.initialized:
            slicer.util.errorDisplay("Build or load a PCA model first.")
            return
        path = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Export PCA explained variance CSV",
            "shapePCA_explained_variance.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        if not str(path).lower().endswith(".csv"):
            path += ".csv"
        try:
            self.logic.export_variance_csv(path)
            slicer.util.infoDisplay(f"Exported PCA metrics:\n{path}")
        except Exception as e:
            traceback.print_exc()
            slicer.util.errorDisplay(f"Could not export PCA metrics:\n{e}")


# =============================================================================
# Logic
# =============================================================================
class ETSE_UV__ShapeModelPCALogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.reset()

    def reset(self):
        self.initialized = False
        self.data_folder_path = None
        self.mesh_paths = []
        self.n_shapes = 0
        self.n_points = 0
        self.n_features = 0
        self.n_pca_components = 0
        self.pca_preprocess = "robust"
        self.center_meshes = True
        self.eps = 1e-12

        self.figure_coordinates = None      # (N, V, 3), after optional centering
        self.mean_shape = None              # (V, 3)
        self.shape_deviations = None        # (N, V, 3)
        self.mean_data = None               # (D,)
        self.feature_scale = None           # (D,)
        self.pca = None
        self.beta_parameters_flat = None    # (D, K)
        self.std_devs = None                # (K,)
        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ = None
        self.explained_variance_ = None

        self.reference_polydata = None
        self.topology_cells = None
        self.generated_shape = None
        self.meanModelNode = None
        self.generatedModelNode = None
        self._last_saved_model_path = None

    # ------------------------------------------------------------------
    # File and VTK helpers
    # ------------------------------------------------------------------
    def find_mesh_files(self, folder, extensions=None, recursive=True):
        if not folder or not os.path.isdir(folder):
            raise RuntimeError(f"Folder not found: {folder}")
        if extensions is None or len(extensions) == 0:
            extensions = ["vtk", "vtp", "ply", "obj", "stl"]
        extensions = [e.lower().lstrip(".") for e in extensions]

        files = []
        if recursive:
            for ext in extensions:
                files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
                files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext.upper()}"), recursive=True))
        else:
            for ext in extensions:
                files.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
                files.extend(glob.glob(os.path.join(folder, f"*.{ext.upper()}")))

        # Unique, deterministic order
        files = sorted(set(files))
        return files

    def read_polydata(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".vtk":
            reader = vtk.vtkPolyDataReader()
        elif ext == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
        elif ext == ".ply":
            reader = vtk.vtkPLYReader()
        elif ext == ".stl":
            reader = vtk.vtkSTLReader()
        elif ext == ".obj":
            reader = vtk.vtkOBJReader()
        else:
            raise RuntimeError(f"Unsupported mesh extension '{ext}' for file:\n{path}")

        reader.SetFileName(path)
        reader.Update()
        out = reader.GetOutput()
        if out is None:
            raise RuntimeError(f"Could not read mesh file:\n{path}")

        if out.IsA("vtkPolyData"):
            poly = vtk.vtkPolyData()
            poly.DeepCopy(out)
            return poly

        geom = vtk.vtkGeometryFilter()
        geom.SetInputData(out)
        geom.Update()
        poly = vtk.vtkPolyData()
        poly.DeepCopy(geom.GetOutput())
        return poly

    def polydata_to_points_array(self, polydata):
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            raise RuntimeError("Mesh has no points.")
        pts = polydata.GetPoints()
        arr = vtk_np.vtk_to_numpy(pts.GetData()).astype(np.float64, copy=True)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise RuntimeError(f"Unexpected point array shape: {arr.shape}")
        return arr

    def polydata_cells_to_list(self, polydata):
        cells = []
        if polydata is None:
            return cells
        for cid in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(cid)
            ids = [int(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())]
            if len(ids) >= 1:
                cells.append(ids)
        return cells

    def cells_list_to_cellarray(self, cells):
        cellArray = vtk.vtkCellArray()
        if not cells:
            return cellArray
        for ids in cells:
            ids = [int(i) for i in ids]
            cellArray.InsertNextCell(len(ids))
            for pid in ids:
                cellArray.InsertCellPoint(pid)
        return cellArray

    def numpy_points_to_polydata(self, points, topology_cells=None):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError(f"Expected points with shape (V, 3), got {points.shape}")

        vtkPoints = vtk.vtkPoints()
        vtkArr = vtk_np.numpy_to_vtk(points, deep=True)
        vtkPoints.SetData(vtkArr)

        poly = vtk.vtkPolyData()
        poly.SetPoints(vtkPoints)

        cells = topology_cells if topology_cells is not None else self.topology_cells
        if cells:
            cellArray = self.cells_list_to_cellarray(cells)
            # In these tools the training meshes are surface meshes, so store cells as polys.
            poly.SetPolys(cellArray)
        else:
            verts = vtk.vtkCellArray()
            for i in range(points.shape[0]):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)
            poly.SetVerts(verts)

        poly.Modified()
        return poly

    def create_or_update_model_node(self, node, polydata, name, color=(1.0, 0.82, 0.05), opacity=1.0):
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
            node.CreateDefaultDisplayNodes()
        node.SetName(name)
        node.SetAndObservePolyData(polydata)
        node.CreateDefaultDisplayNodes()
        dn = node.GetDisplayNode()
        if dn:
            dn.SetColor(float(color[0]), float(color[1]), float(color[2]))
            dn.SetOpacity(float(opacity))
            dn.SetVisibility(True)
            try:
                dn.SetBackfaceCulling(False)
            except Exception:
                pass
        return node

    # ------------------------------------------------------------------
    # PCA preprocessing and modeling
    # ------------------------------------------------------------------
    def _fit_preprocess(self, data):
        data = np.asarray(data, dtype=np.float64)
        self.mean_data = np.mean(data, axis=0)
        mode = (self.pca_preprocess or "robust").lower()

        def safe(x):
            return np.maximum(np.asarray(x, dtype=np.float64), self.eps)

        if mode == "center":
            self.feature_scale = np.ones_like(self.mean_data)
            return

        if mode == "global_zscore":
            s = max(float(np.std(data)), self.eps)
            self.feature_scale = np.full_like(self.mean_data, s)
            return

        if mode == "global_maxabs":
            s = max(float(np.max(np.abs(data - self.mean_data))), self.eps)
            self.feature_scale = np.full_like(self.mean_data, s)
            return

        if mode == "zscore":
            self.feature_scale = safe(np.std(data, axis=0))
            return

        if mode == "maxabs":
            self.feature_scale = safe(np.max(np.abs(data - self.mean_data), axis=0))
            return

        if mode == "robust":
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            self.feature_scale = safe(q75 - q25)
            return

        raise RuntimeError(
            "pca_preprocess must be one of: center, zscore, maxabs, robust, "
            "global_zscore, global_maxabs"
        )

    def _encode_for_pca(self, data):
        return (np.asarray(data, dtype=np.float64) - self.mean_data) / self.feature_scale

    def _decode_from_pca(self, data_norm):
        return self.mean_data + np.asarray(data_norm, dtype=np.float64) * self.feature_scale

    def build_from_folder(
        self,
        folder,
        n_pca_components=30,
        pca_preprocess="robust",
        center_meshes=True,
        extensions=None,
        recursive=True,
        debug=True,
    ):
        if PCA is None:
            raise RuntimeError("scikit-learn is not available. Install scikit-learn in Slicer's Python.")

        paths = self.find_mesh_files(folder, extensions=extensions, recursive=recursive)
        if len(paths) < 2:
            raise RuntimeError(f"Need at least 2 mesh files to build PCA. Found {len(paths)} in:\n{folder}")

        self.reset()
        self.data_folder_path = folder
        self.mesh_paths = paths
        self.pca_preprocess = str(pca_preprocess or "robust")
        self.center_meshes = bool(center_meshes)

        if debug:
            print("[ShapeModelPCA] ---- Loading mesh coordinates ----")
            print(f"[ShapeModelPCA] Folder: {folder}")
            print(f"[ShapeModelPCA] Mesh files: {len(paths)}")

        coords_list = []
        n_points_expected = None
        first_poly = None
        first_cells = None

        for i, path in enumerate(paths):
            poly = self.read_polydata(path)
            pts = self.polydata_to_points_array(poly)
            if n_points_expected is None:
                n_points_expected = pts.shape[0]
                first_poly = vtk.vtkPolyData()
                first_poly.DeepCopy(poly)
                first_cells = self.polydata_cells_to_list(poly)
            elif pts.shape[0] != n_points_expected:
                raise RuntimeError(
                    "All meshes must have the same number of points.\n"
                    f"First mesh: {n_points_expected} points\n"
                    f"This mesh:  {pts.shape[0]} points\n"
                    f"File: {path}"
                )

            if self.center_meshes:
                pts = pts - np.mean(pts, axis=0, keepdims=True)
            coords_list.append(pts)

        self.figure_coordinates = np.asarray(coords_list, dtype=np.float64)  # (N,V,3)
        self.n_shapes = int(self.figure_coordinates.shape[0])
        self.n_points = int(self.figure_coordinates.shape[1])
        self.n_features = int(self.n_points * 3)
        self.reference_polydata = first_poly
        self.topology_cells = first_cells

        if debug:
            print(f"[ShapeModelPCA] figure_coordinates shape: {self.figure_coordinates.shape}")
            print("[ShapeModelPCA] ---- Mean shape ----")

        self.mean_shape = np.mean(self.figure_coordinates, axis=0)
        self.shape_deviations = self.figure_coordinates - self.mean_shape[np.newaxis, :, :]

        flat_deviations = self.shape_deviations.reshape(self.n_shapes, -1)
        self._fit_preprocess(flat_deviations)
        data_norm = self._encode_for_pca(flat_deviations)

        max_components = int(min(data_norm.shape[0], data_norm.shape[1]))
        requested = int(n_pca_components)
        n_components = max(1, min(requested, max_components))
        if requested != n_components:
            print(f"[ShapeModelPCA] Requested {requested} components, clamped to {n_components}.")

        if debug:
            print("[ShapeModelPCA] ---- PCA ----")
            print(f"[ShapeModelPCA] pca_preprocess: {self.pca_preprocess}")
            print(f"[ShapeModelPCA] flat deviations: {flat_deviations.shape}")

        self.pca = PCA(n_components=n_components)
        self.pca.fit(data_norm)

        self.n_pca_components = int(n_components)
        self.beta_parameters_flat = self.pca.components_.T.astype(np.float64, copy=True)
        self.explained_variance_ = np.asarray(self.pca.explained_variance_, dtype=np.float64)
        self.explained_variance_ratio_ = np.asarray(self.pca.explained_variance_ratio_, dtype=np.float64)
        self.cumulative_explained_variance_ = np.cumsum(self.explained_variance_ratio_)
        self.std_devs = np.sqrt(self.explained_variance_)
        self.initialized = True

        if debug:
            print("--------------------------------------------------------")
            print("[ShapeModelPCA] Model built")
            print(f"[ShapeModelPCA] explained variance sum: {self.total_explained_variance():.8f}")
            print(f"[ShapeModelPCA] mean_shape       : {self.mean_shape.shape}")
            print(f"[ShapeModelPCA] mean_data        : {self.mean_data.shape}")
            print(f"[ShapeModelPCA] feature_scale    : {self.feature_scale.shape}")
            print(f"[ShapeModelPCA] beta_parameters  : {self.beta_parameters_flat.shape}")
            print(f"[ShapeModelPCA] std_devs         : {self.std_devs.shape}")
            print("--------------------------------------------------------")

        return self

    def generate_shape(self, beta_std):
        if not self.initialized:
            raise RuntimeError("PCA model is not initialized.")
        beta_std = np.asarray(beta_std, dtype=np.float64).ravel()
        if beta_std.size < self.n_pca_components:
            tmp = np.zeros(self.n_pca_components, dtype=np.float64)
            tmp[: beta_std.size] = beta_std
            beta_std = tmp
        elif beta_std.size > self.n_pca_components:
            beta_std = beta_std[: self.n_pca_components]

        beta_values = beta_std * self.std_devs
        dev_norm_flat = beta_values @ self.beta_parameters_flat.T
        dev_real_flat = self._decode_from_pca(dev_norm_flat)
        generated_shape_flat = self.mean_shape.reshape(-1) + dev_real_flat
        return generated_shape_flat.reshape((self.n_points, 3))

    def show_mean_shape(self):
        if not self.initialized:
            raise RuntimeError("PCA model is not initialized.")
        poly = self.numpy_points_to_polydata(self.mean_shape)
        self.meanModelNode = self.create_or_update_model_node(
            self.meanModelNode,
            poly,
            "ShapePCA_MeanShape",
            color=(0.55, 0.55, 0.55),
            opacity=0.35,
        )
        return self.meanModelNode

    def show_generated_shape(self, beta_std):
        coords = self.generate_shape(beta_std)
        self.generated_shape = coords
        poly = self.numpy_points_to_polydata(coords)
        self.generatedModelNode = self.create_or_update_model_node(
            self.generatedModelNode,
            poly,
            "ShapePCA_GeneratedShape",
            color=(1.0, 0.82, 0.05),
            opacity=1.0,
        )
        return self.generatedModelNode

    def project_model_to_beta(self, modelNode):
        if not self.initialized:
            raise RuntimeError("PCA model is not initialized.")
        if modelNode is None or modelNode.GetPolyData() is None:
            raise RuntimeError("Selected model has no polydata.")

        pts = self.polydata_to_points_array(modelNode.GetPolyData())
        if pts.shape != self.mean_shape.shape:
            raise RuntimeError(
                f"Selected model shape {pts.shape} does not match PCA model shape {self.mean_shape.shape}."
            )
        if self.center_meshes:
            pts = pts - np.mean(pts, axis=0, keepdims=True)

        flat_dev = (pts - self.mean_shape).reshape(1, -1)
        x_norm = self._encode_for_pca(flat_dev)
        scores = self.pca.transform(x_norm)[0]
        beta_std = scores / np.maximum(self.std_devs, self.eps)

        recon = self.generate_shape(beta_std)
        rms = float(np.sqrt(np.mean(np.sum((recon - pts) ** 2, axis=1))))
        return beta_std, rms

    # ------------------------------------------------------------------
    # Metrics / persistence
    # ------------------------------------------------------------------
    def total_explained_variance(self):
        if self.explained_variance_ratio_ is None:
            return 0.0
        return float(np.sum(self.explained_variance_ratio_))

    def model_summary_text(self):
        if not self.initialized:
            return "No PCA model loaded yet."
        lines = []
        lines.append("[ShapeModelPCA]")
        if self.data_folder_path:
            lines.append(f"Training folder: {self.data_folder_path}")
        if self._last_saved_model_path:
            lines.append(f"Loaded/Saved model: {self._last_saved_model_path}")
        lines.append(f"Meshes: {self.n_shapes}")
        lines.append(f"Points per mesh: {self.n_points}")
        lines.append(f"Features: {self.n_features} (= points × 3)")
        lines.append(f"PCA components: {self.n_pca_components}")
        lines.append(f"Preprocess: {self.pca_preprocess}")
        lines.append(f"Center meshes: {self.center_meshes}")
        lines.append(f"Total explained variance: {100.0 * self.total_explained_variance():.4f}%")
        if self.explained_variance_ratio_ is not None and len(self.explained_variance_ratio_) > 0:
            top = min(5, len(self.explained_variance_ratio_))
            txt = ", ".join([f"PC{i}={100.0 * float(self.explained_variance_ratio_[i]):.2f}%" for i in range(top)])
            lines.append(f"First components: {txt}")
        return "\n".join(lines)

    def export_variance_csv(self, path):
        if not self.initialized:
            raise RuntimeError("PCA model is not initialized.")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "component",
                "explained_variance_ratio",
                "cumulative_explained_variance_ratio",
                "explained_variance",
                "std_dev",
            ])
            for i in range(self.n_pca_components):
                writer.writerow([
                    i,
                    float(self.explained_variance_ratio_[i]),
                    float(self.cumulative_explained_variance_[i]),
                    float(self.explained_variance_[i]),
                    float(self.std_devs[i]),
                ])
        return path

    def save_pca_model(self, filepath):
        if joblib is None:
            raise RuntimeError("joblib is not available.")
        if not self.initialized:
            raise RuntimeError("PCA model is not initialized.")
        payload = {
            "version": 1,
            "n_pca_components": self.n_pca_components,
            "n_shapes": self.n_shapes,
            "n_points": self.n_points,
            "n_features": self.n_features,
            "data_folder_path": self.data_folder_path,
            "mesh_paths": self.mesh_paths,
            "pca_preprocess": self.pca_preprocess,
            "center_meshes": self.center_meshes,
            "eps": self.eps,
            "mean_shape": self.mean_shape,
            "mean_data": self.mean_data,
            "feature_scale": self.feature_scale,
            "pca": self.pca,
            "beta_parameters_flat": self.beta_parameters_flat,
            "std_devs": self.std_devs,
            "explained_variance_": self.explained_variance_,
            "explained_variance_ratio_": self.explained_variance_ratio_,
            "cumulative_explained_variance_": self.cumulative_explained_variance_,
            "topology_cells": self.topology_cells,
        }
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)
        joblib.dump(payload, filepath)
        self._last_saved_model_path = filepath
        print(f"[ShapeModelPCA] Saved PCA model: {filepath}")
        return filepath

    def load_pca_model(self, filepath):
        if joblib is None:
            raise RuntimeError("joblib is not available.")
        payload = joblib.load(filepath)
        self.reset()

        self.n_pca_components = int(payload.get("n_pca_components", 0))
        self.n_shapes = int(payload.get("n_shapes", 0))
        self.n_points = int(payload.get("n_points", 0))
        self.n_features = int(payload.get("n_features", self.n_points * 3))
        self.data_folder_path = payload.get("data_folder_path", None)
        self.mesh_paths = payload.get("mesh_paths", []) or []
        self.pca_preprocess = payload.get("pca_preprocess", "robust")
        self.center_meshes = bool(payload.get("center_meshes", True))
        self.eps = float(payload.get("eps", 1e-12))

        self.mean_shape = np.asarray(payload.get("mean_shape"), dtype=np.float64)
        self.mean_data = np.asarray(payload.get("mean_data"), dtype=np.float64)
        self.feature_scale = np.asarray(payload.get("feature_scale"), dtype=np.float64)
        self.pca = payload.get("pca", None)
        self.beta_parameters_flat = payload.get("beta_parameters_flat", None)
        if self.beta_parameters_flat is None and self.pca is not None and hasattr(self.pca, "components_"):
            self.beta_parameters_flat = self.pca.components_.T
        self.beta_parameters_flat = np.asarray(self.beta_parameters_flat, dtype=np.float64)

        self.std_devs = np.asarray(payload.get("std_devs"), dtype=np.float64)
        self.explained_variance_ = np.asarray(payload.get("explained_variance_"), dtype=np.float64)
        self.explained_variance_ratio_ = np.asarray(payload.get("explained_variance_ratio_"), dtype=np.float64)
        self.cumulative_explained_variance_ = payload.get("cumulative_explained_variance_", None)
        if self.cumulative_explained_variance_ is None:
            self.cumulative_explained_variance_ = np.cumsum(self.explained_variance_ratio_)
        else:
            self.cumulative_explained_variance_ = np.asarray(self.cumulative_explained_variance_, dtype=np.float64)

        self.topology_cells = payload.get("topology_cells", None)
        if self.mean_shape is None or self.mean_shape.ndim != 2 or self.mean_shape.shape[1] != 3:
            raise RuntimeError("Loaded PCA model does not contain a valid mean_shape.")
        if self.n_points <= 0:
            self.n_points = int(self.mean_shape.shape[0])
        if self.n_features <= 0:
            self.n_features = int(self.n_points * 3)
        if self.n_pca_components <= 0:
            self.n_pca_components = int(len(self.std_devs))

        self.initialized = True
        self._last_saved_model_path = filepath
        print(f"[ShapeModelPCA] Loaded PCA model: {filepath}")
        return self


# =============================================================================
# Minimal test harness placeholder for Slicer module discovery
# =============================================================================
class ETSE_UV__ShapeModelPCATest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_ETSE_UV__ShapeModelPCA1()

    def test_ETSE_UV__ShapeModelPCA1(self):
        self.delayDisplay("ETSE-UV Shape Model PCA module loaded.")
