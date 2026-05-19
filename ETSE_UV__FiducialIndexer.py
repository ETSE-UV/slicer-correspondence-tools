import os
import numpy as np
import slicer
import vtk
import qt
import ctk
from slicer.ScriptedLoadableModule import *

# ------------------------------------------------------------
# Module metadata
# ------------------------------------------------------------
class ETSE_UV__FiducialIndexer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Fiducial Indexer"
        parent.categories = ["ETSE_UV"]
        parent.contributors = ["ETSE-UV"]
        parent.helpText = """
        <p>Compute and reuse mesh vertex indices from fiducial landmarks.</p>

        <p><b>Workflow:</b></p>
        <ol>
          <li>Select a source mesh and a source fiducial node.</li>
          <li>Choose the 1-based fiducial range to process.</li>
          <li>Compute the closest mesh vertex index for each selected fiducial.</li>
          <li>Save or load the stored indices.</li>
          <li>Apply the same indices to another mesh with matching topology/indexing.</li>
        </ol>

        <p>The template fiducial node is used to copy labels and descriptions. Positions are
        taken from the target mesh vertices.</p>
        """
        parent.acknowledgementText = (
            "Developed by Juan Antonio De Rus Arance at the Escola Tècnica Superior "
            "d'Enginyeria (ETSE-UV), Universitat de València, in the context of the "
            "Signal Processing & Acoustic Technology (SPAT) research group. "
            "Thanks to the 3D Slicer, SlicerMorph, VTK, NumPy, SciPy, Trimesh, and related "
            "open-source communities."
        )

# ------------------------------------------------------------
# Widget (UI) 
# ------------------------------------------------------------
class ETSE_UV__FiducialIndexerWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = ETSE_UV__FiducialIndexerLogic()

        # ------------------------------------------------------------
        # Inputs: Source (compute indices)
        # ------------------------------------------------------------
        srcBox = ctk.ctkCollapsibleButton()
        srcBox.text = "Source (compute indices)"
        self.layout.addWidget(srcBox)
        srcForm = qt.QFormLayout(srcBox)

        self.sourceModelSelector = slicer.qMRMLNodeComboBox()
        self.sourceModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.sourceModelSelector.selectNodeUponCreation = True
        self.sourceModelSelector.addEnabled = False
        self.sourceModelSelector.removeEnabled = False
        self.sourceModelSelector.noneEnabled = True
        self.sourceModelSelector.setMRMLScene(slicer.mrmlScene)
        self.sourceModelSelector.setToolTip("Mesh used to compute indices (closest vertex per fiducial).")
        srcForm.addRow("Source mesh:", self.sourceModelSelector)

        self.sourceFidSelector = slicer.qMRMLNodeComboBox()
        self.sourceFidSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.sourceFidSelector.selectNodeUponCreation = True
        self.sourceFidSelector.addEnabled = False
        self.sourceFidSelector.removeEnabled = False
        self.sourceFidSelector.noneEnabled = True
        self.sourceFidSelector.setMRMLScene(slicer.mrmlScene)
        self.sourceFidSelector.setToolTip("Fiducials used to compute indices (order preserved).")
        srcForm.addRow("Source fiducials:", self.sourceFidSelector)

        self.rangeLineEdit = qt.QLineEdit("1-261")
        self.rangeLineEdit.setToolTip("1-based inclusive ranges, e.g. '1-10,15,20-25'.")
        srcForm.addRow("Fiducial range (1-based):", self.rangeLineEdit)

        self.computeBtn = qt.QPushButton("Compute closest vertex indices (store)")
        self.computeBtn.setToolTip("For each fiducial in the range, find the closest mesh vertex and store its index.")
        self.computeBtn.clicked.connect(self.onComputeIndices)
        srcForm.addRow(self.computeBtn)

        self.storedLabel = qt.QLabel("Stored indices: none")
        srcForm.addRow(self.storedLabel)

        # ------------------------------------------------------------
        # Save / Load indices
        # ------------------------------------------------------------
        fileBox = ctk.ctkCollapsibleButton()
        fileBox.text = "Save / load indices (.npy)"
        self.layout.addWidget(fileBox)
        fileLayout = qt.QVBoxLayout(fileBox)

        btnRow = qt.QHBoxLayout()
        self.saveBtn = qt.QPushButton("Save indices…")
        self.loadBtn = qt.QPushButton("Load indices…")
        btnRow.addWidget(self.saveBtn)
        btnRow.addWidget(self.loadBtn)
        fileLayout.addLayout(btnRow)

        self.saveBtn.toolTip = "Save the currently stored indices as a .npy file (ordered as the fiducial range)."
        self.loadBtn.toolTip = "Load a .npy file with indices (each value is a vertex id)."

        self.saveBtn.clicked.connect(self.onSaveIndices)
        self.loadBtn.clicked.connect(self.onLoadIndices)

        # ------------------------------------------------------------
        # Apply (target)
        # ------------------------------------------------------------
        applyBox = ctk.ctkCollapsibleButton()
        applyBox.text = "Apply indices to target mesh"
        self.layout.addWidget(applyBox)
        applyForm = qt.QFormLayout(applyBox)

        self.targetModelSelector = slicer.qMRMLNodeComboBox()
        self.targetModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.targetModelSelector.selectNodeUponCreation = True
        self.targetModelSelector.addEnabled = False
        self.targetModelSelector.removeEnabled = False
        self.targetModelSelector.noneEnabled = True
        self.targetModelSelector.setMRMLScene(slicer.mrmlScene)
        self.targetModelSelector.setToolTip("Target mesh that will receive the fiducials (same topology/indexing).")
        applyForm.addRow("Target mesh:", self.targetModelSelector)

        self.templateFidSelector = slicer.qMRMLNodeComboBox()
        self.templateFidSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.templateFidSelector.selectNodeUponCreation = True
        self.templateFidSelector.addEnabled = False
        self.templateFidSelector.removeEnabled = False
        self.templateFidSelector.noneEnabled = True
        self.templateFidSelector.setMRMLScene(slicer.mrmlScene)
        self.templateFidSelector.setToolTip(
            "Template fiducial node to copy labels/descriptions from (positions ignored)."
        )
        applyForm.addRow("Template fiducials:", self.templateFidSelector)

        self.applyBtn = qt.QPushButton("Apply stored indices to target mesh (create fiducials)")
        self.applyBtn.toolTip = (
            "Create a new fiducial node on the target mesh at the stored vertex indices.\n"
            "Labels/descriptions copied 1:1 from template."
        )
        self.applyBtn.clicked.connect(self.onApplyIndices)
        applyForm.addRow(self.applyBtn)


        self.makePointMeshBtn = qt.QPushButton("Create model from fiducials (points)")
        self.makePointMeshBtn.toolTip = "Create a Model node that contains only the fiducial points (as a point cloud)."
        self.makePointMeshBtn.clicked.connect(self.onCreatePointMesh)
        applyForm.addRow(self.makePointMeshBtn)

        self.makePointMeshFromIdxBtn = qt.QPushButton("Create model from stored indices (points)")
        self.makePointMeshFromIdxBtn.toolTip = "Create a Model node from stored vertex indices on the target mesh."
        self.makePointMeshFromIdxBtn.clicked.connect(self.onCreatePointMeshFromIndices)
        applyForm.addRow(self.makePointMeshFromIdxBtn)


        # ------------------------------------------------------------
        # Delaunay triangulation (optional surface from points)
        # ------------------------------------------------------------
        delaunayBox = ctk.ctkCollapsibleButton()
        delaunayBox.text = "Surface from POINTS (Fiducials or Stored Indices)"
        self.layout.addWidget(delaunayBox)
        dForm = qt.QFormLayout(delaunayBox)

        self.delaunayEnableCheck = qt.QCheckBox("Enable Delaunay triangulation")
        self.delaunayEnableCheck.checked = True
        dForm.addRow(self.delaunayEnableCheck)

        self.delaunayModeCombo = qt.QComboBox()
        self.delaunayModeCombo.addItems([
            "2D (best-fit plane)",
            "2D (XY)",
            "2D (XZ)",
            "2D (YZ)",
            "3D (tets -> surface)"
        ])
        self.delaunayModeCombo.setCurrentText("3D (tets -> surface)")
        dForm.addRow("Delaunay mode:", self.delaunayModeCombo)

        self.delaunayAlphaSpin = qt.QDoubleSpinBox()
        self.delaunayAlphaSpin.setRange(0.0, 1e9)
        self.delaunayAlphaSpin.setValue(2.75)
        self.delaunayAlphaSpin.setSingleStep(1.0)
        self.delaunayAlphaSpin.setToolTip(
            "Alpha=0 disables alpha-shape trimming. "
            "Try small values (e.g. 2-20 in mm units) to avoid 'outer hull' triangles."
        )
        dForm.addRow("Alpha (0 = off):", self.delaunayAlphaSpin)

        self.delaunayCleanCheck = qt.QCheckBox("Merge duplicates (vtkCleanPolyData)")
        self.delaunayCleanCheck.checked = False
        dForm.addRow(self.delaunayCleanCheck)

        # Buttons
        self.makeDelaunayFromFidsBtn = qt.QPushButton("Create Delaunay surface from fiducials")
        self.makeDelaunayFromFidsBtn.clicked.connect(self.onCreateDelaunayFromFiducials)
        dForm.addRow(self.makeDelaunayFromFidsBtn)

        self.makeDelaunayFromIdxBtn = qt.QPushButton("Create Delaunay surface from stored indices")
        self.makeDelaunayFromIdxBtn.clicked.connect(self.onCreateDelaunayFromIndices)
        dForm.addRow(self.makeDelaunayFromIdxBtn)


        self.makeDelaunayFromFidsBtn.toolTip = "Triangulates the fiducial POSITIONS (creates a surface from fiducial points)."

        self.makeDelaunayFromIdxBtn.toolTip = (
            "Triangulates points taken from the TARGET mesh using the stored vertex indices.\n"
            "It does NOT use the original mesh triangles, only its vertex coordinates at those indices."
        )



        # ------------------------------------------------------------
        # Batch (apply options to folder of target meshes)
        # ------------------------------------------------------------
        batchBox = ctk.ctkCollapsibleButton()
        batchBox.text = "Batch (apply stored indices to folder)"
        self.layout.addWidget(batchBox)
        bForm = qt.QFormLayout(batchBox)

        # Input folder
        rowIn = qt.QHBoxLayout()
        self.batchInputDirEdit = qt.QLineEdit("")
        btnIn = qt.QPushButton("Browse…")
        btnIn.clicked.connect(self.onBrowseBatchInput)
        rowIn.addWidget(self.batchInputDirEdit)
        rowIn.addWidget(btnIn)
        wIn = qt.QWidget()
        wIn.setLayout(rowIn)
        bForm.addRow("Input folder (.vtk/.vtp/.ply):", wIn)

        # Output folder
        rowOut = qt.QHBoxLayout()
        self.batchOutputDirEdit = qt.QLineEdit("")
        btnOut = qt.QPushButton("Browse…")
        btnOut.clicked.connect(self.onBrowseBatchOutput)
        rowOut.addWidget(self.batchOutputDirEdit)
        rowOut.addWidget(btnOut)
        wOut = qt.QWidget()
        wOut.setLayout(rowOut)
        bForm.addRow("Output folder:", wOut)

        # What to generate
        self.batchMakePointCloudCheck = qt.QCheckBox("Save point cloud (from stored indices)")
        self.batchMakePointCloudCheck.checked = True
        bForm.addRow(self.batchMakePointCloudCheck)

        self.batchMakeDelaunayCheck = qt.QCheckBox("Save Delaunay surface (from stored indices)")
        self.batchMakeDelaunayCheck.checked = True
        bForm.addRow(self.batchMakeDelaunayCheck)

        self.batchMakeFiducialsCheck = qt.QCheckBox("Save fiducials (.mrk.json from stored indices)")
        self.batchMakeFiducialsCheck.checked = True
        self.batchMakeFiducialsCheck.setToolTip(
            "For each mesh, create a Markups Fiducial node at the stored vertex indices and save it as .mrk.json.\n"
            "Labels/descriptions are copied from the Template fiducials selector above."
        )
        bForm.addRow(self.batchMakeFiducialsCheck)

        # Run
        self.batchRunBtn = qt.QPushButton("RUN batch")
        self.batchRunBtn.toolTip = (
            "For each mesh in Input folder, extract points at stored indices and save selected outputs.\n"
            "Point clouds and Delaunay surfaces use the Delaunay options selected above.\n"
            "Fiducials are saved as .mrk.json using labels/descriptions from the template fiducials."
        )
        self.batchRunBtn.clicked.connect(self.onRunBatch)
        bForm.addRow(self.batchRunBtn)


        self.layout.addStretch(1)

    # ------------------------------------------------
    # Actions
    # ------------------------------------------------
    def onComputeIndices(self):
        mesh = self.sourceModelSelector.currentNode()
        fids = self.sourceFidSelector.currentNode()
        if not mesh or not fids:
            slicer.util.errorDisplay("Select BOTH a source mesh and a source fiducial node.")
            return
        try:
            indices = self.logic.compute_indices(mesh, fids, self.rangeLineEdit.text)
            self._updateStoredLabel()
            slicer.util.infoDisplay(f"Computed {len(indices)} indices.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to compute indices: {e}")

    def onSaveIndices(self):
        if not self.logic.get_last_indices():
            slicer.util.errorDisplay("No indices to save. Compute or load first.")
            return
        filePath = qt.QFileDialog.getSaveFileName(self.parent, "Save indices", "", "NumPy array (*.npy)")
        if filePath:
            if not filePath.lower().endswith(".npy"):
                filePath += ".npy"
            try:
                self.logic.save_indices(filePath)
                slicer.util.infoDisplay(f"Saved indices to:\n{filePath}")
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to save: {e}")

    def onLoadIndices(self):
        filePath = qt.QFileDialog.getOpenFileName(self.parent, "Load indices", "", "NumPy array (*.npy)")
        if filePath:
            try:
                self.logic.load_indices(filePath)
                self._updateStoredLabel()
                slicer.util.infoDisplay(f"Loaded {len(self.logic.get_last_indices())} indices.")
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to load: {e}")

    def onApplyIndices(self):
        target = self.targetModelSelector.currentNode()
        template = self.templateFidSelector.currentNode()
        if not target or not template:
            slicer.util.errorDisplay("Select BOTH a target mesh and a template fiducial node.")
            return
        if not self.logic.get_last_indices():
            slicer.util.errorDisplay("No stored indices. Load or compute them first.")
            return
        try:
            out = self.logic.apply_indices_to_fiducials(target, template)
            if out:
                slicer.util.infoDisplay(
                    f"Created fiducials: {out.GetName()} ({out.GetNumberOfControlPoints()} points)"
                )
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to apply indices: {e}")

    def _updateStoredLabel(self):
        arr = self.logic.get_last_indices()
        if not arr:
            self.storedLabel.setText("Stored indices: none")
        else:
            self.storedLabel.setText(f"Stored indices: {len(arr)}")

    def onCreatePointMesh(self):
        fids = self.templateFidSelector.currentNode() or self.sourceFidSelector.currentNode()
        if not fids:
            slicer.util.errorDisplay("Select a fiducial node (template or source).")
            return
        try:
            out = self.logic.create_mesh_from_fiducials(
                fids,
                outputName=f"{fids.GetName()}_PointCloud",
                connect_as_polyline=False
            )
            slicer.util.infoDisplay(f"Created model: {out.GetName()}")
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def onCreatePointMeshFromIndices(self):
        target = self.targetModelSelector.currentNode()
        if not target:
            slicer.util.errorDisplay("Select a target mesh.")
            return
        if not self.logic.get_last_indices():
            slicer.util.errorDisplay("No stored indices. Compute or load first.")
            return
        try:
            out = self.logic.create_mesh_from_stored_indices(
                target,
                outputName=f"{target.GetName()}_FiducialSubset",
                connect_as_polyline=False,
                copy_point_arrays=False
            )
            slicer.util.infoDisplay(f"Created model: {out.GetName()}")
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def onCreateDelaunayFromFiducials(self):
        if not bool(self.delaunayEnableCheck.checked):
            slicer.util.errorDisplay("Enable Delaunay triangulation first.")
            return

        fids = self.templateFidSelector.currentNode() or self.sourceFidSelector.currentNode()
        if not fids:
            slicer.util.errorDisplay("Select a fiducial node (template or source).")
            return

        mode = str(self.delaunayModeCombo.currentText)
        alpha = float(self.delaunayAlphaSpin.value)
        clean = bool(self.delaunayCleanCheck.checked)

        rangeText = self.rangeLineEdit.text  #  

        try:
            out = self.logic.create_delaunay_surface_from_fiducials(
                fids,
                outputName=f"{fids.GetName()}_Delaunay",
                mode=mode,
                alpha=alpha,
                clean=clean,
                range_text=rangeText  #  
            )
            slicer.util.infoDisplay(f"Created Delaunay model: {out.GetName()}")
        except Exception as e:
            slicer.util.errorDisplay(str(e))


    def onCreateDelaunayFromIndices(self):
        if not bool(self.delaunayEnableCheck.checked):
            slicer.util.errorDisplay("Enable Delaunay triangulation first.")
            return

        target = self.targetModelSelector.currentNode()
        if not target:
            slicer.util.errorDisplay("Select a target mesh.")
            return
        if not self.logic.get_last_indices():
            slicer.util.errorDisplay("No stored indices. Compute or load first.")
            return

        mode = str(self.delaunayModeCombo.currentText)
        alpha = float(self.delaunayAlphaSpin.value)
        clean = bool(self.delaunayCleanCheck.checked)

        try:
            out = self.logic.create_delaunay_surface_from_stored_indices(
                target,
                outputName=f"{target.GetName()}_FidSubset_Delaunay",
                mode=mode,
                alpha=alpha,
                clean=clean
            )
            slicer.util.infoDisplay(f"Created Delaunay model: {out.GetName()}")
        except Exception as e:
            slicer.util.errorDisplay(str(e))
    def onBrowseBatchInput(self):
        d = qt.QFileDialog.getExistingDirectory(self.parent, "Select input folder with meshes")
        if d:
            self.batchInputDirEdit.setText(d)

    def onBrowseBatchOutput(self):
        d = qt.QFileDialog.getExistingDirectory(self.parent, "Select output folder")
        if d:
            self.batchOutputDirEdit.setText(d)

    def onRunBatch(self):
        if not self.logic.get_last_indices():
            slicer.util.errorDisplay("No stored indices. Compute or load them first.")
            return

        inDir = self.batchInputDirEdit.text.strip()
        outDir = self.batchOutputDirEdit.text.strip()
        if not inDir or not os.path.isdir(inDir):
            slicer.util.errorDisplay("Pick a valid input folder.")
            return
        if not outDir or not os.path.isdir(outDir):
            slicer.util.errorDisplay("Pick a valid output folder.")
            return

        doPts = bool(self.batchMakePointCloudCheck.checked)
        doDel = bool(self.batchMakeDelaunayCheck.checked)
        doFids = bool(self.batchMakeFiducialsCheck.checked)
        if not (doPts or doDel or doFids):
            slicer.util.errorDisplay("Enable at least one output (point cloud, Delaunay and/or fiducials).")
            return

        templateFids = self.templateFidSelector.currentNode() if doFids else None
        if doFids and not templateFids:
            slicer.util.errorDisplay("To save fiducials in batch, select Template fiducials first.")
            return

        # Use SAME options as your current Delaunay UI
        mode = str(self.delaunayModeCombo.currentText)
        alpha = float(self.delaunayAlphaSpin.value)
        clean = bool(self.delaunayCleanCheck.checked)

        try:
            processed, saved = self.logic.batch_apply_options_from_stored_indices(
                inputDir=inDir,
                outputDir=outDir,
                makePointCloud=doPts,
                makeDelaunay=doDel,
                makeFiducials=doFids,
                templateFidNode=templateFids,
                delaunayMode=mode,
                delaunayAlpha=alpha,
                delaunayClean=clean
            )
            slicer.util.infoDisplay(f"Batch done. Processed: {processed}  Files saved: {saved}\nOutput: {outDir}")
        except Exception as e:
            slicer.util.errorDisplay(str(e))

# ------------------------------------------------------------
# Logic 
# ------------------------------------------------------------
class ETSE_UV__FiducialIndexerLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        try:
            super(ETSE_UV__FiducialIndexerLogic, self).__init__()
        except Exception:
            pass
        # ordered list of vertex ids (0-based), one per fiducial in the chosen range
        self._last_indices = None

    # Public helpers
    def get_last_indices(self):
        return self._last_indices

    # ----- core -----
    def compute_indices(self, meshNode, fidNode, range_text="1-261"):
        """
        For each fiducial index in range_text (1-based), get the world position and find the closest vertex
        in meshNode. Store EXACTLY in that order. Returns the list of indices.
        """
        idx_list = self._parse_range_1based(range_text)
        if fidNode.GetNumberOfControlPoints() == 0:
            raise ValueError("Fiducial node is empty.")

        poly = meshNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError("Mesh has no points.")

        locator = vtk.vtkPointLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()

        out = []
        n_fids = fidNode.GetNumberOfControlPoints()
        for one_based in idx_list:
            i = int(one_based) - 1
            if i < 0 or i >= n_fids:
                continue
            pos = [0.0, 0.0, 0.0]
            try:
                fidNode.GetNthControlPointPositionWorld(i, pos)
            except Exception:
                fidNode.GetNthControlPointPosition(i, pos)
            pid = int(locator.FindClosestPoint(pos))
            out.append(pid)

        self._last_indices = out
        return out

    def save_indices(self, filepath_npy):
        if not self._last_indices:
            raise ValueError("No indices stored.")
        arr = np.asarray(self._last_indices, dtype=np.int64)
        np.save(filepath_npy, arr)

    def load_indices(self, filepath_npy):
        arr = np.load(filepath_npy)
        arr = np.asarray(arr, dtype=np.int64).ravel().tolist()
        self._last_indices = arr
        return arr

    def apply_indices_to_fiducials(self, targetModelNode, templateFidNode, outputName=None):
        """
        Assume SAME topology/indexing. For each stored index k, read targetModelNode.GetPolyData().GetPoint(k)
        and create a fiducial there. Label/description are copied from templateFidNode in the SAME order.
        """
        if not self._last_indices:
            raise ValueError("No stored indices to apply.")

        poly = targetModelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError("Target mesh has no points.")

        nPts = poly.GetNumberOfPoints()
        countTemplate = templateFidNode.GetNumberOfControlPoints()
        countToCreate = min(len(self._last_indices), countTemplate)

        outName = outputName or f"{targetModelNode.GetName()}_AppliedFiducials"
        outFids = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", outName)

        for i in range(countToCreate):
            vid = int(self._last_indices[i])
            if vid < 0 or vid >= nPts:
                continue

            pos = [0.0, 0.0, 0.0]
            poly.GetPoint(vid, pos)

            label = templateFidNode.GetNthControlPointLabel(i)
            desc = templateFidNode.GetNthControlPointDescription(i)

            idx = outFids.AddControlPoint(pos, label)
            if idx is None:
                idx = outFids.GetNumberOfControlPoints() - 1
            try:
                outFids.SetNthControlPointDescription(idx, desc)
            except Exception:
                pass

        outFids.CreateDefaultDisplayNodes()
        dn = outFids.GetDisplayNode()
        if dn:
            dn.SetPointSize(6)

        return outFids

    # ----- utils -----
    def _parse_range_1based(self, text):
        """
        '1-10,15, 20-22' -> [1,2,...,10,15,20,21,22]
        Returns a LIST of 1-based indices, unique and sorted.
        """
        if not text or not str(text).strip():
            return []
        items = []
        for chunk in str(text).replace(" ", "").split(","):
            if not chunk:
                continue
            if "-" in chunk:
                a, b = chunk.split("-")
                a, b = int(a), int(b)
                step = 1 if b >= a else -1
                items.extend(range(a, b + step, step))
            else:
                items.append(int(chunk))
        return sorted(set(items))

    def create_mesh_from_fiducials(self, fidNode, outputName="FiducialPointCloud", connect_as_polyline=False):
        """
        Build a vtkPolyData containing only the fiducial points.
        Optionally connect them as a polyline in their current order.
        """
        if fidNode is None or fidNode.GetNumberOfControlPoints() == 0:
            raise ValueError("Fiducial node is empty or None.")

        pts = vtk.vtkPoints()
        n = fidNode.GetNumberOfControlPoints()
        pts.SetNumberOfPoints(n)

        p = [0.0, 0.0, 0.0]
        for i in range(n):
            try:
                fidNode.GetNthControlPointPositionWorld(i, p)
            except Exception:
                fidNode.GetNthControlPointPosition(i, p)
            pts.SetPoint(i, p)

        outPD = vtk.vtkPolyData()
        outPD.SetPoints(pts)

        if connect_as_polyline and n >= 2:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(n)
            for i in range(n):
                line.GetPointIds().SetId(i, i)
            ca = vtk.vtkCellArray()
            ca.InsertNextCell(line)
            outPD.SetLines(ca)

        outNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", outputName)
        outNode.SetAndObservePolyData(outPD)
        outNode.CreateDefaultDisplayNodes()
        dn = outNode.GetDisplayNode()
        if dn:
            dn.SetRepresentation(dn.PointsRepresentation)
            dn.SetPointSize(6)
            dn.SetLineWidth(2)
        return outNode

    def create_mesh_from_stored_indices(self, targetModelNode, outputName=None, connect_as_polyline=False, copy_point_arrays=False):
        """
        Build a vtkPolyData containing only points referenced by stored vertex indices
        from the target model (same topology/indexing).
        Optionally connect them as a polyline. Optionally copy point-data arrays.
        """
        if not self._last_indices:
            raise ValueError("No stored indices. Compute or load first.")
        if targetModelNode is None:
            raise ValueError("Target model is None.")

        poly = targetModelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError("Target mesh has no points.")

        nPts = poly.GetNumberOfPoints()
        ids = [int(i) for i in self._last_indices if 0 <= int(i) < nPts]
        if len(ids) == 0:
            raise ValueError("No valid indices to build mesh.")

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(len(ids))

        p = [0.0, 0.0, 0.0]
        for new_i, old_vid in enumerate(ids):
            poly.GetPoint(old_vid, p)
            pts.SetPoint(new_i, p)

        outPD = vtk.vtkPolyData()
        outPD.SetPoints(pts)

        if connect_as_polyline and len(ids) >= 2:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(ids))
            for i in range(len(ids)):
                line.GetPointIds().SetId(i, i)
            ca = vtk.vtkCellArray()
            ca.InsertNextCell(line)
            outPD.SetLines(ca)

        if copy_point_arrays:
            # Copy point data arrays from the original mesh to the subset (if exist)
            inPD = poly.GetPointData()
            outPD.GetPointData().Initialize()
            for a in range(inPD.GetNumberOfArrays()):
                arr = inPD.GetArray(a)
                if arr is None:
                    continue
                newArr = arr.NewInstance()
                newArr.DeepCopy(arr)
                newArr.SetNumberOfTuples(len(ids))
                # overwrite tuples with the selected ids
                for new_i, old_vid in enumerate(ids):
                    newArr.SetTuple(new_i, old_vid, arr)  # VTK trick: copy tuple old_vid -> new_i
                outPD.GetPointData().AddArray(newArr)
                newArr.Delete()

        outName = outputName or f"{targetModelNode.GetName()}_FiducialSubset"
        outNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", outName)
        outNode.SetAndObservePolyData(outPD)
        outNode.CreateDefaultDisplayNodes()
        dn = outNode.GetDisplayNode()
        if dn:
            dn.SetRepresentation(dn.PointsRepresentation)
            dn.SetPointSize(6)
            dn.SetLineWidth(2)
        return outNode

    # ------------------------------------------------------------
    # Delaunay helpers
    # ------------------------------------------------------------
    def _polydata_points_from_fiducials(self, fidNode, range_text=None):
        if fidNode is None or fidNode.GetNumberOfControlPoints() == 0:
            raise ValueError("Fiducial node is empty or None.")

        n_all = fidNode.GetNumberOfControlPoints()

        if range_text and str(range_text).strip():
            idx_list_1b = self._parse_range_1based_keep_order(range_text)
            idx_list_0b = []
            for oneb in idx_list_1b:
                i = int(oneb) - 1
                if 0 <= i < n_all:
                    idx_list_0b.append(i)
            ids = idx_list_0b
        else:
            ids = list(range(n_all))

        if len(ids) < 3:
            raise ValueError("Need >= 3 fiducial points (after applying range) for Delaunay.")

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(len(ids))

        p = [0.0, 0.0, 0.0]
        for j, i in enumerate(ids):
            try:
                fidNode.GetNthControlPointPositionWorld(i, p)
            except Exception:
                fidNode.GetNthControlPointPosition(i, p)
            pts.SetPoint(j, p)

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd = self._ensure_verts(pd)
        return pd
        
    def _polydata_points_from_indices(self, targetModelNode, ids):
        if targetModelNode is None:
            raise ValueError("Target model is None.")
        poly = targetModelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError("Target mesh has no points.")

        nPts = poly.GetNumberOfPoints()
        ids = [int(i) for i in ids if 0 <= int(i) < nPts]
        if len(ids) == 0:
            raise ValueError("No valid indices to build point set.")

        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(len(ids))

        p = [0.0, 0.0, 0.0]
        for new_i, old_vid in enumerate(ids):
            poly.GetPoint(old_vid, p)
            pts.SetPoint(new_i, p)

        outPD = vtk.vtkPolyData()
        outPD.SetPoints(pts)
        outPD = self._ensure_verts(outPD)
        return outPD

    def _clean_polydata_points(self, pd):
        pd = self._ensure_verts(pd)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(pd)
        clean.Update()
        out = clean.GetOutput()
        out = self._ensure_verts(out)
        return out


    def _project_to_plane_polydata(self, pd, mode="2D (best-fit plane)"):
        """
        Returns:
          projectedPD (points in 2D plane embedded in 3D coords),
          transformBack (vtkTransform) that maps projected->original 3D
        We do: original -> plane coords (x,y,0), run Delaunay2D, then transform back.
        """
        pts = pd.GetPoints()
        n = pts.GetNumberOfPoints()
        if n < 3:
            raise ValueError("Need at least 3 points for Delaunay 2D.")

        X = np.zeros((n, 3), dtype=float)
        p = [0.0, 0.0, 0.0]
        for i in range(n):
            pts.GetPoint(i, p)
            X[i, :] = p

        # Choose basis
        if mode == "2D (XY)":
            origin = X.mean(axis=0)
            e1 = np.array([1.0, 0.0, 0.0])
            e2 = np.array([0.0, 1.0, 0.0])
        elif mode == "2D (XZ)":
            origin = X.mean(axis=0)
            e1 = np.array([1.0, 0.0, 0.0])
            e2 = np.array([0.0, 0.0, 1.0])
        elif mode == "2D (YZ)":
            origin = X.mean(axis=0)
            e1 = np.array([0.0, 1.0, 0.0])
            e2 = np.array([0.0, 0.0, 1.0])
        else:
            # best-fit plane via PCA
            origin = X.mean(axis=0)
            Xc = X - origin[None, :]
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            e1 = Vt[0, :]
            e2 = Vt[1, :]

        # Ensure orthonormal
        e1 = e1 / (np.linalg.norm(e1) + 1e-12)
        e2 = e2 - np.dot(e2, e1) * e1
        e2 = e2 / (np.linalg.norm(e2) + 1e-12)
        e3 = np.cross(e1, e2)
        e3 = e3 / (np.linalg.norm(e3) + 1e-12)

        # Build transform matrices:
        # plane coords u,v,w -> world: origin + u*e1 + v*e2 + w*e3
        # world -> plane: [e1;e2;e3]^T (x-origin)
        R = np.vstack([e1, e2, e3])  # 3x3, rows are basis

        projPts = vtk.vtkPoints()
        projPts.SetNumberOfPoints(n)
        for i in range(n):
            v = X[i, :] - origin
            uvw = R @ v
            projPts.SetPoint(i, float(uvw[0]), float(uvw[1]), 0.0)

        projPD = vtk.vtkPolyData()
        projPD.SetPoints(projPts)

        # transform back: (u,v,0) -> origin + u*e1 + v*e2
        tBack = vtk.vtkTransform()
        m = vtk.vtkMatrix4x4()
        # columns are e1,e2,e3, origin
        m.SetElement(0, 0, float(e1[0])); m.SetElement(1, 0, float(e1[1])); m.SetElement(2, 0, float(e1[2]))
        m.SetElement(0, 1, float(e2[0])); m.SetElement(1, 1, float(e2[1])); m.SetElement(2, 1, float(e2[2]))
        m.SetElement(0, 2, float(e3[0])); m.SetElement(1, 2, float(e3[1])); m.SetElement(2, 2, float(e3[2]))
        m.SetElement(0, 3, float(origin[0])); m.SetElement(1, 3, float(origin[1])); m.SetElement(2, 3, float(origin[2]))
        m.SetElement(3, 0, 0.0); m.SetElement(3, 1, 0.0); m.SetElement(3, 2, 0.0); m.SetElement(3, 3, 1.0)
        tBack.SetMatrix(m)

        return projPD, tBack

    def triangulate_points_delaunay_2d(self, pointsPD, mode="2D (best-fit plane)", alpha=0.0, clean=True):
        """
        Input: vtkPolyData with points only.
        Output: vtkPolyData with triangles (polys) in original 3D space.
        """
        if pointsPD is None or pointsPD.GetNumberOfPoints() < 3:
            raise ValueError("Need >= 3 points for Delaunay 2D.")

        pd = pointsPD
        if clean:
            pd = self._clean_polydata_points(pd)

        projPD, tBack = self._project_to_plane_polydata(pd, mode=mode)

        del2d = vtk.vtkDelaunay2D()
        del2d.SetInputData(projPD)
        if alpha and float(alpha) > 0.0:
            del2d.SetAlpha(float(alpha))
            del2d.AlphaTrimmingOn()
        del2d.Update()
        triPD_proj = del2d.GetOutput()

        # Transform back to original 3D
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetTransform(tBack)
        tf.SetInputData(triPD_proj)
        tf.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(tf.GetOutput())
        out.Modified()
        return out


    def triangulate_points_delaunay_3d_surface(self, pointsPD, alpha=0.0, clean=True):
        """
        Delaunay3D produces tetrahedra; this extracts the outer surface as vtkPolyData.
        NOTE: vtkDelaunay3D has no AlphaTrimmingOn(). Alpha (if >0) is still supported via SetAlpha().
        """
        if pointsPD is None or pointsPD.GetNumberOfPoints() < 4:
            raise ValueError("Need >= 4 points for Delaunay 3D.")

        pd = pointsPD
        if clean:
            pd = self._clean_polydata_points(pd)

        del3d = vtk.vtkDelaunay3D()
        del3d.SetInputData(pd)
        if alpha and float(alpha) > 0.0:
            del3d.SetAlpha(float(alpha))
        del3d.Update()

        surf = vtk.vtkDataSetSurfaceFilter()
        surf.SetInputConnection(del3d.GetOutputPort())
        surf.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(surf.GetOutput())
        out.Modified()
        return out


    def _add_model_from_polydata(self, polydata, name):
        outNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        outNode.SetAndObservePolyData(polydata)
        outNode.CreateDefaultDisplayNodes()
        dn = outNode.GetDisplayNode()
        if dn:
            dn.SetBackfaceCulling(False)
            dn.SetOpacity(0.8)
        return outNode

    def create_delaunay_surface_from_fiducials(self, fidNode, outputName="Fiducials_Delaunay",
                                          mode="3D (tets -> surface)", alpha=3.5, clean=False,
                                          range_text=None):
        ptsPD = self._polydata_points_from_fiducials(fidNode, range_text=range_text)
        if mode.startswith("3D"):
            outPD = self.triangulate_points_delaunay_3d_surface(ptsPD, alpha=alpha, clean=clean)
        else:
            outPD = self.triangulate_points_delaunay_2d(ptsPD, mode=mode, alpha=alpha, clean=clean)

        return self._add_model_from_polydata(outPD, outputName)

    def create_delaunay_surface_from_stored_indices(self, targetModelNode, outputName=None,
                                                    mode="2D (best-fit plane)", alpha=0.0, clean=True):
        if not self._last_indices:
            raise ValueError("No stored indices. Compute or load first.")
        ptsPD = self._polydata_points_from_indices(targetModelNode, self._last_indices)

        if mode.startswith("3D"):
            outPD = self.triangulate_points_delaunay_3d_surface(ptsPD, alpha=alpha, clean=clean)
        else:
            outPD = self.triangulate_points_delaunay_2d(ptsPD, mode=mode, alpha=alpha, clean=clean)

        name = outputName or f"{targetModelNode.GetName()}_DelaunayFromIndices"
        return self._add_model_from_polydata(outPD, name)

    def _ensure_verts(self, pd):
        """
        Ensure the polydata has vertex cells so filters like vtkCleanPolyData
        don't drop all points as 'unused'.
        """
        if pd is None or pd.GetNumberOfPoints() == 0:
            return pd

        # If it already has verts (or any cells), keep as is
        if (pd.GetVerts() is not None and pd.GetVerts().GetNumberOfCells() > 0) or pd.GetNumberOfCells() > 0:
            return pd

        verts = vtk.vtkCellArray()
        n = pd.GetNumberOfPoints()
        for i in range(n):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)

        out = vtk.vtkPolyData()
        out.DeepCopy(pd)
        out.SetVerts(verts)
        out.Modified()
        return out

    def _parse_range_1based_keep_order(self, text):
        """
        '1-3,3,2' -> [1,2,3,3,2]
        - NO ordena
        - NO elimina duplicados
        """
        if not text or not str(text).strip():
            return []
        out = []
        for chunk in str(text).replace(" ", "").split(","):
            if not chunk:
                continue
            if "-" in chunk:
                a, b = chunk.split("-")
                a, b = int(a), int(b)
                step = 1 if b >= a else -1
                out.extend(list(range(a, b + step, step)))
            else:
                out.append(int(chunk))
        return out

    def _save_polydata(self, polyData, filePath):
        """
        Save polydata using Slicer's ModelStorageNode to control coordinate system (RAS),
        avoiding unwanted LPS<->RAS flips on reload.
        Supports .vtp / .vtk / .ply (same as before).
        """
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise RuntimeError("No polydata to save.")

        ext = os.path.splitext(filePath)[1].lower()
        if ext not in (".vtp", ".vtk", ".ply"):
            raise RuntimeError(f"Unsupported output extension: {ext} (use .vtp/.vtk/.ply)")

        # Temporary model node for storage write
        tmpNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "__tmpSaveModel")
        tmpNode.SetAndObservePolyData(polyData)

        storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelStorageNode", "__tmpSaveStorage")
        storage.SetFileName(filePath)

        # IMPORTANT: Force RAS so Slicer does NOT apply an extra LPS->RAS conversion when reloading
        if hasattr(storage, "SetCoordinateSystemToRAS"):
            storage.SetCoordinateSystemToRAS()
        else:
            # fallback for older builds
            try:
                storage.SetCoordinateSystem(slicer.vtkMRMLStorageNode.CoordinateSystemRAS)
            except Exception:
                pass

        ok = storage.WriteData(tmpNode)

        # Cleanup
        slicer.mrmlScene.RemoveNode(storage)
        slicer.mrmlScene.RemoveNode(tmpNode)

        if not ok:
            raise RuntimeError(f"Failed to write model via Slicer storage node: {filePath}")


    def _save_fiducials(self, fidNode, filePath):
        """
        Save a vtkMRMLMarkupsFiducialNode as Slicer Markups JSON (.mrk.json).
        The coordinate system is forced to RAS when supported, matching the model-saving behavior above.
        """
        if fidNode is None or fidNode.GetNumberOfControlPoints() == 0:
            raise RuntimeError("No fiducials to save.")

        if not filePath.lower().endswith(".mrk.json"):
            raise RuntimeError("Fiducial output must use the .mrk.json extension.")

        storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsJsonStorageNode", "__tmpSaveMarkupsStorage")
        storage.SetFileName(filePath)

        # Keep coordinates in RAS when this Slicer build supports explicit coordinate-system selection.
        if hasattr(storage, "SetCoordinateSystemToRAS"):
            storage.SetCoordinateSystemToRAS()
        else:
            try:
                storage.SetCoordinateSystem(slicer.vtkMRMLStorageNode.CoordinateSystemRAS)
            except Exception:
                pass

        ok = storage.WriteData(fidNode)
        slicer.mrmlScene.RemoveNode(storage)

        if not ok:
            raise RuntimeError(f"Failed to write fiducials via Slicer storage node: {filePath}")


    def batch_apply_options_from_stored_indices(self, inputDir, outputDir,
                                               makePointCloud=True,
                                               makeDelaunay=True,
                                               makeFiducials=False,
                                               templateFidNode=None,
                                               delaunayMode="3D (tets -> surface)",
                                               delaunayAlpha=3.5,
                                               delaunayClean=False):
        if not self._last_indices:
            raise RuntimeError("No stored indices.")

        if not os.path.isdir(inputDir) or not os.path.isdir(outputDir):
            raise RuntimeError("Invalid input/output folder.")

        if makeFiducials and templateFidNode is None:
            raise RuntimeError("Template fiducials are required to save .mrk.json outputs.")

        exts = (".vtk", ".vtp", ".ply")
        files = [f for f in os.listdir(inputDir)
                 if os.path.isfile(os.path.join(inputDir, f)) and os.path.splitext(f)[1].lower() in exts]

        processed = 0
        saved = 0

        for fname in files:
            inPath = os.path.join(inputDir, fname)
            ok, node = slicer.util.loadModel(inPath, returnNode=True)
            if not ok or node is None:
                processed += 1
                continue

            try:
                # 1) Points from stored indices (this already ensures verts)
                ptsPD = self._polydata_points_from_indices(node, self._last_indices)

                base = os.path.splitext(fname)[0]

                # Save fiducials (.mrk.json)
                if makeFiducials:
                    outFids = None
                    try:
                        outFids = self.apply_indices_to_fiducials(
                            node,
                            templateFidNode,
                            outputName=f"{base}_AppliedFiducials"
                        )
                        outPathFids = os.path.join(outputDir, base + ".mrk.json")
                        self._save_fiducials(outFids, outPathFids)
                        saved += 1
                    finally:
                        if outFids is not None:
                            slicer.mrmlScene.RemoveNode(outFids)

                # Save point cloud
                if makePointCloud:
                    outPathPts = os.path.join(outputDir, base + ".vtk")
                    self._save_polydata(ptsPD, outPathPts)
                    saved += 1

                # Save Delaunay
                if makeDelaunay:
                    if str(delaunayMode).startswith("3D"):
                        surfPD = self.triangulate_points_delaunay_3d_surface(
                            ptsPD, alpha=float(delaunayAlpha), clean=bool(delaunayClean)
                        )
                    else:
                        surfPD = self.triangulate_points_delaunay_2d(
                            ptsPD, mode=str(delaunayMode), alpha=float(delaunayAlpha), clean=bool(delaunayClean)
                        )

                    outPathDel = os.path.join(outputDir, base + "_delaunay.vtk")
                    self._save_polydata(surfPD, outPathDel)
                    saved += 1

            finally:
                processed += 1
                slicer.mrmlScene.RemoveNode(node)

        return processed, saved


