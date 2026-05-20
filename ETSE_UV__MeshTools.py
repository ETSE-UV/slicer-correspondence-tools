import os
import json
import numpy as np
import slicer
import vtk
import qt
import ctk
from slicer.ScriptedLoadableModule import *


# ------------------------------------------------------------
# Module metadata
# ------------------------------------------------------------
class ETSE_UV__MeshTools(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Mesh Tools"
        parent.categories = ["ETSE_UV"]
        parent.contributors = ["ETSE-UV"]
        parent.helpText = """
        <p>Utility tools for ear mesh post-processing.</p>

        <p><b>Implemented:</b></p>
        <ul>
          <li>Mirror a mesh along X/Y/Z axes in world/RAS coordinates.</li>
          <li>Optionally reverse surface orientation/normals after mirroring.</li>
          <li>Center a mesh on a target world coordinate using either vertex indices or fiducials.</li>
          <li>Batch processing for folders of meshes.</li>
        </ul>

        <p><b>Planned placeholders:</b></p>
        <ul>
          <li>Measured-based true ear scaling.</li>
          <li>Physical displacement from CSV/SOFA/metadata.</li>
        </ul>
        """
        parent.acknowledgementText = (
            "Developed by Juan Antonio De Rus Arance at the Escola Tècnica Superior "
            "d'Enginyeria (ETSE-UV), Universitat de València, in the context of the "
            "Signal Processing & Acoustic Technology (SPAT) research group. "
            "Thanks to the 3D Slicer, SlicerMorph, VTK, NumPy, and open-source communities."
        )


# ------------------------------------------------------------
# Widget
# ------------------------------------------------------------
class ETSE_UV__MeshToolsWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = ETSE_UV__MeshToolsLogic()

        # ------------------------------------------------------------
        # Input mesh
        # ------------------------------------------------------------
        inputBox = ctk.ctkCollapsibleButton()
        inputBox.text = "Input mesh"
        self.layout.addWidget(inputBox)
        inputForm = qt.QFormLayout(inputBox)

        self.inputModelSelector = slicer.qMRMLNodeComboBox()
        self.inputModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.inputModelSelector.selectNodeUponCreation = True
        self.inputModelSelector.addEnabled = False
        self.inputModelSelector.removeEnabled = False
        self.inputModelSelector.noneEnabled = True
        self.inputModelSelector.setMRMLScene(slicer.mrmlScene)
        self.inputModelSelector.setToolTip("Mesh to process. Output meshes are created in world/RAS coordinates with no parent transform.")
        inputForm.addRow("Input mesh:", self.inputModelSelector)

        self.outputNameEdit = qt.QLineEdit("processed_mesh")
        self.outputNameEdit.setToolTip("Name for single-operation output model nodes.")
        inputForm.addRow("Output node name:", self.outputNameEdit)

        # ------------------------------------------------------------
        # Mirror
        # ------------------------------------------------------------
        mirrorBox = ctk.ctkCollapsibleButton()
        mirrorBox.text = "Mirror mesh"
        self.layout.addWidget(mirrorBox)
        mForm = qt.QFormLayout(mirrorBox)

        axisWidget = qt.QWidget()
        axisLayout = qt.QHBoxLayout(axisWidget)
        axisLayout.setContentsMargins(0, 0, 0, 0)
        self.mirrorXCheck = qt.QCheckBox("X")
        self.mirrorYCheck = qt.QCheckBox("Y")
        self.mirrorZCheck = qt.QCheckBox("Z")
        self.mirrorXCheck.checked = False
        self.mirrorYCheck.checked = False
        self.mirrorZCheck.checked = True
        axisLayout.addWidget(self.mirrorXCheck)
        axisLayout.addWidget(self.mirrorYCheck)
        axisLayout.addWidget(self.mirrorZCheck)
        axisLayout.addStretch(1)
        mForm.addRow("Flip axis/axes:", axisWidget)

        self.flipNormalsCheck = qt.QCheckBox("Flip normals / reverse surface orientation when needed")
        self.flipNormalsCheck.checked = True
        self.flipNormalsCheck.setToolTip(
            "Recommended when mirroring one axis. The tool reverses cell winding/normals only when the mirror transform changes handedness."
        )
        mForm.addRow(self.flipNormalsCheck)

        self.mirrorBtn = qt.QPushButton("Create mirrored mesh")
        self.mirrorBtn.clicked.connect(self.onMirrorMesh)
        mForm.addRow(self.mirrorBtn)

        # ------------------------------------------------------------
        # Center
        # ------------------------------------------------------------
        centerBox = ctk.ctkCollapsibleButton()
        centerBox.text = "Center mesh on a world coordinate"
        self.layout.addWidget(centerBox)
        cForm = qt.QFormLayout(centerBox)

        targetWidget = qt.QWidget()
        targetLayout = qt.QHBoxLayout(targetWidget)
        targetLayout.setContentsMargins(0, 0, 0, 0)
        self.targetXSpin = qt.QDoubleSpinBox()
        self.targetYSpin = qt.QDoubleSpinBox()
        self.targetZSpin = qt.QDoubleSpinBox()
        for s in (self.targetXSpin, self.targetYSpin, self.targetZSpin):
            s.setRange(-1e9, 1e9)
            s.setDecimals(6)
            s.setSingleStep(1.0)
            s.setValue(0.0)
        targetLayout.addWidget(qt.QLabel("X")); targetLayout.addWidget(self.targetXSpin)
        targetLayout.addWidget(qt.QLabel("Y")); targetLayout.addWidget(self.targetYSpin)
        targetLayout.addWidget(qt.QLabel("Z")); targetLayout.addWidget(self.targetZSpin)
        cForm.addRow("Target world/RAS coordinate:", targetWidget)

        self.centerSourceCombo = qt.QComboBox()
        self.centerSourceCombo.addItems(["Stored indices", "Fiducials"])
        self.centerSourceCombo.setCurrentText("Stored indices")
        self.centerSourceCombo.setToolTip("How to compute the anchor point. For canal centering, use the mean of the selected canal points.")
        cForm.addRow("Anchor source:", self.centerSourceCombo)

        # Indices controls
        idxRow = qt.QHBoxLayout()
        self.indicesPathEdit = qt.QLineEdit("")
        self.loadIndicesBtn = qt.QPushButton("Load indices…")
        self.loadIndicesBtn.clicked.connect(self.onLoadIndices)
        idxRow.addWidget(self.indicesPathEdit)
        idxRow.addWidget(self.loadIndicesBtn)
        idxWidget = qt.QWidget()
        idxWidget.setLayout(idxRow)
        cForm.addRow("Indices file (.npy/.npz):", idxWidget)

        self.indicesOneBasedCheck = qt.QCheckBox("Loaded indices are 1-based")
        self.indicesOneBasedCheck.checked = False
        self.indicesOneBasedCheck.setToolTip("Leave unchecked for indices saved by the Fiducial Indexer, which are 0-based vertex ids.")
        cForm.addRow(self.indicesOneBasedCheck)

        self.storedIndicesLabel = qt.QLabel("Stored indices: none")
        cForm.addRow(self.storedIndicesLabel)

        # Fiducial controls
        self.fidSelector = slicer.qMRMLNodeComboBox()
        self.fidSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.fidSelector.selectNodeUponCreation = True
        self.fidSelector.addEnabled = False
        self.fidSelector.removeEnabled = False
        self.fidSelector.noneEnabled = True
        self.fidSelector.setMRMLScene(slicer.mrmlScene)
        self.fidSelector.setToolTip("Fiducials used to compute the anchor center. Positions are read in world/RAS coordinates.")
        cForm.addRow("Fiducials:", self.fidSelector)

        self.fidRangeEdit = qt.QLineEdit("1-4")
        self.fidRangeEdit.setToolTip("1-based fiducial range used to compute the center, e.g. canal points: 1-4.")
        cForm.addRow("Fiducial range:", self.fidRangeEdit)

        self.centerBtn = qt.QPushButton("Create centered mesh")
        self.centerBtn.clicked.connect(self.onCenterMesh)
        cForm.addRow(self.centerBtn)

        self.previewCenterBtn = qt.QPushButton("Preview computed anchor center")
        self.previewCenterBtn.clicked.connect(self.onPreviewCenter)
        cForm.addRow(self.previewCenterBtn)

        # ------------------------------------------------------------
        # Combined single operation
        # ------------------------------------------------------------
        combinedBox = ctk.ctkCollapsibleButton()
        combinedBox.text = "Combined single operation"
        self.layout.addWidget(combinedBox)
        combForm = qt.QFormLayout(combinedBox)

        self.applyMirrorInCombinedCheck = qt.QCheckBox("Apply mirror first")
        self.applyMirrorInCombinedCheck.checked = False
        combForm.addRow(self.applyMirrorInCombinedCheck)

        self.applyCenterInCombinedCheck = qt.QCheckBox("Apply centering after mirror")
        self.applyCenterInCombinedCheck.checked = True
        combForm.addRow(self.applyCenterInCombinedCheck)

        self.combinedBtn = qt.QPushButton("Create mesh with selected combined operations")
        self.combinedBtn.setToolTip("Order is always: mirror first, then center. This keeps the chosen anchor at the target coordinate after mirroring.")
        self.combinedBtn.clicked.connect(self.onCombinedSingle)
        combForm.addRow(self.combinedBtn)

        # ------------------------------------------------------------
        # Batch
        # ------------------------------------------------------------
        batchBox = ctk.ctkCollapsibleButton()
        batchBox.text = "Batch processing folder"
        self.layout.addWidget(batchBox)
        bForm = qt.QFormLayout(batchBox)

        rowIn = qt.QHBoxLayout()
        self.batchInputDirEdit = qt.QLineEdit("")
        btnIn = qt.QPushButton("Browse…")
        btnIn.clicked.connect(self.onBrowseBatchInput)
        rowIn.addWidget(self.batchInputDirEdit)
        rowIn.addWidget(btnIn)
        wIn = qt.QWidget()
        wIn.setLayout(rowIn)
        bForm.addRow("Input folder (.vtk/.vtp/.ply):", wIn)

        rowOut = qt.QHBoxLayout()
        self.batchOutputDirEdit = qt.QLineEdit("")
        btnOut = qt.QPushButton("Browse…")
        btnOut.clicked.connect(self.onBrowseBatchOutput)
        rowOut.addWidget(self.batchOutputDirEdit)
        rowOut.addWidget(btnOut)
        wOut = qt.QWidget()
        wOut.setLayout(rowOut)
        bForm.addRow("Output folder:", wOut)

        self.batchOutputExtCombo = qt.QComboBox()
        self.batchOutputExtCombo.addItems([".vtk", ".vtp", ".ply"])
        self.batchOutputExtCombo.setCurrentText(".vtk")
        bForm.addRow("Output extension:", self.batchOutputExtCombo)

        self.batchMirrorCheck = qt.QCheckBox("Mirror meshes")
        self.batchMirrorCheck.checked = False
        bForm.addRow(self.batchMirrorCheck)

        self.batchCenterCheck = qt.QCheckBox("Center meshes")
        self.batchCenterCheck.checked = True
        bForm.addRow(self.batchCenterCheck)

        self.batchCenterSourceCombo = qt.QComboBox()
        self.batchCenterSourceCombo.addItems(["Same loaded indices for all meshes", "Matched fiducials folder (.mrk.json)"])
        self.batchCenterSourceCombo.setCurrentText("Same loaded indices for all meshes")
        bForm.addRow("Batch centering source:", self.batchCenterSourceCombo)

        rowFids = qt.QHBoxLayout()
        self.batchFiducialsDirEdit = qt.QLineEdit("")
        btnFids = qt.QPushButton("Browse…")
        btnFids.clicked.connect(self.onBrowseBatchFiducials)
        rowFids.addWidget(self.batchFiducialsDirEdit)
        rowFids.addWidget(btnFids)
        wFids = qt.QWidget()
        wFids.setLayout(rowFids)
        bForm.addRow("Fiducials folder:", wFids)

        self.batchRunBtn = qt.QPushButton("RUN batch")
        self.batchRunBtn.setToolTip("Order is always: mirror first, then center. Batch uses the mirror/center settings above.")
        self.batchRunBtn.clicked.connect(self.onRunBatch)
        bForm.addRow(self.batchRunBtn)

        # ------------------------------------------------------------
        # Future placeholders
        # ------------------------------------------------------------
        futureBox = ctk.ctkCollapsibleButton()
        futureBox.text = "Future physical scale / displacement placeholders"
        self.layout.addWidget(futureBox)
        fForm = qt.QFormLayout(futureBox)

        self.placeholderScalingBtn = qt.QPushButton("TODO: Measured-based true ear scaling")
        self.placeholderScalingBtn.enabled = False
        self.placeholderScalingBtn.setToolTip("Placeholder only. Later: compute physical scale from tabular/anthropometric measurements.")
        fForm.addRow(self.placeholderScalingBtn)

        self.placeholderDisplaceBtn = qt.QPushButton("TODO: Displacement from CSV / SOFA / metadata")
        self.placeholderDisplaceBtn.enabled = False
        self.placeholderDisplaceBtn.setToolTip("Placeholder only. Later: restore physical translation using available metadata.")
        fForm.addRow(self.placeholderDisplaceBtn)

        self.layout.addStretch(1)
        self._updateStoredIndicesLabel()

    # ------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------
    def _mirror_axes(self):
        return bool(self.mirrorXCheck.checked), bool(self.mirrorYCheck.checked), bool(self.mirrorZCheck.checked)

    def _target_coord(self):
        return np.array([
            float(self.targetXSpin.value),
            float(self.targetYSpin.value),
            float(self.targetZSpin.value),
        ], dtype=float)

    def _updateStoredIndicesLabel(self):
        ids = self.logic.get_indices()
        if ids is None or len(ids) == 0:
            self.storedIndicesLabel.setText("Stored indices: none")
        else:
            self.storedIndicesLabel.setText(f"Stored indices: {len(ids)}")

    def _center_mode(self):
        return str(self.centerSourceCombo.currentText)

    # ------------------------------------------------------------
    # Actions: mirror
    # ------------------------------------------------------------
    def onMirrorMesh(self):
        node = self.inputModelSelector.currentNode()
        if node is None:
            slicer.util.errorDisplay("Select an input mesh.")
            return
        flipX, flipY, flipZ = self._mirror_axes()
        if not (flipX or flipY or flipZ):
            slicer.util.errorDisplay("Select at least one axis to mirror.")
            return
        try:
            poly = self.logic.model_polydata_in_world(node)
            outPD = self.logic.mirror_polydata(
                poly,
                flipX=flipX,
                flipY=flipY,
                flipZ=flipZ,
                flipNormals=bool(self.flipNormalsCheck.checked),
            )
            outName = self.outputNameEdit.text.strip() or f"{node.GetName()}_mirrored"
            outNode = self.logic.add_model_from_polydata(outPD, outName)
            slicer.util.infoDisplay(f"Created mirrored mesh: {outNode.GetName()}")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to mirror mesh: {e}")

    # ------------------------------------------------------------
    # Actions: center
    # ------------------------------------------------------------
    def onLoadIndices(self):
        path = qt.QFileDialog.getOpenFileName(self.parent, "Load vertex indices", "", "NumPy indices (*.npy *.npz)")
        if not path:
            return
        try:
            ids = self.logic.load_indices(path, one_based=bool(self.indicesOneBasedCheck.checked))
            self.indicesPathEdit.setText(path)
            self._updateStoredIndicesLabel()
            slicer.util.infoDisplay(f"Loaded {len(ids)} indices.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load indices: {e}")

    def _compute_anchor_for_current_input(self, poly_for_indices=None):
        mode = self._center_mode()
        node = self.inputModelSelector.currentNode()
        if node is None:
            raise RuntimeError("Select an input mesh.")

        if mode == "Stored indices":
            ids = self.logic.get_indices()
            if ids is None or len(ids) == 0:
                raise RuntimeError("Load an indices file first.")
            if poly_for_indices is None:
                poly_for_indices = self.logic.model_polydata_in_world(node)
            return self.logic.anchor_center_from_indices(poly_for_indices, ids)

        fidNode = self.fidSelector.currentNode()
        if fidNode is None:
            raise RuntimeError("Select a fiducial node.")
        return self.logic.anchor_center_from_fiducials(fidNode, self.fidRangeEdit.text)

    def onPreviewCenter(self):
        try:
            anchor = self._compute_anchor_for_current_input()
            slicer.util.infoDisplay(
                "Computed anchor center (world/RAS):\n"
                f"X={anchor[0]:.6f}, Y={anchor[1]:.6f}, Z={anchor[2]:.6f}"
            )
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def onCenterMesh(self):
        node = self.inputModelSelector.currentNode()
        if node is None:
            slicer.util.errorDisplay("Select an input mesh.")
            return
        try:
            poly = self.logic.model_polydata_in_world(node)
            anchor = self._compute_anchor_for_current_input(poly_for_indices=poly)
            outPD, translation = self.logic.center_polydata(poly, anchor, self._target_coord())
            outName = self.outputNameEdit.text.strip() or f"{node.GetName()}_centered"
            outNode = self.logic.add_model_from_polydata(outPD, outName)
            slicer.util.infoDisplay(
                f"Created centered mesh: {outNode.GetName()}\n"
                f"Anchor: [{anchor[0]:.6f}, {anchor[1]:.6f}, {anchor[2]:.6f}]\n"
                f"Translation: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]"
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to center mesh: {e}")

    # ------------------------------------------------------------
    # Actions: combined single
    # ------------------------------------------------------------
    def onCombinedSingle(self):
        node = self.inputModelSelector.currentNode()
        if node is None:
            slicer.util.errorDisplay("Select an input mesh.")
            return
        doMirror = bool(self.applyMirrorInCombinedCheck.checked)
        doCenter = bool(self.applyCenterInCombinedCheck.checked)
        if not (doMirror or doCenter):
            slicer.util.errorDisplay("Enable at least one combined operation.")
            return

        try:
            poly = self.logic.model_polydata_in_world(node)

            if doMirror:
                flipX, flipY, flipZ = self._mirror_axes()
                if not (flipX or flipY or flipZ):
                    raise RuntimeError("Mirror is enabled, but no axis is selected.")
                poly = self.logic.mirror_polydata(
                    poly,
                    flipX=flipX,
                    flipY=flipY,
                    flipZ=flipZ,
                    flipNormals=bool(self.flipNormalsCheck.checked),
                )

            info = ""
            if doCenter:
                if self._center_mode() == "Stored indices":
                    ids = self.logic.get_indices()
                    if ids is None or len(ids) == 0:
                        raise RuntimeError("Load an indices file first.")
                    anchor = self.logic.anchor_center_from_indices(poly, ids)
                else:
                    anchor = self._compute_anchor_for_current_input()
                    if doMirror:
                        # If centering from fiducials after mirroring, mirror the fiducial-derived anchor consistently.
                        anchor = self.logic.mirror_point(anchor, *self._mirror_axes())
                poly, translation = self.logic.center_polydata(poly, anchor, self._target_coord())
                info = (
                    f"\nAnchor after mirror/order: [{anchor[0]:.6f}, {anchor[1]:.6f}, {anchor[2]:.6f}]"
                    f"\nTranslation: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]"
                )

            outName = self.outputNameEdit.text.strip() or f"{node.GetName()}_meshTools"
            outNode = self.logic.add_model_from_polydata(poly, outName)
            slicer.util.infoDisplay(f"Created processed mesh: {outNode.GetName()}{info}")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to process mesh: {e}")

    # ------------------------------------------------------------
    # Batch actions
    # ------------------------------------------------------------
    def onBrowseBatchInput(self):
        d = qt.QFileDialog.getExistingDirectory(self.parent, "Select input folder with meshes")
        if d:
            self.batchInputDirEdit.setText(d)

    def onBrowseBatchOutput(self):
        d = qt.QFileDialog.getExistingDirectory(self.parent, "Select output folder")
        if d:
            self.batchOutputDirEdit.setText(d)

    def onBrowseBatchFiducials(self):
        d = qt.QFileDialog.getExistingDirectory(self.parent, "Select folder with matched .mrk.json fiducials")
        if d:
            self.batchFiducialsDirEdit.setText(d)

    def onRunBatch(self):
        inDir = self.batchInputDirEdit.text.strip()
        outDir = self.batchOutputDirEdit.text.strip()
        if not inDir or not os.path.isdir(inDir):
            slicer.util.errorDisplay("Pick a valid input folder.")
            return
        if not outDir or not os.path.isdir(outDir):
            slicer.util.errorDisplay("Pick a valid output folder.")
            return

        doMirror = bool(self.batchMirrorCheck.checked)
        doCenter = bool(self.batchCenterCheck.checked)
        if not (doMirror or doCenter):
            slicer.util.errorDisplay("Enable at least one batch operation: mirror and/or center.")
            return

        flipX, flipY, flipZ = self._mirror_axes()
        if doMirror and not (flipX or flipY or flipZ):
            slicer.util.errorDisplay("Batch mirror is enabled, but no mirror axis is selected.")
            return

        centerSource = str(self.batchCenterSourceCombo.currentText)
        if doCenter and centerSource == "Same loaded indices for all meshes":
            ids = self.logic.get_indices()
            if ids is None or len(ids) == 0:
                slicer.util.errorDisplay("Batch centering by indices requires a loaded indices file.")
                return
        if doCenter and centerSource == "Matched fiducials folder (.mrk.json)":
            fDir = self.batchFiducialsDirEdit.text.strip()
            if not fDir or not os.path.isdir(fDir):
                slicer.util.errorDisplay("Pick a valid fiducials folder for matched .mrk.json files.")
                return

        try:
            processed, saved, skipped = self.logic.batch_process_mesh_folder(
                inputDir=inDir,
                outputDir=outDir,
                outputExt=str(self.batchOutputExtCombo.currentText),
                doMirror=doMirror,
                flipX=flipX,
                flipY=flipY,
                flipZ=flipZ,
                flipNormals=bool(self.flipNormalsCheck.checked),
                doCenter=doCenter,
                targetCoord=self._target_coord(),
                centerSource=centerSource,
                indices=self.logic.get_indices(),
                fiducialsDir=self.batchFiducialsDirEdit.text.strip(),
                fiducialRange=self.fidRangeEdit.text,
            )
            slicer.util.infoDisplay(
                f"Batch done. Processed: {processed}  Saved: {saved}  Skipped: {skipped}\n"
                f"Output: {outDir}"
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Batch failed: {e}")


# ------------------------------------------------------------
# Logic
# ------------------------------------------------------------
class ETSE_UV__MeshToolsLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        try:
            super(ETSE_UV__MeshToolsLogic, self).__init__()
        except Exception:
            pass
        self._indices = None

    # ------------------------------------------------------------
    # Indices
    # ------------------------------------------------------------
    def get_indices(self):
        return self._indices

    def load_indices(self, filepath, one_based=False):
        if not filepath or not os.path.isfile(filepath):
            raise ValueError("Invalid indices file path.")
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".npy":
            arr = np.load(filepath)
        elif ext == ".npz":
            z = np.load(filepath)
            keys = list(z.keys())
            if len(keys) == 0:
                raise ValueError("The .npz file contains no arrays.")
            if "indices" in keys:
                key = "indices"
            elif "idx" in keys:
                key = "idx"
            elif "vertex_indices" in keys:
                key = "vertex_indices"
            else:
                key = keys[0]
            arr = z[key]
        else:
            raise ValueError("Unsupported indices format. Use .npy or .npz.")

        ids = np.asarray(arr, dtype=np.int64).ravel()
        if one_based:
            ids = ids - 1
        self._indices = [int(v) for v in ids.tolist()]
        return self._indices

    # ------------------------------------------------------------
    # Coordinate/polydata helpers
    # ------------------------------------------------------------
    def model_polydata_in_world(self, modelNode):
        if modelNode is None:
            raise ValueError("Model node is None.")
        poly = modelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError("Model has no polydata or no points.")

        out = vtk.vtkPolyData()
        out.DeepCopy(poly)

        parentTransformNode = modelNode.GetParentTransformNode()
        if parentTransformNode is not None:
            transform = vtk.vtkGeneralTransform()
            slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(parentTransformNode, None, transform)
            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(transform)
            tf.SetInputData(out)
            tf.Update()
            transformed = vtk.vtkPolyData()
            transformed.DeepCopy(tf.GetOutput())
            transformed.Modified()
            return transformed

        return out

    def add_model_from_polydata(self, polyData, name):
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise ValueError("No polydata to add.")
        outNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        outNode.SetAndObservePolyData(polyData)
        outNode.CreateDefaultDisplayNodes()
        dn = outNode.GetDisplayNode()
        if dn:
            dn.SetBackfaceCulling(False)
            dn.SetOpacity(1.0)
        return outNode

    def _save_polydata(self, polyData, filePath):
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise RuntimeError("No polydata to save.")
        ext = os.path.splitext(filePath)[1].lower()
        if ext not in (".vtk", ".vtp", ".ply"):
            raise RuntimeError(f"Unsupported output extension: {ext} (use .vtk/.vtp/.ply)")

        tmpNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "__tmpMeshToolsSaveModel")
        tmpNode.SetAndObservePolyData(polyData)

        storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelStorageNode", "__tmpMeshToolsSaveStorage")
        storage.SetFileName(filePath)
        if hasattr(storage, "SetCoordinateSystemToRAS"):
            storage.SetCoordinateSystemToRAS()
        else:
            try:
                storage.SetCoordinateSystem(slicer.vtkMRMLStorageNode.CoordinateSystemRAS)
            except Exception:
                pass

        ok = storage.WriteData(tmpNode)
        slicer.mrmlScene.RemoveNode(storage)
        slicer.mrmlScene.RemoveNode(tmpNode)
        if not ok:
            raise RuntimeError(f"Failed to write model: {filePath}")

    # ------------------------------------------------------------
    # Mirror
    # ------------------------------------------------------------
    def mirror_point(self, p, flipX=True, flipY=False, flipZ=False):
        out = np.asarray(p, dtype=float).copy()
        if flipX:
            out[0] *= -1.0
        if flipY:
            out[1] *= -1.0
        if flipZ:
            out[2] *= -1.0
        return out

    def mirror_polydata(self, polyData, flipX=True, flipY=False, flipZ=False, flipNormals=True):
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise ValueError("Input polydata is empty.")
        if not (flipX or flipY or flipZ):
            out = vtk.vtkPolyData()
            out.DeepCopy(polyData)
            return out

        sx = -1.0 if flipX else 1.0
        sy = -1.0 if flipY else 1.0
        sz = -1.0 if flipZ else 1.0

        t = vtk.vtkTransform()
        t.Scale(sx, sy, sz)

        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetTransform(t)
        tf.SetInputData(polyData)
        tf.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(tf.GetOutput())

        # Mirroring an odd number of axes changes handedness and inverts surface orientation.
        odd_number_of_axis_flips = ((1 if flipX else 0) + (1 if flipY else 0) + (1 if flipZ else 0)) % 2 == 1
        if bool(flipNormals) and odd_number_of_axis_flips:
            reverse = vtk.vtkReverseSense()
            reverse.SetInputData(out)
            reverse.ReverseCellsOn()
            reverse.ReverseNormalsOn()
            reverse.Update()
            fixed = vtk.vtkPolyData()
            fixed.DeepCopy(reverse.GetOutput())
            fixed.Modified()
            return fixed

        out.Modified()
        return out

    # ------------------------------------------------------------
    # Centering
    # ------------------------------------------------------------
    def anchor_center_from_indices(self, polyData, indices):
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise ValueError("Polydata is empty.")
        if indices is None or len(indices) == 0:
            raise ValueError("No indices provided.")

        nPts = polyData.GetNumberOfPoints()
        valid = [int(i) for i in indices if 0 <= int(i) < nPts]
        if len(valid) == 0:
            raise ValueError("No valid indices for this mesh.")

        pts = np.zeros((len(valid), 3), dtype=float)
        p = [0.0, 0.0, 0.0]
        for j, vid in enumerate(valid):
            polyData.GetPoint(int(vid), p)
            pts[j, :] = p
        return pts.mean(axis=0)

    def _parse_range_1based_keep_order(self, text):
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

    def anchor_center_from_fiducials(self, fidNode, rangeText="1-4"):
        if fidNode is None or fidNode.GetNumberOfControlPoints() == 0:
            raise ValueError("Fiducial node is empty or None.")

        n = fidNode.GetNumberOfControlPoints()
        if rangeText and str(rangeText).strip():
            ids_1b = self._parse_range_1based_keep_order(rangeText)
            ids = [i - 1 for i in ids_1b if 0 <= i - 1 < n]
        else:
            ids = list(range(n))

        if len(ids) == 0:
            raise ValueError("No valid fiducials selected by the range.")

        pts = np.zeros((len(ids), 3), dtype=float)
        p = [0.0, 0.0, 0.0]
        for j, i in enumerate(ids):
            try:
                fidNode.GetNthControlPointPositionWorld(int(i), p)
            except Exception:
                fidNode.GetNthControlPointPosition(int(i), p)
            pts[j, :] = p
        return pts.mean(axis=0)

    def center_polydata(self, polyData, anchorPoint, targetPoint):
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise ValueError("Input polydata is empty.")
        anchor = np.asarray(anchorPoint, dtype=float).reshape(3)
        target = np.asarray(targetPoint, dtype=float).reshape(3)
        translation = target - anchor

        t = vtk.vtkTransform()
        t.Translate(float(translation[0]), float(translation[1]), float(translation[2]))
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetTransform(t)
        tf.SetInputData(polyData)
        tf.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(tf.GetOutput())
        out.Modified()
        return out, translation

    # ------------------------------------------------------------
    # Batch fiducials from .mrk.json
    # ------------------------------------------------------------
    def _find_matching_markups(self, fiducialsDir, meshBase):
        candidates = [
            meshBase + ".mrk.json",
            meshBase + "_AppliedFiducials.mrk.json",
            meshBase + "_fiducials.mrk.json",
            meshBase + "_Fiducials.mrk.json",
            meshBase + ".json",
        ]
        for c in candidates:
            p = os.path.join(fiducialsDir, c)
            if os.path.isfile(p):
                return p

        # Fallback: any file that starts with the same base and ends with .mrk.json
        for fname in os.listdir(fiducialsDir):
            low = fname.lower()
            if low.endswith(".mrk.json") and fname.startswith(meshBase):
                return os.path.join(fiducialsDir, fname)
        return None

    def _load_anchor_center_from_mrk_json(self, filePath, rangeText="1-4"):
        with open(filePath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "markups" not in data or len(data["markups"]) == 0:
            raise ValueError("No markups found in JSON.")
        mk = data["markups"][0]
        cps = mk.get("controlPoints", [])
        if len(cps) == 0:
            raise ValueError("No control points found in JSON.")

        coordSystem = str(mk.get("coordinateSystem", "RAS")).upper()
        n = len(cps)
        if rangeText and str(rangeText).strip():
            ids_1b = self._parse_range_1based_keep_order(rangeText)
            ids = [i - 1 for i in ids_1b if 0 <= i - 1 < n]
        else:
            ids = list(range(n))
        if len(ids) == 0:
            raise ValueError("No valid control points selected by range.")

        pts = []
        for i in ids:
            pos = cps[int(i)].get("position", None)
            if pos is None or len(pos) < 3:
                continue
            p = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=float)
            # Slicer markups JSON may be stored as LPS. Convert to RAS because model nodes are handled in RAS/world here.
            if coordSystem == "LPS":
                p[0] *= -1.0
                p[1] *= -1.0
            pts.append(p)

        if len(pts) == 0:
            raise ValueError("No valid positions found in selected control points.")
        return np.vstack(pts).mean(axis=0)

    # ------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------
    def batch_process_mesh_folder(self, inputDir, outputDir, outputExt=".vtk",
                                  doMirror=False, flipX=True, flipY=False, flipZ=False, flipNormals=True,
                                  doCenter=True, targetCoord=None, centerSource="Same loaded indices for all meshes",
                                  indices=None, fiducialsDir="", fiducialRange="1-4"):
        if not os.path.isdir(inputDir):
            raise RuntimeError("Invalid input folder.")
        if not os.path.isdir(outputDir):
            raise RuntimeError("Invalid output folder.")

        outputExt = str(outputExt).lower()
        if outputExt not in (".vtk", ".vtp", ".ply"):
            raise RuntimeError("Output extension must be .vtk, .vtp, or .ply.")

        if targetCoord is None:
            targetCoord = np.zeros(3, dtype=float)
        targetCoord = np.asarray(targetCoord, dtype=float).reshape(3)

        exts = (".vtk", ".vtp", ".ply")
        files = [f for f in os.listdir(inputDir)
                 if os.path.isfile(os.path.join(inputDir, f)) and os.path.splitext(f)[1].lower() in exts]
        files.sort()

        processed = 0
        saved = 0
        skipped = 0

        for fname in files:
            inPath = os.path.join(inputDir, fname)
            base = os.path.splitext(fname)[0]
            ok, node = slicer.util.loadModel(inPath, returnNode=True)
            if not ok or node is None:
                processed += 1
                skipped += 1
                continue

            try:
                poly = self.model_polydata_in_world(node)

                if doMirror:
                    poly = self.mirror_polydata(
                        poly,
                        flipX=bool(flipX),
                        flipY=bool(flipY),
                        flipZ=bool(flipZ),
                        flipNormals=bool(flipNormals),
                    )

                if doCenter:
                    if centerSource == "Same loaded indices for all meshes":
                        if indices is None or len(indices) == 0:
                            raise RuntimeError("No indices provided for batch centering.")
                        anchor = self.anchor_center_from_indices(poly, indices)
                    else:
                        if not fiducialsDir or not os.path.isdir(fiducialsDir):
                            raise RuntimeError("Invalid fiducials folder.")
                        markupsPath = self._find_matching_markups(fiducialsDir, base)
                        if markupsPath is None:
                            skipped += 1
                            processed += 1
                            continue
                        anchor = self._load_anchor_center_from_mrk_json(markupsPath, fiducialRange)
                        if doMirror:
                            anchor = self.mirror_point(anchor, bool(flipX), bool(flipY), bool(flipZ))

                    poly, _translation = self.center_polydata(poly, anchor, targetCoord)

                suffix = []
                if doMirror:
                    suffix.append("mirrored")
                if doCenter:
                    suffix.append("centered")
                suffixText = "_" + "_".join(suffix) if suffix else "_processed"
                outPath = os.path.join(outputDir, base + suffixText + outputExt)
                self._save_polydata(poly, outPath)
                saved += 1

            except Exception as e:
                # Keep batch robust. Also print the reason in the Python console.
                print(f"[ETSE-UV Mesh Tools] Skipped {fname}: {e}")
                skipped += 1
            finally:
                processed += 1
                slicer.mrmlScene.RemoveNode(node)

        return processed, saved, skipped
