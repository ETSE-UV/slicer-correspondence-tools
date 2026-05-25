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
        parent.contributors = ["Juan Antonio De Rus Arance (ETSE-UV / SPAT, Universitat de València)"]
        parent.helpText = """
        <p>Utility tools for ear mesh post-processing.</p>

        <p><b>Implemented:</b></p>
        <ul>
          <li>Mirror a mesh along X/Y/Z axes in world/RAS coordinates.</li>
          <li>Optionally reverse surface orientation/normals after mirroring.</li>
          <li>Center a mesh on a target world coordinate using either stored vertex indices or fiducials.</li>
          <li>Preview the target anchor, source anchor points, and computed source anchor center.</li>
          <li>Build anatomical frames from canal and otobasion landmarks and align a target mesh to a reference PGD mesh.</li>
          <li>Preview canal/otobasion points, loop normals, anatomical up vectors, and final frame axes.</li>
          <li>Batch processing for folders of meshes.</li>
        </ul>

        <p><b>Typical workflow:</b></p>
        <ol>
          <li>Load a .npy/.npz file created by ETSE-UV Fiducial Indexer.</li>
          <li>Center the target mesh using canal or selected anchor points.</li>
          <li>Use canal and otobasion landmarks to align target orientation to a selected PGD reference.</li>
        </ol>
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

        self.centerStoredIndexRangeEdit = qt.QLineEdit("1-4")
        self.centerStoredIndexRangeEdit.setToolTip(
            "1-based positions inside the loaded index list used for centering. "
            "Default 1-4 uses only the first four loaded indices, normally the canal points. "
            "This is NOT a mesh vertex-id range."
        )
        cForm.addRow("Stored-index anchor range:", self.centerStoredIndexRangeEdit)

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

        self.previewTargetAnchorBtn = qt.QPushButton("Preview target world anchor")
        self.previewTargetAnchorBtn.setToolTip(
            "Create/update an independent fiducial at the desired target world/RAS coordinate. "
            "By default this is the world origin: [0, 0, 0]."
        )
        self.previewTargetAnchorBtn.clicked.connect(self.onPreviewTargetAnchor)
        cForm.addRow(self.previewTargetAnchorBtn)

        self.previewCenterBtn = qt.QPushButton("Preview source anchor points + center")
        self.previewCenterBtn.setToolTip(
            "Create/update fiducials showing the source points used to compute the anchor, "
            "plus the computed source anchor center and the target world anchor."
        )
        self.previewCenterBtn.clicked.connect(self.onPreviewCenter)
        cForm.addRow(self.previewCenterBtn)

        # ------------------------------------------------------------
        # Anatomical orientation alignment
        # ------------------------------------------------------------
        alignBox = ctk.ctkCollapsibleButton()
        alignBox.text = "Align orientation to reference PGD"
        self.layout.addWidget(alignBox)
        aForm = qt.QFormLayout(alignBox)

        self.referenceModelSelector = slicer.qMRMLNodeComboBox()
        self.referenceModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.referenceModelSelector.selectNodeUponCreation = True
        self.referenceModelSelector.addEnabled = False
        self.referenceModelSelector.removeEnabled = False
        self.referenceModelSelector.noneEnabled = True
        self.referenceModelSelector.setMRMLScene(slicer.mrmlScene)
        self.referenceModelSelector.setToolTip(
            "Reference PGD / mean mesh that defines the desired anatomical orientation."
        )
        aForm.addRow("Reference PGD mesh:", self.referenceModelSelector)

        self.alignSourceCombo = qt.QComboBox()
        self.alignSourceCombo.addItems(["Stored indices", "Fiducials"])
        self.alignSourceCombo.setCurrentText("Stored indices")
        self.alignSourceCombo.setToolTip(
            "Stored indices means a .npy/.npz file created by Fiducial Indexer. "
            "The loaded values are mesh vertex IDs, ordered according to the fiducial labels below."
        )
        aForm.addRow("Alignment landmarks from:", self.alignSourceCombo)

        alignIdxRow = qt.QHBoxLayout()
        self.alignIndicesPathEdit = qt.QLineEdit("")
        self.alignIndicesPathEdit.setToolTip(
            "Dedicated .npy/.npz file for orientation alignment. "
            "This is independent from the centering indices file."
        )
        self.loadAlignIndicesBtn = qt.QPushButton("Load alignment indices…")
        self.loadAlignIndicesBtn.clicked.connect(self.onLoadAlignmentIndices)
        alignIdxRow.addWidget(self.alignIndicesPathEdit)
        alignIdxRow.addWidget(self.loadAlignIndicesBtn)
        alignIdxWidget = qt.QWidget()
        alignIdxWidget.setLayout(alignIdxRow)
        aForm.addRow("Alignment indices file (.npy/.npz):", alignIdxWidget)

        self.alignIndicesOneBasedCheck = qt.QCheckBox("Alignment indices are 1-based")
        self.alignIndicesOneBasedCheck.checked = False
        self.alignIndicesOneBasedCheck.setToolTip(
            "Leave unchecked for files saved by ETSE-UV Fiducial Indexer, which are 0-based vertex ids."
        )
        aForm.addRow(self.alignIndicesOneBasedCheck)

        self.alignStoredIndicesLabel = qt.QLabel("Alignment indices: none")
        aForm.addRow(self.alignStoredIndicesLabel)

        self.loadedIndexLabelsEdit = qt.QLineEdit("1-4,246-261")
        self.loadedIndexLabelsEdit.setToolTip(
            "Fiducial labels represented by the loaded .npy/.npz indices, in order. "
            "Use '1-4,246-261' if the file contains only canal + otobasion landmarks. "
            "Use '1-261' if the file contains all fiducials."
        )
        aForm.addRow("Loaded-index labels:", self.loadedIndexLabelsEdit)

        self.referenceFidSelector = slicer.qMRMLNodeComboBox()
        self.referenceFidSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.referenceFidSelector.selectNodeUponCreation = True
        self.referenceFidSelector.addEnabled = False
        self.referenceFidSelector.removeEnabled = False
        self.referenceFidSelector.noneEnabled = True
        self.referenceFidSelector.setMRMLScene(slicer.mrmlScene)
        self.referenceFidSelector.setToolTip("Reference PGD fiducials, used only when alignment source is Fiducials.")
        aForm.addRow("Reference fiducials:", self.referenceFidSelector)

        self.targetFidSelector = slicer.qMRMLNodeComboBox()
        self.targetFidSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.targetFidSelector.selectNodeUponCreation = True
        self.targetFidSelector.addEnabled = False
        self.targetFidSelector.removeEnabled = False
        self.targetFidSelector.noneEnabled = True
        self.targetFidSelector.setMRMLScene(slicer.mrmlScene)
        self.targetFidSelector.setToolTip("Target fiducials, used only when alignment source is Fiducials.")
        aForm.addRow("Target fiducials:", self.targetFidSelector)

        self.alignCanalLabelsEdit = qt.QLineEdit("1-4")
        self.alignCanalLabelsEdit.setToolTip("Fiducial labels used as canal landmarks.")
        aForm.addRow("Canal labels:", self.alignCanalLabelsEdit)

        self.alignOtobasionLabelsEdit = qt.QLineEdit("246-261")
        self.alignOtobasionLabelsEdit.setToolTip("Fiducial labels used as otobasion landmarks.")
        aForm.addRow("Otobasion labels:", self.alignOtobasionLabelsEdit)

        self.alignUpPairsEdit = qt.QLineEdit("254-246,255-261,253-247")
        self.alignUpPairsEdit.setToolTip(
            "Pairs used to estimate the anatomical UP vector. "
            "Each pair is interpreted as first→second, e.g. 254-246 means P246 - P254."
        )
        aForm.addRow("Up-vector pairs:", self.alignUpPairsEdit)

        weightWidget = qt.QWidget()
        weightLayout = qt.QHBoxLayout(weightWidget)
        weightLayout.setContentsMargins(0, 0, 0, 0)

        self.otobasionNormalWeightSpin = qt.QDoubleSpinBox()
        self.otobasionNormalWeightSpin.setRange(0.0, 100.0)
        self.otobasionNormalWeightSpin.setDecimals(3)
        self.otobasionNormalWeightSpin.setSingleStep(0.1)
        self.otobasionNormalWeightSpin.setValue(0.80)

        self.canalNormalWeightSpin = qt.QDoubleSpinBox()
        self.canalNormalWeightSpin.setRange(0.0, 100.0)
        self.canalNormalWeightSpin.setDecimals(3)
        self.canalNormalWeightSpin.setSingleStep(0.1)
        self.canalNormalWeightSpin.setValue(0.20)

        weightLayout.addWidget(qt.QLabel("Otobasion"))
        weightLayout.addWidget(self.otobasionNormalWeightSpin)
        weightLayout.addWidget(qt.QLabel("Canal"))
        weightLayout.addWidget(self.canalNormalWeightSpin)
        aForm.addRow("Normal weights:", weightWidget)

        self.alignRotationCenterCombo = qt.QComboBox()
        self.alignRotationCenterCombo.addItems([
            "Target world anchor",
            "Target otobasion center",
            "Target canal center",
            "Target combined center",
        ])
        self.alignRotationCenterCombo.setCurrentText("Target canal center")
        self.alignRotationCenterCombo.setToolTip(
            "Point around which the target mesh is rotated. "
            "Default: Target canal center, which preserves the canal anchor while fixing orientation."
        )
        aForm.addRow("Rotation center:", self.alignRotationCenterCombo)

        self.alignVectorScaleSpin = qt.QDoubleSpinBox()
        self.alignVectorScaleSpin.setRange(0.001, 1e9)
        self.alignVectorScaleSpin.setDecimals(3)
        self.alignVectorScaleSpin.setSingleStep(1.0)
        self.alignVectorScaleSpin.setValue(20.0)
        self.alignVectorScaleSpin.setToolTip("Length of preview normal/frame vectors in scene units.")
        aForm.addRow("Preview vector length:", self.alignVectorScaleSpin)

        self.previewLegendLabel = qt.QLabel(
            "<b>Preview legend:</b><br>"
            "<span style='color:rgb(0,220,0)'>Green points</span>: reference PGD landmarks. "
            "<span style='color:rgb(255,60,60)'>Red points</span>: target landmarks.<br>"
            "<span style='color:rgb(255,255,0)'>Yellow point</span>: anatomical frame origin.<br>"
            "<span style='color:rgb(255,0,255)'>Magenta line</span>: final combined normal / anatomical Z. "
            "<span style='color:rgb(255,128,0)'>Orange line</span>: otobasion plane normal. "
            "<span style='color:rgb(0,150,255)'>Cyan line</span>: canal plane normal.<br>"
            "Frame axes: red=X, green=Y/up, blue=Z."
        )
        self.previewLegendLabel.setWordWrap(True)
        aForm.addRow(self.previewLegendLabel)

        self.previewLegendBtn = qt.QPushButton("Show preview legend")
        self.previewLegendBtn.clicked.connect(self.onShowAnatomicalPreviewLegend)
        aForm.addRow(self.previewLegendBtn)

        self.previewAnatomicalFrameBtn = qt.QPushButton("Preview anatomical frames and normals")
        self.previewAnatomicalFrameBtn.clicked.connect(self.onPreviewAnatomicalFrames)
        aForm.addRow(self.previewAnatomicalFrameBtn)

        self.alignOrientationBtn = qt.QPushButton("Create orientation-aligned mesh")
        self.alignOrientationBtn.setToolTip(
            "Rotate the input target mesh so its anatomical frame matches the selected reference PGD frame."
        )
        self.alignOrientationBtn.clicked.connect(self.onAlignOrientationToReference)
        aForm.addRow(self.alignOrientationBtn)

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

        self.applyAlignInCombinedCheck = qt.QCheckBox("Apply orientation alignment after centering")
        self.applyAlignInCombinedCheck.checked = True
        self.applyAlignInCombinedCheck.setToolTip(
            "Order is: mirror first, then center, then anatomical orientation alignment to the reference PGD."
        )
        combForm.addRow(self.applyAlignInCombinedCheck)

        self.combinedBtn = qt.QPushButton("Create mesh with selected combined operations")
        self.combinedBtn.setToolTip(
            "Order is always: mirror first, then center, then anatomical orientation alignment. "
            "This keeps the chosen anchor at the target coordinate before orientation correction."
        )
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

        self.batchAlignCheck = qt.QCheckBox("Align orientation to reference PGD")
        self.batchAlignCheck.checked = False
        self.batchAlignCheck.setToolTip(
            "After optional mirror and centering, align each mesh orientation to the reference PGD "
            "using the settings from 'Align orientation to reference PGD'."
        )
        bForm.addRow(self.batchAlignCheck)

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
        self.batchRunBtn.setToolTip("Order is always: mirror first, then center, then orientation alignment. Batch uses the settings above.")
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
        self._updateAlignmentIndicesLabel()


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

    def _updateAlignmentIndicesLabel(self):
        ids = self.logic.get_alignment_indices()
        if ids is None or len(ids) == 0:
            self.alignStoredIndicesLabel.setText("Alignment indices: none")
        else:
            self.alignStoredIndicesLabel.setText(f"Alignment indices: {len(ids)}")

    def _center_mode(self):
        return str(self.centerSourceCombo.currentText)

    def _remove_node_by_name(self, name):
        """Remove all MRML nodes with this exact name."""
        while True:
            node = slicer.mrmlScene.GetFirstNodeByName(name)
            if node is None:
                break
            slicer.mrmlScene.RemoveNode(node)

    def _create_preview_fiducials(self, name, points, labels=None, color=(0.2, 0.8, 1.0), glyphScale=2.0):
        """Create or replace a fiducial preview node."""
        self._remove_node_by_name(name)

        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, 3)

        if labels is None:
            labels = [f"P{i}" for i in range(points.shape[0])]

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.CreateDefaultDisplayNodes()

        for p, label in zip(points, labels):
            try:
                node.AddControlPoint(vtk.vtkVector3d(float(p[0]), float(p[1]), float(p[2])), str(label))
            except Exception:
                node.AddControlPoint([float(p[0]), float(p[1]), float(p[2])], str(label))

        displayNode = node.GetDisplayNode()
        if displayNode:
            try:
                displayNode.SetColor(float(color[0]), float(color[1]), float(color[2]))
                displayNode.SetSelectedColor(float(color[0]), float(color[1]), float(color[2]))
            except Exception:
                pass
            if hasattr(displayNode, "SetGlyphScale"):
                displayNode.SetGlyphScale(float(glyphScale))
            if hasattr(displayNode, "SetTextScale"):
                displayNode.SetTextScale(2.5)
            if hasattr(displayNode, "SetVisibility"):
                displayNode.SetVisibility(True)

        return node

    def _build_alignment_frame_for_current_polydata(self, poly):
        if str(self.alignSourceCombo.currentText) == "Stored indices":
            ids = self.logic.get_alignment_indices()
            if ids is None or len(ids) == 0:
                raise RuntimeError("Load an alignment .npy/.npz indices file first.")

            return self.logic.anatomical_frame_from_stored_indices(
                polyData=poly,
                storedIndices=ids,
                loadedIndexLabels=self.loadedIndexLabelsEdit.text,
                canalLabels=self.alignCanalLabelsEdit.text,
                otobasionLabels=self.alignOtobasionLabelsEdit.text,
                upPairsText=self.alignUpPairsEdit.text,
                otobasionWeight=float(self.otobasionNormalWeightSpin.value),
                canalWeight=float(self.canalNormalWeightSpin.value),
            )

        raise RuntimeError(
            "Combined alignment currently requires 'Stored indices'. "
            "Use the standalone alignment button for fiducial-based alignment."
        )


    def _create_preview_vector(self, name, origin, vector, scale=20.0, color=(1.0, 1.0, 1.0), lineWidth=4):
        """Create or replace a line model previewing a vector from an origin."""
        self._remove_node_by_name(name)

        origin = np.asarray(origin, dtype=float).reshape(3)
        vector = np.asarray(vector, dtype=float).reshape(3)
        n = np.linalg.norm(vector)
        if n < 1e-12:
            raise ValueError(f"Cannot preview zero-length vector: {name}")
        vector = vector / n

        p0 = origin
        p1 = origin + float(scale) * vector

        points = vtk.vtkPoints()
        points.InsertNextPoint(float(p0[0]), float(p0[1]), float(p0[2]))
        points.InsertNextPoint(float(p1[0]), float(p1[1]), float(p1[2]))

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(1, 1)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(cells)
        poly.Modified()

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.SetAndObservePolyData(poly)
        node.CreateDefaultDisplayNodes()

        dn = node.GetDisplayNode()
        if dn:
            dn.SetColor(float(color[0]), float(color[1]), float(color[2]))
            dn.SetLineWidth(int(lineWidth))
            dn.SetVisibility(True)

        return node

    def onPreviewTargetAnchor(self):
        """Preview the desired target anchor as an independent world/RAS fiducial."""
        try:
            target = self._target_coord()
            node = self._create_preview_fiducials(
                name="ETSE_UV_MeshTools_TARGET_WORLD_ANCHOR",
                points=[target],
                labels=["TARGET_WORLD_ANCHOR"],
                color=(0.1, 0.9, 0.1),
                glyphScale=3.0,
            )
            slicer.util.infoDisplay(
                f"Target world anchor preview created:\n"
                f"Node: {node.GetName()}\n"
                f"X={target[0]:.6f}, Y={target[1]:.6f}, Z={target[2]:.6f}"
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to preview target anchor: {e}")

    def _compute_anchor_preview_for_current_input(self, poly_for_indices=None):
        """
        Return anchor center, source points, and labels used for preview.
        Supports both stored indices and fiducials.
        """
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

            centerIds = self.logic.subset_indices_by_position(ids, self.centerStoredIndexRangeEdit.text)
            points, validIds = self.logic.anchor_points_from_indices(poly_for_indices, centerIds)
            labels = [f"idxpos_{j + 1}:v{int(i)}" for j, i in enumerate(validIds)]
            anchor = points.mean(axis=0)
            return anchor, points, labels

        fidNode = self.fidSelector.currentNode()
        if fidNode is None:
            raise RuntimeError("Select a fiducial node.")

        points, validFidIds = self.logic.anchor_points_from_fiducials(fidNode, self.fidRangeEdit.text)
        labels = [f"fid_{int(i) + 1}" for i in validFidIds]
        anchor = points.mean(axis=0)
        return anchor, points, labels

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
            slicer.util.infoDisplay(f"Loaded {len(ids)} centering indices.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load centering indices: {e}")

    def onLoadAlignmentIndices(self):
        path = qt.QFileDialog.getOpenFileName(self.parent, "Load alignment vertex indices", "", "NumPy indices (*.npy *.npz)")
        if not path:
            return
        try:
            ids = self.logic.load_alignment_indices(path, one_based=bool(self.alignIndicesOneBasedCheck.checked))
            self.alignIndicesPathEdit.setText(path)
            self._updateAlignmentIndicesLabel()
            slicer.util.infoDisplay(f"Loaded {len(ids)} alignment indices.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load alignment indices: {e}")

    def _compute_anchor_for_current_input(self, poly_for_indices=None):
        anchor, _points, _labels = self._compute_anchor_preview_for_current_input(
            poly_for_indices=poly_for_indices
        )
        return anchor

    def onPreviewCenter(self):
        try:
            anchor, sourcePoints, sourceLabels = self._compute_anchor_preview_for_current_input()
            target = self._target_coord()

            sourcePointsNode = self._create_preview_fiducials(
                name="ETSE_UV_MeshTools_SOURCE_ANCHOR_POINTS",
                points=sourcePoints,
                labels=sourceLabels,
                color=(0.1, 0.4, 1.0),
                glyphScale=1.5,
            )

            sourceCenterNode = self._create_preview_fiducials(
                name="ETSE_UV_MeshTools_SOURCE_ANCHOR_CENTER",
                points=[anchor],
                labels=["SOURCE_ANCHOR_CENTER"],
                color=(1.0, 0.6, 0.0),
                glyphScale=3.0,
            )

            targetNode = self._create_preview_fiducials(
                name="ETSE_UV_MeshTools_TARGET_WORLD_ANCHOR",
                points=[target],
                labels=["TARGET_WORLD_ANCHOR"],
                color=(0.1, 0.9, 0.1),
                glyphScale=3.0,
            )

            slicer.util.infoDisplay(
                f"Anchor preview created.\n\n"
                f"Source anchor points node: {sourcePointsNode.GetName()}\n"
                f"Source anchor center node: {sourceCenterNode.GetName()}\n"
                f"Target world anchor node: {targetNode.GetName()}\n\n"
                f"Source anchor center:\n"
                f"X={anchor[0]:.6f}, Y={anchor[1]:.6f}, Z={anchor[2]:.6f}\n\n"
                f"Target world anchor:\n"
                f"X={target[0]:.6f}, Y={target[1]:.6f}, Z={target[2]:.6f}"
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to preview anchor: {e}")

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
        doAlign = bool(self.applyAlignInCombinedCheck.checked)

        if not (doMirror or doCenter or doAlign):
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
                        raise RuntimeError("Load a centering indices file first.")
                    centerIds = self.logic.subset_indices_by_position(ids, self.centerStoredIndexRangeEdit.text)
                    anchor = self.logic.anchor_center_from_indices(poly, centerIds)
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
            if doAlign:
                refNode = self.referenceModelSelector.currentNode()
                if refNode is None:
                    raise RuntimeError("Select a reference PGD mesh for orientation alignment.")

                _refPoly, refFrame = self._build_alignment_frame_for_model(
                    refNode,
                    fidNode=self.referenceFidSelector.currentNode(),
                )

                targetFrame = self._build_alignment_frame_for_current_polydata(poly)
                rotation = self.logic.rotation_from_frame_to_frame(targetFrame, refFrame)

                rotationCenterMode = str(self.alignRotationCenterCombo.currentText)
                if rotationCenterMode == "Target world anchor":
                    rotationCenter = self._target_coord()
                elif rotationCenterMode == "Target otobasion center":
                    rotationCenter = targetFrame["otobasion_center"]
                elif rotationCenterMode == "Target canal center":
                    rotationCenter = targetFrame["canal_center"]
                elif rotationCenterMode == "Target combined center":
                    rotationCenter = targetFrame["origin"]
                else:
                    rotationCenter = targetFrame["canal_center"]

                poly = self.logic.rotate_polydata_about_point(poly, rotation, rotationCenter)

                angleDeg = self.logic.rotation_angle_degrees(rotation)
                info += (
                    f"\nOrientation alignment angle: {angleDeg:.3f} degrees"
                    f"\nRotation center: [{rotationCenter[0]:.6f}, {rotationCenter[1]:.6f}, {rotationCenter[2]:.6f}]"
                )

            outName = self.outputNameEdit.text.strip() or f"{node.GetName()}_meshTools"
            outNode = self.logic.add_model_from_polydata(poly, outName)
            slicer.util.infoDisplay(f"Created processed mesh: {outNode.GetName()}{info}")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to process mesh: {e}")

    # ------------------------------------------------------------
    # Actions: anatomical orientation alignment
    # ------------------------------------------------------------
    def _build_alignment_frame_for_model(self, modelNode, fidNode=None):
        if modelNode is None:
            raise RuntimeError("Select a model.")

        poly = self.logic.model_polydata_in_world(modelNode)

        if str(self.alignSourceCombo.currentText) == "Stored indices":
            ids = self.logic.get_alignment_indices()
            if ids is None or len(ids) == 0:
                raise RuntimeError("Load a dedicated alignment .npy/.npz indices file first.")

            frame = self.logic.anatomical_frame_from_stored_indices(
                polyData=poly,
                storedIndices=ids,
                loadedIndexLabels=self.loadedIndexLabelsEdit.text,
                canalLabels=self.alignCanalLabelsEdit.text,
                otobasionLabels=self.alignOtobasionLabelsEdit.text,
                upPairsText=self.alignUpPairsEdit.text,
                otobasionWeight=float(self.otobasionNormalWeightSpin.value),
                canalWeight=float(self.canalNormalWeightSpin.value),
            )
            return poly, frame

        if fidNode is None:
            raise RuntimeError("Select fiducials for this model.")

        frame = self.logic.anatomical_frame_from_fiducials(
            fidNode=fidNode,
            canalLabels=self.alignCanalLabelsEdit.text,
            otobasionLabels=self.alignOtobasionLabelsEdit.text,
            upPairsText=self.alignUpPairsEdit.text,
            otobasionWeight=float(self.otobasionNormalWeightSpin.value),
            canalWeight=float(self.canalNormalWeightSpin.value),
        )
        return poly, frame

    def _alignment_rotation_center(self, targetFrame):
        mode = str(self.alignRotationCenterCombo.currentText)
        if mode == "Target world anchor":
            return self._target_coord()
        if mode == "Target otobasion center":
            return targetFrame["otobasion_center"]
        if mode == "Target canal center":
            return targetFrame["canal_center"]
        if mode == "Target combined center":
            return targetFrame["origin"]
        return self._target_coord()

    def _preview_frame(self, prefix, frame, pointColor, vectorScale):
        self._create_preview_fiducials(
            f"ETSE_UV_MeshTools_{prefix}_CANAL_POINTS",
            frame["canal_points"],
            labels=[f"C_{label}" for label in frame["canal_labels"]],
            color=pointColor,
            glyphScale=1.5,
        )
        self._create_preview_fiducials(
            f"ETSE_UV_MeshTools_{prefix}_OTOBASION_POINTS",
            frame["otobasion_points"],
            labels=[f"O_{label}" for label in frame["otobasion_labels"]],
            color=pointColor,
            glyphScale=1.5,
        )
        self._create_preview_fiducials(
            f"ETSE_UV_MeshTools_{prefix}_FRAME_ORIGIN",
            [frame["origin"]],
            labels=[f"{prefix}_FRAME_ORIGIN"],
            color=(1.0, 1.0, 0.0),
            glyphScale=3.0,
        )

        origin = frame["origin"]
        self._create_preview_vector(f"ETSE_UV_MeshTools_{prefix}_UP_VECTOR", origin, frame["up"], scale=vectorScale, color=(0.0, 1.0, 0.0), lineWidth=5)
        self._create_preview_vector(f"ETSE_UV_MeshTools_{prefix}_COMBINED_NORMAL", origin, frame["normal"], scale=vectorScale, color=(1.0, 0.0, 1.0), lineWidth=5)
        self._create_preview_vector(f"ETSE_UV_MeshTools_{prefix}_CANAL_NORMAL", frame["canal_center"], frame["canal_normal"], scale=vectorScale * 0.75, color=(0.0, 0.6, 1.0), lineWidth=3)
        self._create_preview_vector(f"ETSE_UV_MeshTools_{prefix}_OTOBASION_NORMAL", frame["otobasion_center"], frame["otobasion_normal"], scale=vectorScale * 0.75, color=(1.0, 0.4, 0.0), lineWidth=3)
        self._create_preview_vector(f"ETSE_UV_MeshTools_{prefix}_FRAME_X", origin, frame["x"], scale=vectorScale * 0.6, color=(1.0, 0.0, 0.0), lineWidth=3)
        self._create_preview_vector(f"ETSE_UV_MeshTools_{prefix}_FRAME_Y", origin, frame["y"], scale=vectorScale * 0.6, color=(0.0, 1.0, 0.0), lineWidth=3)
        self._create_preview_vector(f"ETSE_UV_MeshTools_{prefix}_FRAME_Z", origin, frame["z"], scale=vectorScale * 0.6, color=(0.0, 0.0, 1.0), lineWidth=3)

    def onShowAnatomicalPreviewLegend(self):
        slicer.util.infoDisplay(
            "Anatomical frame preview legend:\n\n"
            "Landmark points:\n"
            "  - Green points: reference PGD canal/otobasion landmarks.\n"
            "  - Red points: target canal/otobasion landmarks.\n"
            "  - Yellow point: anatomical frame origin, halfway between canal and otobasion centers.\n\n"
            "Normal/vector lines:\n"
            "  - Magenta: final combined anatomical normal, weighted from otobasion + canal normals.\n"
            "  - Orange: otobasion best-fit plane normal.\n"
            "  - Cyan/blue: canal best-fit plane normal.\n"
            "  - Green: anatomical UP vector from the configured otobasion pairs.\n\n"
            "Frame axes:\n"
            "  - Red: frame X axis.\n"
            "  - Green: frame Y axis / UP.\n"
            "  - Blue: frame Z axis / final anatomical normal.\n\n"
            "Alignment rotates the target frame so its red/green/blue axes match the reference frame.",
            windowTitle="ETSE-UV Mesh Tools: anatomical frame preview legend",
        )

    def onPreviewAnatomicalFrames(self):
        refNode = self.referenceModelSelector.currentNode()
        targetNode = self.inputModelSelector.currentNode()
        if refNode is None:
            slicer.util.errorDisplay("Select a reference PGD mesh.")
            return
        if targetNode is None:
            slicer.util.errorDisplay("Select an input/target mesh.")
            return

        try:
            _refPoly, refFrame = self._build_alignment_frame_for_model(refNode, fidNode=self.referenceFidSelector.currentNode())
            _targetPoly, targetFrame = self._build_alignment_frame_for_model(targetNode, fidNode=self.targetFidSelector.currentNode())
            scale = float(self.alignVectorScaleSpin.value)
            self._preview_frame("REFERENCE", refFrame, pointColor=(0.0, 1.0, 0.0), vectorScale=scale)
            self._preview_frame("TARGET", targetFrame, pointColor=(1.0, 0.2, 0.2), vectorScale=scale)
            rotation = self.logic.rotation_from_frame_to_frame(targetFrame, refFrame)
            angleDeg = self.logic.rotation_angle_degrees(rotation)
            slicer.util.infoDisplay(
                f"Anatomical frame preview created.\n\n"
                f"Required full rotation angle: {angleDeg:.3f} degrees\n\n"
                f"Green points/vectors: reference PGD\n"
                f"Red points/vectors: target mesh\n"
                f"Magenta: final combined normal\n"
                f"Orange: otobasion normal\n"
                f"Blue/cyan: canal normal"
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to preview anatomical frames: {e}")

    def onAlignOrientationToReference(self):
        refNode = self.referenceModelSelector.currentNode()
        targetNode = self.inputModelSelector.currentNode()
        if refNode is None:
            slicer.util.errorDisplay("Select a reference PGD mesh.")
            return
        if targetNode is None:
            slicer.util.errorDisplay("Select an input/target mesh.")
            return

        try:
            _refPoly, refFrame = self._build_alignment_frame_for_model(refNode, fidNode=self.referenceFidSelector.currentNode())
            targetPoly, targetFrame = self._build_alignment_frame_for_model(targetNode, fidNode=self.targetFidSelector.currentNode())
            rotation = self.logic.rotation_from_frame_to_frame(targetFrame, refFrame)
            rotationCenter = self._alignment_rotation_center(targetFrame)
            outPD = self.logic.rotate_polydata_about_point(targetPoly, rotation, rotationCenter)
            outName = self.outputNameEdit.text.strip() or f"{targetNode.GetName()}_alignedToPGD"
            outNode = self.logic.add_model_from_polydata(outPD, outName)
            angleDeg = self.logic.rotation_angle_degrees(rotation)
            slicer.util.infoDisplay(
                f"Created orientation-aligned mesh: {outNode.GetName()}\n"
                f"Full anatomical-frame rotation angle: {angleDeg:.3f} degrees\n"
                f"Rotation center: [{rotationCenter[0]:.6f}, {rotationCenter[1]:.6f}, {rotationCenter[2]:.6f}]"
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to align orientation: {e}")

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
        doAlign = bool(self.batchAlignCheck.checked)
        if not (doMirror or doCenter or doAlign):
            slicer.util.errorDisplay("Enable at least one batch operation: mirror, center, and/or align.")
            return

        flipX, flipY, flipZ = self._mirror_axes()
        if doMirror and not (flipX or flipY or flipZ):
            slicer.util.errorDisplay("Batch mirror is enabled, but no mirror axis is selected.")
            return

        centerSource = str(self.batchCenterSourceCombo.currentText)
        if doCenter and centerSource == "Same loaded indices for all meshes":
            ids = self.logic.get_indices()
            if ids is None or len(ids) == 0:
                slicer.util.errorDisplay("Batch centering by indices requires a loaded centering indices file.")
                return
        if doCenter and centerSource == "Matched fiducials folder (.mrk.json)":
            fDir = self.batchFiducialsDirEdit.text.strip()
            if not fDir or not os.path.isdir(fDir):
                slicer.util.errorDisplay("Pick a valid fiducials folder for matched .mrk.json files.")
                return

        referenceFrame = None
        alignSource = str(self.alignSourceCombo.currentText)
        if doAlign:
            refNode = self.referenceModelSelector.currentNode()
            if refNode is None:
                slicer.util.errorDisplay("Batch alignment requires a reference PGD mesh in the alignment section.")
                return
            if alignSource == "Stored indices":
                alignIds = self.logic.get_alignment_indices()
                if alignIds is None or len(alignIds) == 0:
                    slicer.util.errorDisplay("Batch alignment by indices requires a loaded alignment indices file.")
                    return
            else:
                if self.referenceFidSelector.currentNode() is None:
                    slicer.util.errorDisplay("Batch alignment by fiducials requires reference PGD fiducials.")
                    return
                fDir = self.batchFiducialsDirEdit.text.strip()
                if not fDir or not os.path.isdir(fDir):
                    slicer.util.errorDisplay("Batch alignment by fiducials requires a valid matched fiducials folder.")
                    return

            try:
                _refPoly, referenceFrame = self._build_alignment_frame_for_model(
                    refNode,
                    fidNode=self.referenceFidSelector.currentNode(),
                )
            except Exception as e:
                slicer.util.errorDisplay(f"Could not build reference alignment frame: {e}")
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
                centerIndexRange=self.centerStoredIndexRangeEdit.text,
                fiducialsDir=self.batchFiducialsDirEdit.text.strip(),
                fiducialRange=self.fidRangeEdit.text,
                doAlign=doAlign,
                referenceFrame=referenceFrame,
                alignSource=alignSource,
                alignIndices=self.logic.get_alignment_indices(),
                alignLoadedIndexLabels=self.loadedIndexLabelsEdit.text,
                alignCanalLabels=self.alignCanalLabelsEdit.text,
                alignOtobasionLabels=self.alignOtobasionLabelsEdit.text,
                alignUpPairsText=self.alignUpPairsEdit.text,
                alignOtobasionWeight=float(self.otobasionNormalWeightSpin.value),
                alignCanalWeight=float(self.canalNormalWeightSpin.value),
                alignRotationCenterMode=str(self.alignRotationCenterCombo.currentText),
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
        self._align_indices = None

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

    def get_alignment_indices(self):
        return self._align_indices

    def load_alignment_indices(self, filepath, one_based=False):
        if not filepath or not os.path.isfile(filepath):
            raise ValueError("Invalid alignment indices file path.")
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
            raise ValueError("Unsupported alignment indices format. Use .npy or .npz.")

        ids = np.asarray(arr, dtype=np.int64).ravel()
        if one_based:
            ids = ids - 1
        self._align_indices = [int(v) for v in ids.tolist()]
        return self._align_indices

    def subset_indices_by_position(self, indices, rangeText="1-4"):
        """
        Select entries from a loaded index list by 1-based POSITION in the list.

        Example:
          indices = [v_for_fid_1, v_for_fid_2, v_for_fid_3, v_for_fid_4, ...]
          rangeText = "1-4" -> first four vertex IDs, normally the canal landmarks.

        This function does not interpret rangeText as mesh vertex IDs.
        """
        if indices is None or len(indices) == 0:
            raise ValueError("No indices provided.")

        if rangeText is None or not str(rangeText).strip():
            return [int(v) for v in indices]

        positions = self._parse_range_1based_keep_order(rangeText)
        if len(positions) == 0:
            raise ValueError("Stored-index anchor range is empty.")

        out = []
        n = len(indices)
        for pos in positions:
            idx = int(pos) - 1
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"Loaded index position {pos} is out of range. "
                    f"The loaded index list has {n} entries."
                )
            out.append(int(indices[idx]))

        return out

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
    def anchor_points_from_indices(self, polyData, indices):
        """
        Return the actual mesh points used for anchor computation.

        Returns:
          points: Nx3 numpy array in world/RAS coordinates
          valid_indices: list of valid vertex ids used
        """
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
        return pts, valid

    def anchor_center_from_indices(self, polyData, indices):
        points, _valid = self.anchor_points_from_indices(polyData, indices)
        return points.mean(axis=0)

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

    def anchor_points_from_fiducials(self, fidNode, rangeText="1-4"):
        """
        Return the actual fiducial points used for anchor computation.

        Returns:
          points: Nx3 numpy array in world/RAS coordinates
          valid_fiducial_ids: list of 0-based fiducial ids used
        """
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
        return pts, ids

    def anchor_center_from_fiducials(self, fidNode, rangeText="1-4"):
        points, _ids = self.anchor_points_from_fiducials(fidNode, rangeText)
        return points.mean(axis=0)

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
    # Anatomical orientation frame alignment
    # ------------------------------------------------------------
    def _normalize_vector(self, v, name="vector"):
        v = np.asarray(v, dtype=float).reshape(3)
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError(f"Cannot normalize zero-length {name}.")
        return v / n

    def _parse_label_pairs(self, text):
        if not text or not str(text).strip():
            raise ValueError("Up-vector pair specification cannot be empty.")
        pairs = []
        for chunk in str(text).replace(" ", "").split(","):
            if not chunk:
                continue
            if "-" not in chunk:
                raise ValueError(f"Invalid pair '{chunk}'. Expected format like 254-246.")
            a, b = chunk.split("-", 1)
            pairs.append((int(a), int(b)))
        if len(pairs) == 0:
            raise ValueError("No valid up-vector pairs were parsed.")
        return pairs

    def _points_from_poly_vertex_ids(self, polyData, vertexIds):
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise ValueError("Polydata is empty.")
        nPts = polyData.GetNumberOfPoints()
        valid = [int(i) for i in vertexIds if 0 <= int(i) < nPts]
        if len(valid) == 0:
            raise ValueError("No valid vertex IDs for this mesh.")
        pts = np.zeros((len(valid), 3), dtype=float)
        p = [0.0, 0.0, 0.0]
        for j, vid in enumerate(valid):
            polyData.GetPoint(int(vid), p)
            pts[j, :] = p
        return pts, valid

    def _stored_index_label_map(self, storedIndices, loadedIndexLabels):
        stored = [int(v) for v in storedIndices]
        labels = self._parse_range_1based_keep_order(loadedIndexLabels)
        if len(labels) != len(stored):
            raise ValueError(
                "Loaded-index label count does not match the number of loaded indices.\n"
                f"Labels parsed: {len(labels)}\n"
                f"Loaded indices: {len(stored)}\n\n"
                "If your .npy contains only canal + otobasion, use labels like: 1-4,246-261.\n"
                "If your .npy contains all fiducials, use: 1-261."
            )
        return {int(label): int(vertexId) for label, vertexId in zip(labels, stored)}

    def _points_from_stored_fiducial_labels(self, polyData, storedIndices, loadedIndexLabels, requestedLabels):
        labelMap = self._stored_index_label_map(storedIndices, loadedIndexLabels)
        labels = self._parse_range_1based_keep_order(requestedLabels)
        missing = [label for label in labels if int(label) not in labelMap]
        if missing:
            raise ValueError(f"Requested fiducial labels are not present in the loaded-index mapping: {missing}")
        vertexIds = [labelMap[int(label)] for label in labels]
        points, _validVertexIds = self._points_from_poly_vertex_ids(polyData, vertexIds)
        return points, labels, labelMap

    def _points_from_fiducial_labels(self, fidNode, requestedLabels):
        if fidNode is None or fidNode.GetNumberOfControlPoints() == 0:
            raise ValueError("Fiducial node is empty or None.")
        labels = self._parse_range_1based_keep_order(requestedLabels)
        n = fidNode.GetNumberOfControlPoints()
        validLabels = []
        pts = []
        p = [0.0, 0.0, 0.0]
        for label in labels:
            idx = int(label) - 1
            if idx < 0 or idx >= n:
                continue
            try:
                fidNode.GetNthControlPointPositionWorld(idx, p)
            except Exception:
                fidNode.GetNthControlPointPosition(idx, p)
            pts.append([float(p[0]), float(p[1]), float(p[2])])
            validLabels.append(int(label))
        if len(pts) == 0:
            raise ValueError("No valid fiducials selected by the requested labels.")
        return np.asarray(pts, dtype=float), validLabels

    def _plane_normal_from_points(self, points, name="points"):
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 3:
            raise ValueError(f"Need at least 3 points to compute a plane normal for {name}.")
        center = pts.mean(axis=0)
        centered = pts - center
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1, :]
        return self._normalize_vector(normal, f"{name} normal")

    def _compute_up_from_label_pairs(self, pointByLabel, upPairsText):
        pairs = self._parse_label_pairs(upPairsText)
        vectors = []
        for a, b in pairs:
            if int(a) not in pointByLabel:
                raise ValueError(f"Up-vector label {a} is missing.")
            if int(b) not in pointByLabel:
                raise ValueError(f"Up-vector label {b} is missing.")
            v = np.asarray(pointByLabel[int(b)], dtype=float) - np.asarray(pointByLabel[int(a)], dtype=float)
            vectors.append(self._normalize_vector(v, f"up pair {a}-{b}"))
        return self._normalize_vector(np.vstack(vectors).mean(axis=0), "mean up vector")

    def _build_anatomical_frame(self, canalPoints, canalLabels, otobasionPoints, otobasionLabels,
                                upPairsText, otobasionWeight=0.80, canalWeight=0.20):
        canalPoints = np.asarray(canalPoints, dtype=float)
        otobasionPoints = np.asarray(otobasionPoints, dtype=float)
        canalCenter = canalPoints.mean(axis=0)
        otobasionCenter = otobasionPoints.mean(axis=0)
        origin = 0.5 * (canalCenter + otobasionCenter)

        pointByLabel = {}
        for label, point in zip(canalLabels, canalPoints):
            pointByLabel[int(label)] = np.asarray(point, dtype=float)
        for label, point in zip(otobasionLabels, otobasionPoints):
            pointByLabel[int(label)] = np.asarray(point, dtype=float)

        up = self._compute_up_from_label_pairs(pointByLabel, upPairsText)
        canalNormal = self._plane_normal_from_points(canalPoints, "canal")
        otobasionNormal = self._plane_normal_from_points(otobasionPoints, "otobasion")
        sideDirection = self._normalize_vector(canalCenter - otobasionCenter, "otobasion-to-canal direction")

        if np.dot(canalNormal, sideDirection) < 0:
            canalNormal *= -1.0
        if np.dot(otobasionNormal, sideDirection) < 0:
            otobasionNormal *= -1.0

        normal = float(otobasionWeight) * otobasionNormal + float(canalWeight) * canalNormal
        normal = self._normalize_vector(normal, "combined anatomical normal")
        if np.dot(normal, sideDirection) < 0:
            normal *= -1.0

        y = self._normalize_vector(up, "anatomical up")
        z = normal - np.dot(normal, y) * y
        z = self._normalize_vector(z, "anatomical normal orthogonalized to up")
        x = self._normalize_vector(np.cross(y, z), "anatomical x")
        z = self._normalize_vector(np.cross(x, y), "anatomical z")
        frame = np.column_stack([x, y, z])
        if np.linalg.det(frame) < 0:
            x *= -1.0
            frame = np.column_stack([x, y, z])

        return {
            "origin": origin,
            "canal_center": canalCenter,
            "otobasion_center": otobasionCenter,
            "canal_points": canalPoints,
            "otobasion_points": otobasionPoints,
            "canal_labels": list(canalLabels),
            "otobasion_labels": list(otobasionLabels),
            "canal_normal": canalNormal,
            "otobasion_normal": otobasionNormal,
            "normal": z,
            "up": y,
            "x": x,
            "y": y,
            "z": z,
            "frame": frame,
        }

    def anatomical_frame_from_stored_indices(self, polyData, storedIndices, loadedIndexLabels,
                                             canalLabels, otobasionLabels, upPairsText,
                                             otobasionWeight=0.80, canalWeight=0.20):
        canalPoints, canalLabelList, _labelMap = self._points_from_stored_fiducial_labels(polyData, storedIndices, loadedIndexLabels, canalLabels)
        otobasionPoints, otobasionLabelList, _labelMap = self._points_from_stored_fiducial_labels(polyData, storedIndices, loadedIndexLabels, otobasionLabels)
        return self._build_anatomical_frame(canalPoints, canalLabelList, otobasionPoints, otobasionLabelList, upPairsText, otobasionWeight, canalWeight)

    def anatomical_frame_from_fiducials(self, fidNode, canalLabels, otobasionLabels, upPairsText,
                                        otobasionWeight=0.80, canalWeight=0.20):
        canalPoints, canalLabelList = self._points_from_fiducial_labels(fidNode, canalLabels)
        otobasionPoints, otobasionLabelList = self._points_from_fiducial_labels(fidNode, otobasionLabels)
        return self._build_anatomical_frame(canalPoints, canalLabelList, otobasionPoints, otobasionLabelList, upPairsText, otobasionWeight, canalWeight)

    def rotation_from_frame_to_frame(self, sourceFrame, referenceFrame):
        source = np.asarray(sourceFrame["frame"], dtype=float).reshape(3, 3)
        reference = np.asarray(referenceFrame["frame"], dtype=float).reshape(3, 3)
        rotation = reference @ source.T
        u, _s, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        return rotation

    def rotation_angle_degrees(self, rotation):
        r = np.asarray(rotation, dtype=float).reshape(3, 3)
        value = (np.trace(r) - 1.0) / 2.0
        value = max(-1.0, min(1.0, float(value)))
        return float(np.degrees(np.arccos(value)))

    def rotate_polydata_about_point(self, polyData, rotation, center):
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise ValueError("Input polydata is empty.")
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        center = np.asarray(center, dtype=float).reshape(3)
        out = vtk.vtkPolyData()
        out.DeepCopy(polyData)
        newPoints = vtk.vtkPoints()
        newPoints.SetNumberOfPoints(polyData.GetNumberOfPoints())
        p = np.zeros(3, dtype=float)
        tmp = [0.0, 0.0, 0.0]
        for i in range(polyData.GetNumberOfPoints()):
            polyData.GetPoint(i, tmp)
            p[:] = tmp
            q = center + rotation @ (p - center)
            newPoints.SetPoint(i, float(q[0]), float(q[1]), float(q[2]))
        out.SetPoints(newPoints)
        out.Modified()
        return out


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

    def _load_labeled_points_from_mrk_json(self, filePath, requestedLabels):
        """
        Load selected control points from a Slicer .mrk.json file using 1-based labels/ranges.
        Returns points in RAS/world coordinates and the valid 1-based labels used.
        """
        with open(filePath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "markups" not in data or len(data["markups"]) == 0:
            raise ValueError("No markups found in JSON.")
        mk = data["markups"][0]
        cps = mk.get("controlPoints", [])
        if len(cps) == 0:
            raise ValueError("No control points found in JSON.")

        coordSystem = str(mk.get("coordinateSystem", "RAS")).upper()
        labels = self._parse_range_1based_keep_order(requestedLabels)
        if len(labels) == 0:
            raise ValueError("Requested label range is empty.")

        pts = []
        validLabels = []
        n = len(cps)
        for label in labels:
            idx = int(label) - 1
            if idx < 0 or idx >= n:
                continue
            pos = cps[idx].get("position", None)
            if pos is None or len(pos) < 3:
                continue
            point = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=float)
            if coordSystem == "LPS":
                point[0] *= -1.0
                point[1] *= -1.0
            pts.append(point)
            validLabels.append(int(label))

        if len(pts) == 0:
            raise ValueError("No valid positions found for requested labels.")

        return np.vstack(pts), validLabels

    # ------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------
    def batch_process_mesh_folder(self, inputDir, outputDir, outputExt=".vtk",
                                  doMirror=False, flipX=True, flipY=False, flipZ=False, flipNormals=True,
                                  doCenter=True, targetCoord=None, centerSource="Same loaded indices for all meshes",
                                  indices=None, centerIndexRange="1-4", fiducialsDir="", fiducialRange="1-4",
                                  doAlign=False, referenceFrame=None, alignSource="Stored indices",
                                  alignIndices=None, alignLoadedIndexLabels="1-4,246-261",
                                  alignCanalLabels="1-4", alignOtobasionLabels="246-261",
                                  alignUpPairsText="254-246,255-261,253-247",
                                  alignOtobasionWeight=0.80, alignCanalWeight=0.20,
                                  alignRotationCenterMode="Target canal center"):
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

        if doAlign and referenceFrame is None:
            raise RuntimeError("Batch alignment requires a reference anatomical frame.")

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
                translation = np.zeros(3, dtype=float)

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
                        centerIds = self.subset_indices_by_position(indices, centerIndexRange)
                        anchor = self.anchor_center_from_indices(poly, centerIds)
                    else:
                        if not fiducialsDir or not os.path.isdir(fiducialsDir):
                            raise RuntimeError("Invalid fiducials folder.")
                        markupsPath = self._find_matching_markups(fiducialsDir, base)
                        if markupsPath is None:
                            skipped += 1
                            continue
                        anchor = self._load_anchor_center_from_mrk_json(markupsPath, fiducialRange)
                        if doMirror:
                            anchor = self.mirror_point(anchor, bool(flipX), bool(flipY), bool(flipZ))

                    poly, translation = self.center_polydata(poly, anchor, targetCoord)

                if doAlign:
                    if alignSource == "Stored indices":
                        if alignIndices is None or len(alignIndices) == 0:
                            raise RuntimeError("No alignment indices provided for batch alignment.")
                        targetFrame = self.anatomical_frame_from_stored_indices(
                            polyData=poly,
                            storedIndices=alignIndices,
                            loadedIndexLabels=alignLoadedIndexLabels,
                            canalLabels=alignCanalLabels,
                            otobasionLabels=alignOtobasionLabels,
                            upPairsText=alignUpPairsText,
                            otobasionWeight=float(alignOtobasionWeight),
                            canalWeight=float(alignCanalWeight),
                        )
                    else:
                        if not fiducialsDir or not os.path.isdir(fiducialsDir):
                            raise RuntimeError("Invalid fiducials folder for batch alignment.")
                        markupsPath = self._find_matching_markups(fiducialsDir, base)
                        if markupsPath is None:
                            skipped += 1
                            continue
                        canalPoints, canalLabelList = self._load_labeled_points_from_mrk_json(markupsPath, alignCanalLabels)
                        otobasionPoints, otobasionLabelList = self._load_labeled_points_from_mrk_json(markupsPath, alignOtobasionLabels)

                        if doMirror:
                            canalPoints = np.vstack([self.mirror_point(p, bool(flipX), bool(flipY), bool(flipZ)) for p in canalPoints])
                            otobasionPoints = np.vstack([self.mirror_point(p, bool(flipX), bool(flipY), bool(flipZ)) for p in otobasionPoints])
                        if doCenter:
                            canalPoints = canalPoints + translation.reshape(1, 3)
                            otobasionPoints = otobasionPoints + translation.reshape(1, 3)

                        targetFrame = self._build_anatomical_frame(
                            canalPoints=canalPoints,
                            canalLabels=canalLabelList,
                            otobasionPoints=otobasionPoints,
                            otobasionLabels=otobasionLabelList,
                            upPairsText=alignUpPairsText,
                            otobasionWeight=float(alignOtobasionWeight),
                            canalWeight=float(alignCanalWeight),
                        )

                    rotation = self.rotation_from_frame_to_frame(targetFrame, referenceFrame)
                    if alignRotationCenterMode == "Target world anchor":
                        rotationCenter = targetCoord
                    elif alignRotationCenterMode == "Target otobasion center":
                        rotationCenter = targetFrame["otobasion_center"]
                    elif alignRotationCenterMode == "Target canal center":
                        rotationCenter = targetFrame["canal_center"]
                    elif alignRotationCenterMode == "Target combined center":
                        rotationCenter = targetFrame["origin"]
                    else:
                        rotationCenter = targetFrame["canal_center"]

                    poly = self.rotate_polydata_about_point(poly, rotation, rotationCenter)

                suffix = []
                if doMirror:
                    suffix.append("mirrored")
                if doCenter:
                    suffix.append("centered")
                if doAlign:
                    suffix.append("aligned")
                suffixText = "_" + "_".join(suffix) if suffix else "_processed"
                outPath = os.path.join(outputDir, base + suffixText + outputExt)
                self._save_polydata(poly, outPath)
                saved += 1

            except Exception as e:
                print(f"[ETSE-UV Mesh Tools] Skipped {fname}: {e}")
                skipped += 1
            finally:
                processed += 1
                slicer.mrmlScene.RemoveNode(node)

        return processed, saved, skipped
