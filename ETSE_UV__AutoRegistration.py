# ETSE_UV__AutoRegistration.py
# 3D Slicer scripted module
#
# One-click automatic registration pipeline:
#   1) Rigid alignment (Procrustes / ICP / mesh_other).
#   2) Non-rigid NRICP (Sumner and/or Amberg) with default schedules
#      taken from ETSE_UV__SumnerAmbergRegistration.
#
# Supports single TARGET model and batch folder processing.

import os
import time

import numpy as np
import vtk
import vtkmodules.util.numpy_support as vtk_np
import slicer
import qt
import ctk

from slicer.ScriptedLoadableModule import *




from Resources.ETSE_UV__Dependencies import ensure_packages

ensure_packages(
    [("trimesh", "trimesh")],
    interactive=False,
    module_name="ETSE-UV Auto Registration",
)

import trimesh
# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class ETSE_UV__AutoRegistration(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Auto Registration"
        parent.categories = ["ETSE_UV"]
        parent.dependencies = []
        parent.contributors = ["ETSE-UV"]
        parent.helpText = """
        <p><b>Automatic registration pipeline</b> for registering a SOURCE template mesh to a TARGET mesh.</p>

        <p><b>Workflow:</b></p>
        <ol>
          <li>Select the SOURCE model.</li>
          <li>Select either a single TARGET model or a folder of TARGET meshes for batch mode.</li>
          <li>Optionally select corresponding SOURCE/TARGET fiducials.</li>
          <li>Click <b>Run automatic registration</b>.</li>
        </ol>

        <p><b>Pipeline:</b></p>
        <ul>
          <li>Rigid alignment: Procrustes, ICP, mesh_other, or none.</li>
          <li>Non-rigid deformation: Trimesh NRICP using Sumner, Amberg, both, or none.</li>
        </ul>

        <p>Advanced options control the rigid method, landmark usage, NRICP distance threshold,
        and non-rigid method selection.</p>
        """
        parent.acknowledgementText = (
            "Developed by Juan Antonio De Rus Arance at the Escola Tècnica Superior "
            "d'Enginyeria (ETSE-UV), Universitat de València, in the context of the "
            "Signal Processing & Acoustic Technology (SPAT) research group. "
            "Thanks to the 3D Slicer, SlicerMorph, VTK, NumPy, SciPy, Trimesh, and related "
            "open-source communities."
        )


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class ETSE_UV__AutoRegistrationWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()

        self.logic = ETSE_UV__AutoRegistrationLogic()

        form = qt.QFormLayout()
        self.layout.addLayout(form)

        # ---------------- Common inputs ----------------
        # Template model
        self.sourceSelector = slicer.qMRMLNodeComboBox()
        self.sourceSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.sourceSelector.selectNodeUponCreation = True
        self.sourceSelector.addEnabled = False
        self.sourceSelector.removeEnabled = False
        self.sourceSelector.noneEnabled = False
        self.sourceSelector.setMRMLScene(slicer.mrmlScene)
        self.sourceSelector.setToolTip("Template mesh that will be deformed (SOURCE).")
        form.addRow("Source model:", self.sourceSelector)

        # Template fiducials
        self.sourceFiducialSelector = slicer.qMRMLNodeComboBox()
        self.sourceFiducialSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.sourceFiducialSelector.selectNodeUponCreation = True
        self.sourceFiducialSelector.addEnabled = False
        self.sourceFiducialSelector.removeEnabled = False
        self.sourceFiducialSelector.noneEnabled = True
        self.sourceFiducialSelector.setMRMLScene(slicer.mrmlScene)
        self.sourceFiducialSelector.setToolTip(
            "Fiducial set on the SOURCE model. Required for Procrustes; "
            "used as landmarks for NRICP if 'Use landmarks' is enabled."
        )
        form.addRow("Source fiducials:", self.sourceFiducialSelector)

        # Target model (single)
        self.targetSelector = slicer.qMRMLNodeComboBox()
        self.targetSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.targetSelector.selectNodeUponCreation = True
        self.targetSelector.addEnabled = False
        self.targetSelector.removeEnabled = False
        self.targetSelector.noneEnabled = True
        self.targetSelector.setMRMLScene(slicer.mrmlScene)
        self.targetSelector.setToolTip("Single TARGET mesh. The SOURCE will be registered to this.")
        form.addRow("Target model:", self.targetSelector)

        # Target fiducials (single)
        self.targetFiducialSelector = slicer.qMRMLNodeComboBox()
        self.targetFiducialSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.targetFiducialSelector.selectNodeUponCreation = True
        self.targetFiducialSelector.addEnabled = False
        self.targetFiducialSelector.removeEnabled = False
        self.targetFiducialSelector.noneEnabled = True
        self.targetFiducialSelector.setMRMLScene(slicer.mrmlScene)
        self.targetFiducialSelector.setToolTip(
            "Fiducial set on the TARGET model. Required for Procrustes rigid alignment; "
            "for NRICP it is used as landmarks if enabled."
        )
        form.addRow("Target fiducials:", self.targetFiducialSelector)

        # Output names for single run
        self.registeredModelNameEdit = qt.QLineEdit()
        self.registeredModelNameEdit.placeholderText = "Use target model name"
        self.registeredModelNameEdit.setToolTip(
            "Name for the final registered (non-rigid) model node.\n"
            "If empty, the TARGET model name will be used."
        )
        form.addRow("Registered model name:", self.registeredModelNameEdit)

        self.projectedFiducialsNameEdit = qt.QLineEdit()
        self.projectedFiducialsNameEdit.placeholderText = "Use source fiducials name or derived"
        self.projectedFiducialsNameEdit.setToolTip(
            "Name for projected fiducials on the final registered model.\n"
            "If empty, the SOURCE fiducials name will be used, or '<target>_Fiducials'."
        )
        form.addRow("Projected fiducials name:", self.projectedFiducialsNameEdit)

        # ---------------- Rigid / non-rigid options ----------------
        optionsBox = ctk.ctkCollapsibleButton()
        optionsBox.text = "Advanced registration options"
        self.layout.addWidget(optionsBox)
        optForm = qt.QFormLayout(optionsBox)

        # Rigid method
        self.rigidMethodCombo = qt.QComboBox()
        self.rigidMethodCombo.addItem("Procrustes (landmark-based, default)", "procrustes")
        self.rigidMethodCombo.addItem("Rigid ICP (icp)", "icp")
        self.rigidMethodCombo.addItem("Rigid ICP with PCA init (mesh_other)", "mesh_other")
        self.rigidMethodCombo.addItem("None (skip rigid, non-rigid only)", "none")
        self.rigidMethodCombo.setCurrentIndex(0)
        self.rigidMethodCombo.setToolTip(
            "Rigid alignment method applied BEFORE the non-rigid NRICP stage.\n"
            "Procrustes requires SOURCE and TARGET fiducials with identical ordering."
        )
        optForm.addRow("Rigid method:", self.rigidMethodCombo)

        # Non-rigid method
        self.nonRigidMethodCombo = qt.QComboBox()
        self.nonRigidMethodCombo.addItem("Sumner + Amberg (default)", "sumner+amberg")
        self.nonRigidMethodCombo.addItem("Sumner only", "sumner")
        self.nonRigidMethodCombo.addItem("Amberg only", "amberg")
        self.nonRigidMethodCombo.addItem("None (rigid only)", "none")
        self.nonRigidMethodCombo.setCurrentIndex(0)
        self.nonRigidMethodCombo.setToolTip(
            "Non-rigid NRICP variant(s) to use AFTER the rigid stage.\n"
            "Default: Sumner followed by Amberg, with the schedules tuned in the dedicated module."
        )
        optForm.addRow("Non-rigid method:", self.nonRigidMethodCombo)

        # Use landmarks
        self.useLandmarksCheckBox = qt.QCheckBox("Use fiducials as landmarks in NRICP")
        self.useLandmarksCheckBox.checked = True
        self.useLandmarksCheckBox.toolTip = (
            "If enabled, SOURCE and TARGET fiducials are used as NRICP landmarks.\n"
            "If disabled, NRICP will run purely geometry-based."
        )
        optForm.addRow(self.useLandmarksCheckBox)

        # Distance threshold (NRICP)
        self.distanceThresholdSpinBox = ctk.ctkDoubleSpinBox()
        self.distanceThresholdSpinBox.decimals = 3
        self.distanceThresholdSpinBox.singleStep = 0.1
        self.distanceThresholdSpinBox.minimum = 0.0
        self.distanceThresholdSpinBox.maximum = 1e6
        self.distanceThresholdSpinBox.value = 0.05  # good default from Sumner+Amberg module
        self.distanceThresholdSpinBox.setToolTip(
            "Maximum correspondence distance for NRICP (in model units).\n"
            "Larger values allow looser matches but may increase outliers."
        )
        optForm.addRow("NRICP distance threshold:", self.distanceThresholdSpinBox)

        # Copy display
        self.copyDisplayCheckBox = qt.QCheckBox("Copy display properties from SOURCE to output")
        self.copyDisplayCheckBox.checked = True
        optForm.addRow(self.copyDisplayCheckBox)

        # ---------------- Single-run button ----------------
        self.applyButton = qt.QPushButton("Run automatic registration")
        self.applyButton.toolTip = (
            "Run the full pipeline (rigid + non-rigid) on the selected SOURCE and TARGET models."
        )
        self.layout.addWidget(self.applyButton)
        self.applyButton.clicked.connect(self.onApplySingle)

        # ---------------- Batch group ----------------
        batchBox = ctk.ctkCollapsibleButton()
        batchBox.text = "Batch automatic registration"
        self.layout.addWidget(batchBox)
        batchForm = qt.QFormLayout(batchBox)

        self.targetModelDirectoryButton = ctk.ctkDirectoryButton()
        self.targetModelDirectoryButton.setToolTip(
            "Folder containing TARGET meshes (.ply, .obj, .vtk) for batch processing."
        )
        self.targetModelDirectoryButton.setMaximumWidth(400)
        batchForm.addRow("Target models folder:", self.targetModelDirectoryButton)

        self.targetMarkupDirectoryButton = ctk.ctkDirectoryButton()
        self.targetMarkupDirectoryButton.setToolTip(
            "Folder containing TARGET fiducials (.mrk.json). Required for Procrustes; "
            "used as landmarks in NRICP if available."
        )
        self.targetMarkupDirectoryButton.setMaximumWidth(400)
        batchForm.addRow("Target fiducials folder:", self.targetMarkupDirectoryButton)

        self.outputDirectoryButton = ctk.ctkDirectoryButton()
        self.outputDirectoryButton.setToolTip(
            "Folder to save final registered models and projected fiducials.\n"
            "Files will be named after each TARGET model."
        )
        self.outputDirectoryButton.setMaximumWidth(400)
        batchForm.addRow("Output folder:", self.outputDirectoryButton)

        self.batchButton = qt.QPushButton("Run batch automatic registration")
        self.batchButton.toolTip = (
            "For each TARGET mesh (and fiducials, if required), run rigid + non-rigid\n"
            "registration of the SOURCE template and save the result."
        )
        batchForm.addRow(self.batchButton)
        self.batchButton.clicked.connect(self.onApplyBatch)

        self.layout.addStretch(1)

    # ------------------------------------------------------------------
    # Single run handler
    # ------------------------------------------------------------------
    def onApplySingle(self):
        sourceModel = self.sourceSelector.currentNode()
        targetModel = self.targetSelector.currentNode()
        sourceFiducial = self.sourceFiducialSelector.currentNode()
        targetFiducial = self.targetFiducialSelector.currentNode()

        if not sourceModel or not targetModel:
            slicer.util.errorDisplay("Please select both SOURCE and TARGET model nodes.")
            return

        rigidMethod = self.rigidMethodCombo.currentData
        nonRigidMethod = self.nonRigidMethodCombo.currentData
        useLandmarks = self.useLandmarksCheckBox.checked
        distance_threshold = self.distanceThresholdSpinBox.value

        # Procrustes rigid stage requires fiducials
        if rigidMethod == "procrustes":
            if not sourceFiducial or not targetFiducial:
                slicer.util.errorDisplay(
                    "Procrustes rigid alignment requires SOURCE and TARGET fiducials "
                    "with corresponding points."
                )
                return

        # Landmarks for NRICP must be paired
        if useLandmarks and nonRigidMethod != "none":
            if (sourceFiducial is None) or (targetFiducial is None):
                slicer.util.warningDisplay(
                    "Landmarks are enabled for NRICP but fiducials are missing.\n"
                    "The non-rigid stage will run WITHOUT landmarks."
                )
                useLandmarks = False

        registeredModelName = self.registeredModelNameEdit.text.strip()
        if not registeredModelName:
            registeredModelName = targetModel.GetName()

        projectedFiducialsName = self.projectedFiducialsNameEdit.text.strip()
        if not projectedFiducialsName:
            if sourceFiducial:
                projectedFiducialsName = sourceFiducial.GetName()
            else:
                projectedFiducialsName = registeredModelName + "_Fiducials"

        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            finalModelNode, finalFiducialsNode, metrics = self.logic.runSingleAuto(
                sourceModel=sourceModel,
                targetModel=targetModel,
                sourceFiducial=sourceFiducial,
                targetFiducial=targetFiducial,
                rigidMethod=rigidMethod,
                nonRigidMethod=nonRigidMethod,
                distance_threshold=distance_threshold,
                registeredModelName=registeredModelName,
                projectedFiducialsName=projectedFiducialsName,
                useLandmarks=useLandmarks,
                keepIntermediates=True,
            )
        except Exception as e:
            slicer.app.restoreOverrideCursor()
            slicer.util.errorDisplay(f"Automatic registration failed:\n{e}")
            raise
        finally:
            slicer.app.restoreOverrideCursor()

        if finalModelNode and self.copyDisplayCheckBox.checked and sourceModel.GetDisplayNode():
            srcDisp = sourceModel.GetDisplayNode()
            dstDisp = finalModelNode.GetDisplayNode()
            if dstDisp:
                dstDisp.Copy(srcDisp)

        if finalModelNode:
            print("[AUTO_REG] Automatic registration finished.")
            print(f"[AUTO_REG] Rigid method: {metrics.get('rigid_method')}")
            print(f"[AUTO_REG] Non-rigid method: {metrics.get('nonrigid_method')}")
            print(f"[AUTO_REG] Total time: {metrics.get('total_time')}")
            print(f"[AUTO_REG] Final mean distance: {metrics.get('final_mean_distance')}")

    # ------------------------------------------------------------------
    # Batch handler
    # ------------------------------------------------------------------
    def onApplyBatch(self):
        sourceModel = self.sourceSelector.currentNode()
        sourceFiducial = self.sourceFiducialSelector.currentNode()

        if not sourceModel:
            slicer.util.errorDisplay("Please select a SOURCE model (template) for batch registration.")
            return

        modelDirectory = self.targetModelDirectoryButton.directory
        markupDirectory = self.targetMarkupDirectoryButton.directory
        outputDirectory = self.outputDirectoryButton.directory

        rigidMethod = self.rigidMethodCombo.currentData
        nonRigidMethod = self.nonRigidMethodCombo.currentData
        useLandmarks = self.useLandmarksCheckBox.checked
        distance_threshold = self.distanceThresholdSpinBox.value

        if not modelDirectory or not os.path.isdir(modelDirectory):
            slicer.util.errorDisplay("Please select a valid TARGET models folder.")
            return

        # Procrustes needs target markups in batch
        if rigidMethod == "procrustes":
            if not markupDirectory or not os.path.isdir(markupDirectory):
                slicer.util.errorDisplay(
                    "Procrustes rigid alignment requires a valid TARGET fiducials folder (.mrk.json)."
                )
                return

        if not outputDirectory:
            slicer.util.errorDisplay("Please select an output folder.")
            return
        if not os.path.isdir(outputDirectory):
            os.makedirs(outputDirectory, exist_ok=True)

        print("[AUTO_REG][Batch] Starting batch automatic registration.")
        print(f"[AUTO_REG][Batch] Rigid method: {rigidMethod}")
        print(f"[AUTO_REG][Batch] Non-rigid method: {nonRigidMethod}")
        print(f"[AUTO_REG][Batch] Template: {sourceModel.GetName()}")
        print(f"[AUTO_REG][Batch] Models folder:  {modelDirectory}")
        print(f"[AUTO_REG][Batch] Markups folder: {markupDirectory if markupDirectory else '(none)'}")
        print(f"[AUTO_REG][Batch] Output folder:  {outputDirectory}")

        total = 0
        success = 0
        failures = []

        exts = (".ply", ".obj", ".vtk")

        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)

            for modelFileName in os.listdir(modelDirectory):
                if not modelFileName.lower().endswith(exts):
                    continue

                total += 1
                modelFilePath = os.path.join(modelDirectory, modelFileName)
                baseName, _ = os.path.splitext(modelFileName)

                registeredModelName = baseName
                projectedFiducialsName = baseName + "_Fiducials"

                targetFiducialNode = None
                markupFilePath = None
                if markupDirectory:
                    markupFileName = baseName + ".mrk.json"
                    markupFilePath = os.path.join(markupDirectory, markupFileName)

                # For rigid Procrustes we strictly require target fiducials
                if rigidMethod == "procrustes":
                    if not markupFilePath or not os.path.exists(markupFilePath):
                        msg = f"Missing TARGET fiducials for {modelFileName} (required by Procrustes)."
                        print(f"[AUTO_REG][Batch][Skip] {msg}")
                        failures.append((modelFileName, msg))
                        continue

                print(f"[AUTO_REG][Batch] Processing {modelFileName} ...")
                slicer.app.processEvents()

                try:
                    # Load target model
                    targetModelNode = slicer.util.loadModel(modelFilePath)
                    if not targetModelNode:
                        msg = "Failed to load target model."
                        print(f"[AUTO_REG][Batch][Error] {msg}")
                        failures.append((modelFileName, msg))
                        continue

                    # Load target fiducials if present
                    if markupFilePath and os.path.exists(markupFilePath):
                        targetFiducialNode = slicer.util.loadMarkups(markupFilePath)
                        if not targetFiducialNode:
                            msg = "Failed to load target fiducials."
                            print(f"[AUTO_REG][Batch][Error] {msg}")
                            failures.append((modelFileName, msg))
                            slicer.mrmlScene.RemoveNode(targetModelNode)
                            continue
                    else:
                        targetFiducialNode = None

                    # In batch, if landmarks are requested but target fiducials are missing,
                    # fall back to landmark-free NRICP (except for Procrustes, handled above).
                    useLandmarksThis = useLandmarks
                    if useLandmarks and nonRigidMethod != "none":
                        if (sourceFiducial is None) or (targetFiducialNode is None):
                            print(
                                "[AUTO_REG][Batch] Landmarks requested but fiducials missing; "
                                "running NRICP without landmarks for this target."
                            )
                            useLandmarksThis = False

                    finalModelNode, finalFiducialsNode, metrics = self.logic.runSingleAuto(
                        sourceModel=sourceModel,
                        targetModel=targetModelNode,
                        sourceFiducial=sourceFiducial,
                        targetFiducial=targetFiducialNode,
                        rigidMethod=rigidMethod,
                        nonRigidMethod=nonRigidMethod,
                        distance_threshold=distance_threshold,
                        registeredModelName=registeredModelName,
                        projectedFiducialsName=projectedFiducialsName,
                        useLandmarks=useLandmarksThis,
                        keepIntermediates=False,
                    )

                    if finalModelNode is None:
                        msg = "Logic returned no final model."
                        print(f"[AUTO_REG][Batch][Error] {msg}")
                        failures.append((modelFileName, msg))
                        if targetModelNode:
                            slicer.mrmlScene.RemoveNode(targetModelNode)
                        if targetFiducialNode:
                            slicer.mrmlScene.RemoveNode(targetFiducialNode)
                        continue

                    # Save outputs
                    outModelPath = os.path.join(outputDirectory, registeredModelName + ".vtk")
                    if not slicer.util.saveNode(finalModelNode, outModelPath):
                        msg = "Could not save registered model."
                        print(f"[AUTO_REG][Batch][Error] {msg}")
                        failures.append((modelFileName, msg))
                    else:
                        success += 1

                    if finalFiducialsNode:
                        outMarkupPath = os.path.join(outputDirectory, projectedFiducialsName + ".mrk.json")
                        if not slicer.util.saveNode(finalFiducialsNode, outMarkupPath):
                            print("[AUTO_REG][Batch][Warning] Could not save projected fiducials.")

                    # Clean up scene for this target
                    if finalModelNode:
                        slicer.mrmlScene.RemoveNode(finalModelNode)
                    if finalFiducialsNode:
                        slicer.mrmlScene.RemoveNode(finalFiducialsNode)
                    if targetModelNode:
                        slicer.mrmlScene.RemoveNode(targetModelNode)
                    if targetFiducialNode:
                        slicer.mrmlScene.RemoveNode(targetFiducialNode)

                except Exception as e:
                    msg = f"Exception during processing: {e}"
                    print(f"[AUTO_REG][Batch][Error] {msg}")
                    failures.append((modelFileName, msg))
                    # Try to clean up any half-created nodes
                    for nodeName in [registeredModelName, projectedFiducialsName,
                                     registeredModelName + "_rigid", projectedFiducialsName + "_rigid",
                                     registeredModelName + "_sumner", projectedFiducialsName + "_sumner"]:
                        try:
                            n = slicer.util.getNode(nodeName)
                            slicer.mrmlScene.RemoveNode(n)
                        except Exception:
                            pass

            total_time_all = "n/a"  # could compute if we wanted
        finally:
            slicer.app.restoreOverrideCursor()

        summary = (
            "Batch automatic registration finished.\n\n"
            f"Total target models found: {total}\n"
            f"Successfully processed:    {success}\n"
            f"Failed or skipped:         {len(failures)}"
        )
        slicer.util.infoDisplay(summary, "ETSE_UV__AutoRegistration batch")

        if failures:
            print("[AUTO_REG][Batch] Failures:")
            for fname, reason in failures:
                print(f"  - {fname}: {reason}")


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class ETSE_UV__AutoRegistrationLogic(ScriptedLoadableModuleLogic):

    # Default Sumner & Amberg schedules copied from
    # ETSE_UV__SumnerAmbergRegistration.
    DEFAULT_SUMNER_STEPS = [
        # PHASE 1 — landmarks-only regularization (wC = 0)
        [0.0,    0.001,  1.0, 120.0, 0.80],
        [0.0,    0.001,  1.0, 110.0, 0.80],
        [0.0,    0.001,  1.0, 100.0, 0.80],
        [0.0,    0.001,  1.0,  90.0, 0.80],
        [0.0,    0.001,  1.0,  80.0, 0.80],
        [0.0,    0.001,  1.0,  70.0, 0.80],
        [0.0,    0.001,  1.0,  60.0, 0.80],
        [0.0,    0.001,  1.0,  50.0, 0.80],
        [0.0,    0.001,  1.0,  40.0, 0.80],
        [0.0,    0.001,  1.0,  30.0, 0.80],

        # PHASE 2 — correspondences progressively activated (wC → 5000)
        # Section A: tiny wC
        [1.0,    0.001,  1.0,  30.0, 0.85],
        [2.0,    0.001,  1.0,  28.0, 0.85],
        [5.0,    0.001,  1.0,  26.0, 0.85],
        [10.0,   0.001,  1.0,  24.0, 0.85],
        [20.0,   0.001,  1.0,  22.0, 0.85],

        # Section B: low wC
        [40.0,   0.001,  1.0,  20.0, 0.90],
        [75.0,   0.001,  1.0,  18.0, 0.92],
        [100.0,  0.001,  1.0,  16.0, 0.94],
        [150.0,  0.001,  1.0,  14.0, 0.96],

        # Section C: medium wC
        [200.0,  0.001,  1.0,  13.0, 0.98],
        [300.0,  0.001,  1.0,  12.0, 1.00],
        [400.0,  0.001,  1.0,  11.0, 1.00],
        [500.0,  0.001,  1.0,  10.0, 1.00],

        # Section D: strong wC
        [600.0,  0.001,  1.0,   9.5, 1.00],
        [750.0,  0.001,  1.0,   9.0, 1.00],
        [900.0,  0.001,  1.0,   8.5, 1.00],
        [1000.0, 0.001,  1.0,   8.0, 1.00],

        # Section E: very strong wC
        [1200.0, 0.001,  1.0,   7.5, 1.00],
        [1500.0, 0.001,  1.0,   7.0, 1.00],
        [1800.0, 0.001,  1.0,   6.5, 1.00],
        [2200.0, 0.001,  1.0,   6.0, 1.00],
        [2500.0, 0.001,  1.0,   5.5, 1.00],

        # Section F: final aggressive fitting
        [3000.0, 0.001,  1.0,   5.0, 1.00],
        [3500.0, 0.001,  1.0,   4.8, 1.00],
        [4000.0, 0.001,  1.0,   4.6, 1.00],
        [4500.0, 0.001,  1.0,   4.4, 1.00],
        [5000.0, 0.001,  1.0,   4.2, 1.00],
        [5000.0, 0.001,  1.0,   4.0, 1.00],
        [5000.0, 0.001,  1.0,   3.8, 1.00],
        [5000.0, 0.001,  1.0,   3.6, 1.00],
        [5000.0, 0.001,  1.0,   3.4, 1.00],
        [5000.0, 0.001,  1.0,   3.2, 1.00],
    ]

    DEFAULT_AMBERG_STEPS = [
        # [ws,     wl,   wn,   max_iter]
        [0.20, 20.0, 0.90, 16],
        [0.18, 18.0, 0.92, 16],
        [0.16, 16.0, 0.93, 16],
        [0.14, 14.0, 0.94, 18],
        [0.12, 12.0, 0.95, 18],
        [0.10, 10.0, 0.96, 20],
        [0.09,  8.0, 0.97, 20],
        [0.08,  6.0, 0.97, 22],
        [0.07,  5.0, 0.98, 22],
        [0.06,  4.0, 0.98, 24],
        [0.05,  3.0, 0.99, 24],
        [0.045, 2.5, 0.99, 26],
        [0.040, 2.0, 1.00, 26],
        [0.035, 1.5, 1.00, 28],
        [0.030, 1.2, 1.00, 30],
        [0.028, 1.0, 1.00, 30],
        [0.026, 0.8, 1.00, 32],
        [0.024, 0.6, 1.00, 32],
        [0.022, 0.5, 1.00, 34],
        [0.020, 0.4, 1.00, 34],
    ]

    

    def runSingleAuto(
        self,
        sourceModel,
        targetModel,
        sourceFiducial,
        targetFiducial,
        rigidMethod,
        nonRigidMethod,
        distance_threshold,
        registeredModelName,
        projectedFiducialsName,
        useLandmarks=True,
        keepIntermediates=False,
    ):
        """
        Full pipeline on MRML nodes.

        Returns (finalModelNode, finalFiducialsNode, metrics_dict).
        """
       

        if not sourceModel or not targetModel:
            raise RuntimeError("Source and Target models are required.")

        # Lazy-import helper modules here so that 'trimesh' is already available
        from ETSE_UV__TrimeshRegistration import ETSE_UV__TrimeshRegistrationLogic
        from ETSE_UV__SumnerAmbergRegistration import ETSE_UV__SumnerAmbergRegistrationLogic

        total_start = time.time()

        rigidLogic = ETSE_UV__TrimeshRegistrationLogic()
        nricpLogic = ETSE_UV__SumnerAmbergRegistrationLogic()

        # ---------- Stage 1: Rigid alignment ----------
        rigidModelNode = sourceModel
        rigidSourceFiducialNode = sourceFiducial
        rigidMetrics = None

        if rigidMethod and rigidMethod != "none":
            method_name_map = {
                "procrustes": "Procrustes (landmark-based)",
                "icp": "Rigid ICP",
                "mesh_other": "Rigid ICP with PCA init",
            }
            method_name = method_name_map.get(rigidMethod, rigidMethod)

            rigid_model_name = registeredModelName + "_rigid"
            rigid_fids_name = projectedFiducialsName + "_rigid"

            rigidModelNode, rigidMetrics = rigidLogic.run(
                method=rigidMethod,
                method_name=method_name,
                sourceModel=sourceModel,
                targetModel=targetModel,
                sourceFiducial=sourceFiducial,
                targetFiducial=targetFiducial,
                registeredModelName=rigid_model_name,
                projectedFiducialsName=rigid_fids_name,
            )

            # If projected fiducials were created, use them as SOURCE landmarks
            if sourceFiducial is not None:
                try:
                    rigidSourceFiducialNode = slicer.util.getNode(rigid_fids_name)
                except Exception:
                    rigidSourceFiducialNode = sourceFiducial

            # Hide rigid intermediate by default
            if rigidModelNode and rigidModelNode.GetDisplayNode():
                rigidModelNode.GetDisplayNode().SetVisibility(False)
            if rigidSourceFiducialNode and rigidSourceFiducialNode.GetDisplayNode():
                rigidSourceFiducialNode.GetDisplayNode().SetVisibility(False)

        # ---------- Stage 2: Non-rigid NRICP ----------
        finalModelNode = rigidModelNode
        finalFiducialsNode = rigidSourceFiducialNode
        nricpMetrics = None

        if nonRigidMethod and nonRigidMethod != "none":
            useSumner = nonRigidMethod in ("sumner", "sumner+amberg")
            useAmberg = nonRigidMethod in ("amberg", "sumner+amberg")

            stepsSumner = self.DEFAULT_SUMNER_STEPS if useSumner else None
            stepsAmberg = self.DEFAULT_AMBERG_STEPS if useAmberg else None

            finalModelNode, nricpMetrics = nricpLogic.run(
                sourceModel=rigidModelNode,
                targetModel=targetModel,
                sourceFiducial=rigidSourceFiducialNode if useLandmarks else None,
                targetFiducial=targetFiducial if useLandmarks else None,
                useSumner=useSumner,
                stepsSumner=stepsSumner,
                useAmberg=useAmberg,
                stepsAmberg=stepsAmberg,
                distance_threshold=distance_threshold,
                registeredModelName=registeredModelName,
                projectedFiducialsName=projectedFiducialsName,
                useLandmarks=useLandmarks,
            )

            # NRICP logic creates projected fiducials internally if landmarks are used
            finalFiducialsNode = None
            if useLandmarks and rigidSourceFiducialNode is not None:
                try:
                    finalFiducialsNode = slicer.util.getNode(projectedFiducialsName)
                except Exception:
                    finalFiducialsNode = None

        # Cleanup intermediate nodes if requested
        if not keepIntermediates:
            # Rigid
            if rigidMethod and rigidMethod != "none":
                try:
                    rigid_model_name = registeredModelName + "_rigid"
                    node = slicer.util.getNode(rigid_model_name)
                    if node != finalModelNode:
                        slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass
                try:
                    rigid_fids_name = projectedFiducialsName + "_rigid"
                    node = slicer.util.getNode(rigid_fids_name)
                    if node and node != finalFiducialsNode:
                        slicer.mrmlScene.RemoveNode(node)
                except Exception:
                    pass

            # Sumner intermediate (if Amberg was used)
            if nonRigidMethod in ("sumner+amberg", "amberg"):
                for name in (registeredModelName + "_sumner", projectedFiducialsName + "_sumner"):
                    try:
                        node = slicer.util.getNode(name)
                        if node and node not in (finalModelNode, finalFiducialsNode):
                            slicer.mrmlScene.RemoveNode(node)
                    except Exception:
                        pass

        total_time = time.time() - total_start

        # Compose metrics summary
        metrics = {
            "rigid_method": rigidMethod,
            "nonrigid_method": nonRigidMethod,
            "rigid_metrics": rigidMetrics,
            "nricp_metrics": nricpMetrics,
            "total_time": total_time,
            "final_mean_distance": None,
        }
        if nricpMetrics and "final_mean_distance" in nricpMetrics:
            metrics["final_mean_distance"] = nricpMetrics["final_mean_distance"]

        return finalModelNode, finalFiducialsNode, metrics
