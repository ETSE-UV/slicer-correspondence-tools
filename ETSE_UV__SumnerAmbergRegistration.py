# ETSE_UV__SumnerAmbergRegistration.py
# ETSE_UV__SumnerAmbergRegistration.py
# 3D Slicer scripted module
#
# Non-rigid registration using Trimesh NRICP (Sumner & Amberg variants).
# Requires source/target meshes + corresponding fiducial sets.

import time
import os
import csv

import numpy as np
from Resources.ETSE_UV__Dependencies import ensure_packages

ensure_packages(
    [("trimesh", "trimesh")],
    interactive=False,
    module_name="ETSE-UV Sumner-Amberg Registration",
)

import trimesh
import vtk
import vtkmodules.util.numpy_support as vtk_np
import slicer
import qt
import ctk



from slicer.ScriptedLoadableModule import *


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class ETSE_UV__SumnerAmbergRegistration(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Sumner-Amberg Registration"
        parent.categories = ["ETSE_UV"]
        parent.dependencies = []
        parent.contributors = ["ETSE-UV"]
        parent.helpText = """
        <p>Non-rigidly register a SOURCE mesh to a TARGET mesh using Trimesh NRICP
        with the Sumner and/or Amberg variants.</p>

        <p><b>Inputs:</b></p>
        <ul>
          <li>SOURCE model: the template mesh that will be deformed.</li>
          <li>TARGET model: the mesh to match.</li>
          <li>Optional SOURCE/TARGET fiducials: used as landmarks when landmark usage is enabled.</li>
        </ul>

        <p><b>Outputs:</b></p>
        <ul>
          <li>A registered model.</li>
          <li>Projected fiducials on the registered model, when applicable.</li>
          <li>Batch CSV logs when running folder processing.</li>
        </ul>

        <p>The Sumner and Amberg optimization schedules can be edited from the parameter tables.</p>
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
class ETSE_UV__SumnerAmbergRegistrationWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()

        self.logic = ETSE_UV__SumnerAmbergRegistrationLogic()

        # Main form layout (single registration)
        form = qt.QFormLayout()
        self.layout.addLayout(form)

        # -------------------- Source model --------------------
        self.sourceSelector = slicer.qMRMLNodeComboBox()
        self.sourceSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.sourceSelector.selectNodeUponCreation = True
        self.sourceSelector.addEnabled = False
        self.sourceSelector.removeEnabled = False
        self.sourceSelector.noneEnabled = False
        self.sourceSelector.setMRMLScene(slicer.mrmlScene)
        self.sourceSelector.setToolTip(
            "Template mesh that will be deformed (SOURCE)."
        )
        form.addRow("Source model:", self.sourceSelector)

        # -------------------- Source fiducials --------------------
        self.sourceFiducialSelector = slicer.qMRMLNodeComboBox()
        self.sourceFiducialSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.sourceFiducialSelector.selectNodeUponCreation = True
        self.sourceFiducialSelector.addEnabled = False
        self.sourceFiducialSelector.removeEnabled = False
        self.sourceFiducialSelector.noneEnabled = False
        self.sourceFiducialSelector.setMRMLScene(slicer.mrmlScene)
        self.sourceFiducialSelector.setToolTip(
            "Fiducial set on the SOURCE model. Ordering must correspond to TARGET fiducials."
        )
        form.addRow("Source fiducials:", self.sourceFiducialSelector)

        # -------------------- Target model --------------------
        self.targetSelector = slicer.qMRMLNodeComboBox()
        self.targetSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.targetSelector.selectNodeUponCreation = True
        self.targetSelector.addEnabled = False
        self.targetSelector.removeEnabled = False
        self.targetSelector.noneEnabled = False
        self.targetSelector.setMRMLScene(slicer.mrmlScene)
        self.targetSelector.setToolTip(
            "Target mesh. The SOURCE model will be warped to match this shape."
        )
        form.addRow("Target model:", self.targetSelector)

        # -------------------- Target fiducials --------------------
        self.targetFiducialSelector = slicer.qMRMLNodeComboBox()
        self.targetFiducialSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.targetFiducialSelector.selectNodeUponCreation = True
        self.targetFiducialSelector.addEnabled = False
        self.targetFiducialSelector.removeEnabled = False
        self.targetFiducialSelector.noneEnabled = False
        self.targetFiducialSelector.setMRMLScene(slicer.mrmlScene)
        self.targetFiducialSelector.setToolTip(
            "Fiducial set on the TARGET model. Must have same ordering as SOURCE fiducials."
        )
        form.addRow("Target fiducials:", self.targetFiducialSelector)
        # -------------------- Use / ignore fiducials --------------------
        self.ignoreFiducialsCheckbox = qt.QCheckBox("Ignore fiducials (geometry-only)")
        self.ignoreFiducialsCheckbox.checked = False  # default: use fiducials if available
        self.ignoreFiducialsCheckbox.setToolTip(
            "If checked, registration will ignore SOURCE/TARGET fiducials even if they are provided."
        )
        form.addRow("Landmarks:", self.ignoreFiducialsCheckbox)

        # -------------------- Output names --------------------
        self.registeredModelNameEdit = qt.QLineEdit()
        self.registeredModelNameEdit.placeholderText = "Use source model name"
        self.registeredModelNameEdit.setToolTip(
            "Name for the registered (deformed) model node.\n"
            "If left empty, the SOURCE model name will be used."
        )
        form.addRow("Registered model name:", self.registeredModelNameEdit)

        self.projectedFiducialsNameEdit = qt.QLineEdit()
        self.projectedFiducialsNameEdit.placeholderText = "Use source fiducials name"
        self.projectedFiducialsNameEdit.setToolTip(
            "Name for projected fiducials on the registered model.\n"
            "If left empty, the SOURCE fiducials name will be used."
        )
        form.addRow("Projected fiducials name:", self.projectedFiducialsNameEdit)

        # -------------------- Methods checkboxes --------------------
        self.computeSumnerCheckbox = qt.QCheckBox("Use Sumner NRICP")
        self.computeSumnerCheckbox.checked = True
        self.computeSumnerCheckbox.setToolTip(
            "Enable the Sumner variant of NRICP (Trimesh implementation)."
        )

        self.computeAmbergCheckbox = qt.QCheckBox("Use Amberg NRICP")
        self.computeAmbergCheckbox.checked = True
        self.computeAmbergCheckbox.setToolTip(
            "Enable the Amberg variant of NRICP (Trimesh implementation)."
        )

        methodsWidget = qt.QWidget()
        methodsLayout = qt.QHBoxLayout()
        methodsLayout.setContentsMargins(0, 0, 0, 0)
        methodsLayout.addWidget(self.computeSumnerCheckbox)
        methodsLayout.addWidget(self.computeAmbergCheckbox)
        methodsWidget.setLayout(methodsLayout)
        form.addRow("Registration methods:", methodsWidget)

        # -------------------- Distance threshold --------------------
        self.distanceThresholdSpinBox = ctk.ctkDoubleSpinBox()
        self.distanceThresholdSpinBox.decimals = 3
        self.distanceThresholdSpinBox.singleStep = 0.1
        self.distanceThresholdSpinBox.minimum = 0.0
        self.distanceThresholdSpinBox.maximum = 1e6
        self.distanceThresholdSpinBox.value = 0.05 # original 1.0
        self.distanceThresholdSpinBox.setToolTip(
            "Maximum correspondence distance (in model units).\n"
            "Larger values allow looser matches but may increase outliers."
        )
        form.addRow("Distance threshold:", self.distanceThresholdSpinBox)

        # -------------------- Apply button --------------------
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Run the non-rigid registration on the selected SOURCE and TARGET."
        self.layout.addWidget(self.applyButton)
        self.applyButton.clicked.connect(self.onApplyButton)

        # ------------------------------------------------------------------
        # Sumner parameters group
        # ------------------------------------------------------------------
        sumnerBox = ctk.ctkCollapsibleButton()
        sumnerBox.text = "Sumner parameters"
        self.layout.addWidget(sumnerBox)
        sumnerLayout = qt.QVBoxLayout(sumnerBox)

        sumnerLabel = qt.QLabel(
            "Sumner optimization steps.\n"
            "Columns: correspondence, identity, smoothness, landmark importance, normal importance."
        )
        sumnerLayout.addWidget(sumnerLabel)


        """ original optim -> hard edges
        defaultSumnerSteps = [
            [1e-05, 0.001, 1.0, 100.0, 0.5],
            [0.001, 0.001, 1.0, 25.0, 0.5],
            [0.01, 0.001, 1.0, 10.0, 0.5],
            [0.1, 0.001, 1.0, 5.0, 0.25],
            [0.5, 0.001, 1.0, 3.5, 0.25],
            [1.0, 0.001, 1.0, 2.5, 0.25],
            [5.0, 0.001, 1.0, 1.0, 0.25],
            [10.0, 0.001, 1.0, 2.5, 0.25],
            [25.0, 0.001, 1.0, 1.0, 0.25],
            [50.0, 0.001, 1.0, 1.0, 0.25],
            [100.0, 0.001, 1.0, 1.0, 0.25],
            [200.0, 0.001, 1.0, 1.0, 0.25],
            [300.0, 0.001, 1.0, 1.0, 0.25],
            [400.0, 0.001, 1.0, 1.0, 0.25],
            [500.0, 0.001, 1.0, 1.0, 0.1],
        ]
        """
        defaultSumnerSteps = [

            # -------------------------------------------------------------
            # PHASE 1 — INITIAL REGULARIZATION (NO CORRESPONDENCES)
            # Generates smooth deformation guided only by landmarks.
            # Prevents early sharp distortions.
            # wC must be exactly 0 here.
            # -------------------------------------------------------------
            # [wc,       wi,     ws,  wl,   wn]
            
            # Strong landmark term; high smoothing; normals active.
            [0.0,    0.001,  1.0, 120.0, 0.80],   # Step 1
            [0.0,    0.001,  1.0, 110.0, 0.80],   # Step 2
            [0.0,    0.001,  1.0, 100.0, 0.80],   # Step 3
            [0.0,    0.001,  1.0,  90.0, 0.80],   # Step 4
            [0.0,    0.001,  1.0,  80.0, 0.80],   # Step 5
            [0.0,    0.001,  1.0,  70.0, 0.80],   # Step 6
            [0.0,    0.001,  1.0,  60.0, 0.80],   # Step 7
            [0.0,    0.001,  1.0,  50.0, 0.80],   # Step 8
            [0.0,    0.001,  1.0,  40.0, 0.80],   # Step 9
            [0.0,    0.001,  1.0,  30.0, 0.80],   # Step 10

            # -------------------------------------------------------------
            # PHASE 2 — WE INTRODUCE CORRESPONDENCES (wC grows smoothly)
            # Sumner recommends wC going 1 → 5000.
            # We implement a very smooth progressive schedule.
            # High smoothness & strong normals to suppress sharp edges.
            # -------------------------------------------------------------

            # Section A — Tiny wC (1 → 20)
            [1.0,    0.001,  1.0,  30.0, 0.85],   # Step 11
            [2.0,    0.001,  1.0,  28.0, 0.85],   # Step 12
            [5.0,    0.001,  1.0,  26.0, 0.85],   # Step 13
            [10.0,   0.001,  1.0,  24.0, 0.85],   # Step 14
            [20.0,   0.001,  1.0,  22.0, 0.85],   # Step 15

            # Section B — Low wC (20 → 150)
            [40.0,   0.001,  1.0,  20.0, 0.90],   # Step 16
            [75.0,   0.001,  1.0,  18.0, 0.92],   # Step 17
            [100.0,  0.001,  1.0,  16.0, 0.94],   # Step 18
            [150.0,  0.001,  1.0,  14.0, 0.96],   # Step 19

            # Section C — Medium wC (150 → 500)
            [200.0,  0.001,  1.0,  13.0, 0.98],   # Step 20
            [300.0,  0.001,  1.0,  12.0, 1.00],   # Step 21
            [400.0,  0.001,  1.0,  11.0, 1.00],   # Step 22
            [500.0,  0.001,  1.0,  10.0, 1.00],   # Step 23

            # Section D — Strong wC (500 → 1000)
            [600.0,  0.001,  1.0,   9.5, 1.00],   # Step 24
            [750.0,  0.001,  1.0,   9.0, 1.00],   # Step 25
            [900.0,  0.001,  1.0,   8.5, 1.00],   # Step 26
            [1000.0, 0.001,  1.0,   8.0, 1.00],   # Step 27

            # Section E — Very strong wC (1000 → 2500)
            [1200.0, 0.001,  1.0,   7.5, 1.00],   # Step 28
            [1500.0, 0.001,  1.0,   7.0, 1.00],   # Step 29
            [1800.0, 0.001,  1.0,   6.5, 1.00],   # Step 30
            [2200.0, 0.001,  1.0,   6.0, 1.00],   # Step 31
            [2500.0, 0.001,  1.0,   5.5, 1.00],   # Step 32

            # Section F — Final aggressive fitting (2500 → 5000)
            # But smooth because we keep strong normals & smoothness.
            [3000.0, 0.001,  1.0,   5.0, 1.00],   # Step 33
            [3500.0, 0.001,  1.0,   4.8, 1.00],   # Step 34
            [4000.0, 0.001,  1.0,   4.6, 1.00],   # Step 35
            [4500.0, 0.001,  1.0,   4.4, 1.00],   # Step 36
            [5000.0, 0.001,  1.0,   4.2, 1.00],   # Step 37
            [5000.0, 0.001,  1.0,   4.0, 1.00],   # Step 38
            [5000.0, 0.001,  1.0,   3.8, 1.00],   # Step 39
            [5000.0, 0.001,  1.0,   3.6, 1.00],   # Step 40
            [5000.0, 0.001,  1.0,   3.4, 1.00],   # Step 41
            [5000.0, 0.001,  1.0,   3.2, 1.00],   # Step 42
        ]

        


        self.stepsSumnerTable = qt.QTableWidget(len(defaultSumnerSteps), 5)
        self.stepsSumnerTable.setHorizontalHeaderLabels(
            ["Correspondence", "Identity", "Smoothness", "Landmark importance", "Normal importance"]
        )
        sumnerLayout.addWidget(self.stepsSumnerTable)

        for i, step in enumerate(defaultSumnerSteps):
            for j, value in enumerate(step):
                item = qt.QTableWidgetItem(str(value))
                self.stepsSumnerTable.setItem(i, j, item)

        # Buttons for Sumner table
        self.addRowSumnerButton = qt.QPushButton("Add Sumner step")
        self.addRowSumnerButton.toolTip = "Add an extra optimization step for the Sumner NRICP method."
        self.addRowSumnerButton.clicked.connect(self.onAddRowSumner)
        sumnerLayout.addWidget(self.addRowSumnerButton)

        self.removeRowSumnerButton = qt.QPushButton("Remove selected Sumner step")
        self.removeRowSumnerButton.toolTip = "Remove the currently selected Sumner step."
        self.removeRowSumnerButton.clicked.connect(self.onRemoveRowSumner)
        sumnerLayout.addWidget(self.removeRowSumnerButton)

        self.restoreDefaultsSumnerButton = qt.QPushButton("Restore default Sumner steps")
        self.restoreDefaultsSumnerButton.toolTip = "Restore the default Sumner step schedule."
        self.restoreDefaultsSumnerButton.clicked.connect(self.onRestoreDefaultStepsSumner)
        sumnerLayout.addWidget(self.restoreDefaultsSumnerButton)

        

        # ------------------------------------------------------------------
        # Amberg parameters group
        # ------------------------------------------------------------------
        ambergBox = ctk.ctkCollapsibleButton()
        ambergBox.text = "Amberg parameters"
        self.layout.addWidget(ambergBox)
        ambergLayout = qt.QVBoxLayout(ambergBox)

        ambergLabel = qt.QLabel(
            "Amberg optimization steps.\n"
            "Columns: smoothness (alpha), landmarks (beta), normal weighting, max iterations."
        )
        ambergLayout.addWidget(ambergLabel)

        """ ORIGINAL OPTIM -> HARD EDGES
        defaultAmbergSteps = [
            [0.05, 10.0, 0.5, 10],
            [0.04, 5.0, 0.5, 10],
            [0.03, 2.5, 0.5, 10],
            [0.02, 1.0, 0.0, 10],
            [0.01, 0.0, 0.0, 10],
            [0.001, 0.0, 0.0, 10],
            [0.0001, 0.0, 0.0, 10],
        ]
        """
        defaultAmbergSteps = [
            # [ws,     wl,   wn,   max_iter]
            # High smoothness + high landmark + high normals
            [0.20, 20.0, 0.90, 16],   # 1
            [0.18, 18.0, 0.92, 16],   # 2
            [0.16, 16.0, 0.93, 16],   # 3
            [0.14, 14.0, 0.94, 18],   # 4
            [0.12, 12.0, 0.95, 18],   # 5

            # Lowering smoothness progressively
            [0.10, 10.0, 0.96, 20],   # 6
            [0.09,  8.0, 0.97, 20],   # 7
            [0.08,  6.0, 0.97, 22],   # 8
            [0.07,  5.0, 0.98, 22],   # 9
            [0.06,  4.0, 0.98, 24],   # 10

            # Mid-low smoothness
            [0.05,  3.0, 0.99, 24],   # 11
            [0.045, 2.5, 0.99, 26],   # 12
            [0.040, 2.0, 1.00, 26],   # 13
            [0.035, 1.5, 1.00, 28],   # 14
            [0.030, 1.2, 1.00, 30],   # 15

            # Final refinement
            [0.028, 1.0, 1.00, 30],   # 16
            [0.026, 0.8, 1.00, 32],   # 17
            [0.024, 0.6, 1.00, 32],   # 18
            [0.022, 0.5, 1.00, 34],   # 19
            [0.020, 0.4, 1.00, 34],   # 20
        ]


        self.stepsAmbergTable = qt.QTableWidget(len(defaultAmbergSteps), 4)
        self.stepsAmbergTable.setHorizontalHeaderLabels(
            ["Smoothness", "Markers", "Normal weighting", "Max iterations"]
        )
        ambergLayout.addWidget(self.stepsAmbergTable)

        for i, step in enumerate(defaultAmbergSteps):
            for j, value in enumerate(step):
                item = qt.QTableWidgetItem(str(value))
                self.stepsAmbergTable.setItem(i, j, item)

        self.addRowAmbergButton = qt.QPushButton("Add Amberg step")
        self.addRowAmbergButton.toolTip = "Add an extra optimization step for the Amberg NRICP method."
        self.addRowAmbergButton.clicked.connect(self.onAddRowAmberg)
        ambergLayout.addWidget(self.addRowAmbergButton)

        self.removeRowAmbergButton = qt.QPushButton("Remove selected Amberg step")
        self.removeRowAmbergButton.toolTip = "Remove the currently selected Amberg step."
        self.removeRowAmbergButton.clicked.connect(self.onRemoveRowAmberg)
        ambergLayout.addWidget(self.removeRowAmbergButton)

        self.restoreDefaultsAmbergButton = qt.QPushButton("Restore default Amberg steps")
        self.restoreDefaultsAmbergButton.toolTip = "Restore the default Amberg step schedule."
        self.restoreDefaultsAmbergButton.clicked.connect(self.onRestoreDefaultStepsAmberg)
        ambergLayout.addWidget(self.restoreDefaultsAmbergButton)

        
        # ------------------------------------------------------------------
        # Batch registration group
        # ------------------------------------------------------------------
        batchBox = ctk.ctkCollapsibleButton()
        batchBox.text = "Batch registration"
        self.layout.addWidget(batchBox)
        batchForm = qt.QFormLayout(batchBox)

        # Target models folder
        self.targetModelDirectoryButton = ctk.ctkDirectoryButton()
        self.targetModelDirectoryButton.setToolTip(
            "Folder containing TARGET meshes (.ply, .obj, .vtk) for batch processing."
        )
        self.targetModelDirectoryButton.setMaximumWidth(400)
        batchForm.addRow("Target models folder:", self.targetModelDirectoryButton)

        # Target markups folder
        self.targetMarkupDirectoryButton = ctk.ctkDirectoryButton()
        self.targetMarkupDirectoryButton.setToolTip(
            "Folder containing TARGET fiducials (.mrk.json) corresponding to each TARGET mesh."
        )
        self.targetMarkupDirectoryButton.setMaximumWidth(400)
        batchForm.addRow("Target fiducials folder:", self.targetMarkupDirectoryButton)

        # Output folder
        self.outputDirectoryButton = ctk.ctkDirectoryButton()
        self.outputDirectoryButton.setToolTip(
            "Folder to save registered models and projected fiducials for batch processing."
        )
        self.outputDirectoryButton.setMaximumWidth(400)
        batchForm.addRow("Output folder:", self.outputDirectoryButton)

        # Batch button
        self.batchProcessButton = qt.QPushButton("Run batch registration")
        self.batchProcessButton.toolTip = (
            "For each TARGET mesh + fiducials in the selected folders,\n"
            "register the SOURCE template and save the result to the output folder."
        )
        batchForm.addRow(self.batchProcessButton)
        self.batchProcessButton.clicked.connect(self.onBatchProcessButtonClicked)

        self.layout.addStretch(1)

    # ------------------------------------------------------------------
    # Helpers to read tables safely (skip invalid rows)
    # ------------------------------------------------------------------
    def _readAmbergSteps(self):
        steps = []
        rows = self.stepsAmbergTable.rowCount
        cols = self.stepsAmbergTable.columnCount  # should be 4
        for row in range(rows):
            rowValues = []
            validRow = True
            for col in range(cols):
                item = self.stepsAmbergTable.item(row, col)
                if item is None:
                    validRow = False
                    break
                text = item.text().strip()
                if not text:
                    validRow = False
                    break
                try:
                    # last column: max iterations (int), others float
                    if col == 3:
                        value = int(text)
                    else:
                        value = float(text)
                except ValueError:
                    validRow = False
                    break
                rowValues.append(value)
            if validRow:
                steps.append(rowValues)
        return steps

    def _readSumnerSteps(self):
        steps = []
        rows = self.stepsSumnerTable.rowCount
        cols = self.stepsSumnerTable.columnCount  # should be 5
        for row in range(rows):
            rowValues = []
            validRow = True
            for col in range(cols):
                item = self.stepsSumnerTable.item(row, col)
                if item is None:
                    validRow = False
                    break
                text = item.text().strip()
                if not text:
                    validRow = False
                    break
                try:
                    value = float(text)
                except ValueError:
                    validRow = False
                    break
                rowValues.append(value)
            if validRow:
                steps.append(rowValues)
        return steps

    # ------------------------------------------------------------------
    # Single registration
    # ------------------------------------------------------------------
    def onApplyButton(self):
        sourceModel = self.sourceSelector.currentNode()
        targetModel = self.targetSelector.currentNode()
        sourceFiducial = self.sourceFiducialSelector.currentNode()
        targetFiducial = self.targetFiducialSelector.currentNode()

        if not sourceModel or not targetModel:
            slicer.util.errorDisplay("Please select both SOURCE and TARGET model nodes.")
            return
        
        # Fiducials optional: either both or none
        useLandmarks = not self.ignoreFiducialsCheckbox.checked

        if useLandmarks:
            # Fiducials optional: either both or none
            if (sourceFiducial is None) != (targetFiducial is None):
                slicer.util.errorDisplay("Either provide both SOURCE and TARGET fiducials or none.")
                return
            if sourceFiducial is None and targetFiducial is None:
                slicer.util.warningDisplay(
                    "No SOURCE/TARGET fiducials selected.\n"
                    "Registration will run without the landmark term."
                )
        else:
            # We explicitly ignore fiducials; just log if they exist
            if sourceFiducial or targetFiducial:
                print("[TrimeshReg] INFO: Fiducials are present but will be ignored (Ignore fiducials checkbox).")


        useAmberg = bool(self.computeAmbergCheckbox.checked)
        useSumner = bool(self.computeSumnerCheckbox.checked)
        if not useAmberg and not useSumner:
            slicer.util.errorDisplay("Please enable at least one method: Sumner or Amberg.")
            return

        stepsAmberg = self._readAmbergSteps() if useAmberg else []
        stepsSumner = self._readSumnerSteps() if useSumner else []

        distance_threshold = float(self.distanceThresholdSpinBox.value)

        # Default names: same as original nodes if user leaves fields empty
        registeredModelName = self.registeredModelNameEdit.text.strip()
        if not registeredModelName:
            registeredModelName = targetModel.GetName()

        projectedFiducialsName = self.projectedFiducialsNameEdit.text.strip()
        if not projectedFiducialsName:
            if useLandmarks and sourceFiducial:
                projectedFiducialsName = sourceFiducial.GetName()
            else:
                projectedFiducialsName = registeredModelName + "_Fiducials"



        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            projectedModelNode, metrics = self.logic.run(
                                                            sourceModel,
                                                            targetModel,
                                                            sourceFiducial,
                                                            targetFiducial,
                                                            useSumner,
                                                            stepsSumner,
                                                            useAmberg,
                                                            stepsAmberg,
                                                            distance_threshold,
                                                            registeredModelName,
                                                            projectedFiducialsName,
                                                            useLandmarks,
                                                        )

            # You can optionally print a quick summary:
            if projectedModelNode:
                print("[TrimeshReg] Registration done.")
                print(f"[TrimeshReg] Total time: {metrics.get('total_time')}")
                print(f"[TrimeshReg] Final mean distance: {metrics.get('final_mean_distance')}")

        except Exception as e:
            slicer.app.restoreOverrideCursor()
            slicer.util.errorDisplay(f"Registration failed:\n{e}")
            raise
        finally:
            slicer.app.restoreOverrideCursor()

        if projectedModelNode:
            print("[TrimeshReg] Registration done.")

    # ------------------------------------------------------------------
    # Batch registration
    # ------------------------------------------------------------------
    def onBatchProcessButtonClicked(self):
        sourceModel = self.sourceSelector.currentNode()
        sourceFiducial = self.sourceFiducialSelector.currentNode()

        if not sourceModel:
            slicer.util.errorDisplay("Please select a SOURCE model (template) for batch registration.")
            return
        
        useLandmarks = not self.ignoreFiducialsCheckbox.checked

        if useLandmarks and not sourceFiducial:
            slicer.util.errorDisplay("Please select SOURCE fiducials for batch registration.")
            return

        useAmberg = bool(self.computeAmbergCheckbox.checked)
        useSumner = bool(self.computeSumnerCheckbox.checked)
        if not useAmberg and not useSumner:
            slicer.util.errorDisplay("Please enable at least one method: Sumner or Amberg.")
            return

        stepsAmberg = self._readAmbergSteps() if useAmberg else []
        stepsSumner = self._readSumnerSteps() if useSumner else []

        distance_threshold = float(self.distanceThresholdSpinBox.value)

        modelDirectory = self.targetModelDirectoryButton.directory
        markupDirectory = self.targetMarkupDirectoryButton.directory
        outputDirectory = self.outputDirectoryButton.directory

        if not modelDirectory or not os.path.isdir(modelDirectory):
            slicer.util.errorDisplay("Please select a valid TARGET models folder.")
            return
        if useLandmarks and (not markupDirectory or not os.path.isdir(markupDirectory)):
            slicer.util.errorDisplay("Please select a valid TARGET fiducials folder.")
            return
        if not outputDirectory:
            slicer.util.errorDisplay("Please select an output folder.")
            return
        if not os.path.isdir(outputDirectory):
            os.makedirs(outputDirectory, exist_ok=True)

        print("[TrimeshReg][Batch] Starting batch registration...")
        print(f"[TrimeshReg][Batch] Template: {sourceModel.GetName()}")
        print(f"[TrimeshReg][Batch] Models folder: {modelDirectory}")
        print(f"[TrimeshReg][Batch] Markups folder: {markupDirectory}")
        print(f"[TrimeshReg][Batch] Output folder: {outputDirectory}")

        total = 0
        success = 0
        failures = []
        results = []  # per-mesh log rows

        exts = (".ply", ".obj", ".vtk")

        # for global stats
        total_times = []
        final_mean_distances = []
        sumner_times = []
        amberg_times = []

        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)

            for modelFileName in os.listdir(modelDirectory):
                if not modelFileName.lower().endswith(exts):
                    continue

                total += 1
                modelFilePath = os.path.join(modelDirectory, modelFileName)
                baseName = os.path.splitext(modelFileName)[0]
                markupFileName = baseName + ".mrk.json"
                markupFilePath = os.path.join(markupDirectory, markupFileName) if useLandmarks else None

                if useLandmarks and not os.path.exists(markupFilePath):
                    msg = f"Markup file not found for model: {modelFileName} (expected: {markupFilePath})"
                    print(f"[TrimeshReg][Batch][Skip] {msg}")
                    failures.append((modelFileName, "markup not found"))
                    results.append({
                        "filename": modelFileName,
                        "status": "fail",
                        "error": "markup not found",
                        "use_sumner": useSumner,
                        "use_amberg": useAmberg,
                        "distance_threshold": distance_threshold,
                        "sumner_time": "",
                        "amberg_time": "",
                        "total_time": "",
                        "sumner_total_distance": "",
                        "sumner_mean_distance": "",
                        "final_total_distance": "",
                        "final_mean_distance": "",
                        "num_source_points": "",
                        "num_target_points": "",
                        "num_source_landmarks": "",
                        "num_target_landmarks": "",
                    })
                    continue

                print(f"[TrimeshReg][Batch] Processing {modelFileName} ...")

                registeredModelName = baseName
                projectedFiducialsName = baseName + "_Fiducials"

                try:
                    # Load target model
                    targetModelNode = slicer.util.loadModel(modelFilePath)
                    if not targetModelNode:
                        msg = "Failed to load target model."
                        print(f"[TrimeshReg][Batch][Error] {msg}")
                        failures.append((modelFileName, msg))
                        results.append({
                            "filename": modelFileName,
                            "status": "fail",
                            "error": msg,
                            "use_sumner": useSumner,
                            "use_amberg": useAmberg,
                            "distance_threshold": distance_threshold,
                            "sumner_time": "",
                            "amberg_time": "",
                            "total_time": "",
                            "sumner_total_distance": "",
                            "sumner_mean_distance": "",
                            "final_total_distance": "",
                            "final_mean_distance": "",
                            "num_source_points": "",
                            "num_target_points": "",
                            "num_source_landmarks": "",
                            "num_target_landmarks": "",
                        })
                        continue

                    # Load target fiducials (no returnNode, returns node directly)
                    if useLandmarks:
                        targetFiducialNode = slicer.util.loadMarkups(markupFilePath)
                        if not targetFiducialNode:
                            msg = "Failed to load target fiducials."
                            print(f"[TrimeshReg][Batch][Error] {msg}")
                            failures.append((modelFileName, msg))
                            results.append({
                                "filename": modelFileName,
                                "status": "fail",
                                "error": msg,
                                "use_sumner": useSumner,
                                "use_amberg": useAmberg,
                                "distance_threshold": distance_threshold,
                                "sumner_time": "",
                                "amberg_time": "",
                                "total_time": "",
                                "sumner_total_distance": "",
                                "sumner_mean_distance": "",
                                "final_total_distance": "",
                                "final_mean_distance": "",
                                "num_source_points": "",
                                "num_target_points": "",
                                "num_source_landmarks": "",
                                "num_target_landmarks": "",
                            })
                            # clean up model node
                            slicer.mrmlScene.RemoveNode(targetModelNode)
                            continue
                    else:
                        targetFiducialNode = None

                    # Hide target model
                    if targetModelNode.GetDisplayNode():
                        targetModelNode.GetDisplayNode().SetVisibility(False)

                    projectedModelNode, metrics = self.logic.run(
                        sourceModel,
                        targetModelNode,
                        sourceFiducial,
                        targetFiducialNode,
                        useSumner,
                        stepsSumner,
                        useAmberg,
                        stepsAmberg,
                        distance_threshold,
                        registeredModelName,
                        projectedFiducialsName,
                        useLandmarks,
                    )

                    # Remove target nodes from scene
                    slicer.mrmlScene.RemoveNode(targetModelNode)
                    if targetFiducialNode:
                        slicer.mrmlScene.RemoveNode(targetFiducialNode)

                    if projectedModelNode is None:
                        msg = "Logic returned no projected model."
                        print(f"[TrimeshReg][Batch][Error] {msg}")
                        failures.append((modelFileName, msg))
                        results.append({
                            "filename": modelFileName,
                            "status": "fail",
                            "error": msg,
                            "use_sumner": useSumner,
                            "use_amberg": useAmberg,
                            "distance_threshold": distance_threshold,
                            "sumner_time": "",
                            "amberg_time": "",
                            "total_time": "",
                            "sumner_total_distance": "",
                            "sumner_mean_distance": "",
                            "final_total_distance": "",
                            "final_mean_distance": "",
                            "num_source_points": "",
                            "num_target_points": "",
                            "num_source_landmarks": "",
                            "num_target_landmarks": "",
                        })
                        continue

                    # Save registered model
                    registeredModelPath = os.path.join(outputDirectory, registeredModelName + ".vtk")
                    if not slicer.util.saveNode(projectedModelNode, registeredModelPath):
                        msg = "Could not save registered model."
                        print(f"[TrimeshReg][Batch][Error] {msg}")
                        failures.append((modelFileName, msg))
                        results.append({
                            "filename": modelFileName,
                            "status": "fail",
                            "error": msg,
                            "use_sumner": metrics.get("use_sumner"),
                            "use_amberg": metrics.get("use_amberg"),
                            "distance_threshold": metrics.get("distance_threshold"),
                            "sumner_time": metrics.get("sumner_time") or "",
                            "amberg_time": metrics.get("amberg_time") or "",
                            "total_time": metrics.get("total_time") or "",
                            "sumner_total_distance": metrics.get("sumner_total_distance") or "",
                            "sumner_mean_distance": metrics.get("sumner_mean_distance") or "",
                            "final_total_distance": metrics.get("final_total_distance") or "",
                            "final_mean_distance": metrics.get("final_mean_distance") or "",
                            "num_source_points": metrics.get("num_source_points") or "",
                            "num_target_points": metrics.get("num_target_points") or "",
                            "num_source_landmarks": metrics.get("num_source_landmarks") or "",
                            "num_target_landmarks": metrics.get("num_target_landmarks") or "",
                        })
                        continue

                    # Save projected fiducials
                    if useLandmarks:
                        projectedFiducialsNode = slicer.util.getNode(projectedFiducialsName)
                        projectedFiducialsPath = os.path.join(outputDirectory, projectedFiducialsName + ".json")
                        if not slicer.util.saveNode(projectedFiducialsNode, projectedFiducialsPath):
                            msg = "Could not save projected fiducials."
                            print(f"[TrimeshReg][Batch][Error] {msg}")
                            failures.append((modelFileName, msg))
                            results.append({
                                "filename": modelFileName,
                                "status": "fail",
                                "error": msg,
                                "use_sumner": metrics.get("use_sumner"),
                                "use_amberg": metrics.get("use_amberg"),
                                "distance_threshold": metrics.get("distance_threshold"),
                                "sumner_time": metrics.get("sumner_time") or "",
                                "amberg_time": metrics.get("amberg_time") or "",
                                "total_time": metrics.get("total_time") or "",
                                "sumner_total_distance": metrics.get("sumner_total_distance") or "",
                                "sumner_mean_distance": metrics.get("sumner_mean_distance") or "",
                                "final_total_distance": metrics.get("final_total_distance") or "",
                                "final_mean_distance": metrics.get("final_mean_distance") or "",
                                "num_source_points": metrics.get("num_source_points") or "",
                                "num_target_points": metrics.get("num_target_points") or "",
                                "num_source_landmarks": metrics.get("num_source_landmarks") or "",
                                "num_target_landmarks": metrics.get("num_target_landmarks") or "",
                            })
                            continue

                    # Successful case
                    success += 1
                    print(f"[TrimeshReg][Batch] Saved results for {modelFileName}")

                    total_times.append(metrics.get("total_time") or 0.0)
                    if metrics.get("final_mean_distance") is not None:
                        final_mean_distances.append(metrics["final_mean_distance"])
                    if metrics.get("sumner_time") is not None:
                        sumner_times.append(metrics["sumner_time"])
                    if metrics.get("amberg_time") is not None:
                        amberg_times.append(metrics["amberg_time"])

                    results.append({
                        "filename": modelFileName,
                        "status": "ok",
                        "error": "",
                        "use_sumner": metrics.get("use_sumner"),
                        "use_amberg": metrics.get("use_amberg"),
                        "distance_threshold": metrics.get("distance_threshold"),
                        "sumner_time": metrics.get("sumner_time") or "",
                        "amberg_time": metrics.get("amberg_time") or "",
                        "total_time": metrics.get("total_time") or "",
                        "sumner_total_distance": metrics.get("sumner_total_distance") or "",
                        "sumner_mean_distance": metrics.get("sumner_mean_distance") or "",
                        "final_total_distance": metrics.get("final_total_distance") or "",
                        "final_mean_distance": metrics.get("final_mean_distance") or "",
                        "num_source_points": metrics.get("num_source_points") or "",
                        "num_target_points": metrics.get("num_target_points") or "",
                        "num_source_landmarks": metrics.get("num_source_landmarks") or "",
                        "num_target_landmarks": metrics.get("num_target_landmarks") or "",
                    })

                except Exception as e:
                    msg = str(e)
                    print(f"[TrimeshReg][Batch][Exception] {modelFileName}: {msg}")
                    failures.append((modelFileName, msg))
                    results.append({
                        "filename": modelFileName,
                        "status": "fail",
                        "error": msg,
                        "use_sumner": useSumner,
                        "use_amberg": useAmberg,
                        "distance_threshold": distance_threshold,
                        "sumner_time": "",
                        "amberg_time": "",
                        "total_time": "",
                        "sumner_total_distance": "",
                        "sumner_mean_distance": "",
                        "final_total_distance": "",
                        "final_mean_distance": "",
                        "num_source_points": "",
                        "num_target_points": "",
                        "num_source_landmarks": "",
                        "num_target_landmarks": "",
                    })
                    # Attempt to clean up any intermediate nodes
                    for nodeName in [registeredModelName, projectedFiducialsName]:
                        try:
                            node = slicer.util.getNode(nodeName)
                            slicer.mrmlScene.RemoveNode(node)
                        except Exception:
                            pass
                    continue

        finally:
            slicer.app.restoreOverrideCursor()

        # ---- Compute global stats ----
        total_time_all = sum(total_times) if total_times else 0.0
        mean_time = (total_time_all / len(total_times)) if total_times else 0.0
        mean_final_mean_distance = (
            sum(final_mean_distances) / len(final_mean_distances)
            if final_mean_distances
            else 0.0
        )
        mean_sumner_time = (
            sum(sumner_times) / len(sumner_times) if sumner_times else 0.0
        )
        mean_amberg_time = (
            sum(amberg_times) / len(amberg_times) if amberg_times else 0.0
        )

        # ---- Write CSV log in output folder ----
        csv_path = os.path.join(outputDirectory, "batch_log_trimesh_nricp.csv")
        try:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)

                # Header with configuration
                writer.writerow(["CONFIGURATION"])
                writer.writerow(["use_sumner", useSumner])
                writer.writerow(["use_amberg", useAmberg])
                writer.writerow(["distance_threshold", distance_threshold])
                writer.writerow(["steps_sumner", repr(stepsSumner)])
                writer.writerow(["steps_amberg", repr(stepsAmberg)])
                writer.writerow([])

                # Per-mesh rows
                writer.writerow([
                    "filename",
                    "status",
                    "error",
                    "use_sumner",
                    "use_amberg",
                    "distance_threshold",
                    "sumner_time",
                    "amberg_time",
                    "total_time",
                    "sumner_total_distance",
                    "sumner_mean_distance",
                    "final_total_distance",
                    "final_mean_distance",
                    "num_source_points",
                    "num_target_points",
                    "num_source_landmarks",
                    "num_target_landmarks",
                ])
                for row in results:
                    writer.writerow([
                        row["filename"],
                        row["status"],
                        row["error"],
                        row["use_sumner"],
                        row["use_amberg"],
                        row["distance_threshold"],
                        row["sumner_time"],
                        row["amberg_time"],
                        row["total_time"],
                        row["sumner_total_distance"],
                        row["sumner_mean_distance"],
                        row["final_total_distance"],
                        row["final_mean_distance"],
                        row["num_source_points"],
                        row["num_target_points"],
                        row["num_source_landmarks"],
                        row["num_target_landmarks"],
                    ])

                # Summary
                writer.writerow([])
                writer.writerow(["SUMMARY"])
                writer.writerow(["total_models", total])
                writer.writerow(["success", success])
                writer.writerow(["failures", len(failures)])
                writer.writerow(["total_time_all", total_time_all])
                writer.writerow(["mean_total_time", mean_time])
                writer.writerow(["mean_final_mean_distance", mean_final_mean_distance])
                writer.writerow(["mean_sumner_time", mean_sumner_time])
                writer.writerow(["mean_amberg_time", mean_amberg_time])

            print(f"[TrimeshReg][Batch] Wrote log CSV: {csv_path}")
        except Exception as e:
            print(f"[TrimeshReg][Batch] Could not write log CSV: {e}")

        summary = (
            f"Batch registration finished.\n\n"
            f"Total target models found: {total}\n"
            f"Successfully processed:    {success}\n"
            f"Failed or skipped:         {len(failures)}\n\n"
            f"Total time (all meshes):   {total_time_all:.3f} s\n"
            f"Mean time per mesh:        {mean_time:.3f} s\n"
            f"Mean final mean distance:  {mean_final_mean_distance:.6f}"
        )
        slicer.util.infoDisplay(summary, "ETSE_UV__SumnerAmbergRegistration batch")

        if failures:
            print("[TrimeshReg][Batch] Failures:")
            for fname, reason in failures:
                print(f"  - {fname}: {reason}")

    # ------------------------------------------------------------------
    # Sumner table handlers
    # ------------------------------------------------------------------
    def onAddRowSumner(self):
        currentRowCount = self.stepsSumnerTable.rowCount
        self.stepsSumnerTable.insertRow(currentRowCount)
        for i in range(self.stepsSumnerTable.columnCount):
            self.stepsSumnerTable.setItem(currentRowCount, i, qt.QTableWidgetItem(""))

    def onRemoveRowSumner(self):
        selectedRow = self.stepsSumnerTable.currentRow()
        if selectedRow != -1:
            self.stepsSumnerTable.removeRow(selectedRow)

    
    def onRestoreDefaultStepsSumner(self):
        defaultSteps = [

            # -------------------------------------------------------------
            # PHASE 1 — INITIAL REGULARIZATION (NO CORRESPONDENCES)
            # Generates smooth deformation guided only by landmarks.
            # Prevents early sharp distortions.
            # wC must be exactly 0 here.
            # -------------------------------------------------------------
            # [wc,       wi,     ws,  wl,   wn]
            
            # Strong landmark term; high smoothing; normals active.
            [0.0,    0.001,  1.0, 120.0, 0.80],   # Step 1
            [0.0,    0.001,  1.0, 110.0, 0.80],   # Step 2
            [0.0,    0.001,  1.0, 100.0, 0.80],   # Step 3
            [0.0,    0.001,  1.0,  90.0, 0.80],   # Step 4
            [0.0,    0.001,  1.0,  80.0, 0.80],   # Step 5
            [0.0,    0.001,  1.0,  70.0, 0.80],   # Step 6
            [0.0,    0.001,  1.0,  60.0, 0.80],   # Step 7
            [0.0,    0.001,  1.0,  50.0, 0.80],   # Step 8
            [0.0,    0.001,  1.0,  40.0, 0.80],   # Step 9
            [0.0,    0.001,  1.0,  30.0, 0.80],   # Step 10

            # -------------------------------------------------------------
            # PHASE 2 — WE INTRODUCE CORRESPONDENCES (wC grows smoothly)
            # Sumner recommends wC going 1 → 5000.
            # We implement a very smooth progressive schedule.
            # High smoothness & strong normals to suppress sharp edges.
            # -------------------------------------------------------------

            # Section A — Tiny wC (1 → 20)
            [1.0,    0.001,  1.0,  30.0, 0.85],   # Step 11
            [2.0,    0.001,  1.0,  28.0, 0.85],   # Step 12
            [5.0,    0.001,  1.0,  26.0, 0.85],   # Step 13
            [10.0,   0.001,  1.0,  24.0, 0.85],   # Step 14
            [20.0,   0.001,  1.0,  22.0, 0.85],   # Step 15

            # Section B — Low wC (20 → 150)
            [40.0,   0.001,  1.0,  20.0, 0.90],   # Step 16
            [75.0,   0.001,  1.0,  18.0, 0.92],   # Step 17
            [100.0,  0.001,  1.0,  16.0, 0.94],   # Step 18
            [150.0,  0.001,  1.0,  14.0, 0.96],   # Step 19

            # Section C — Medium wC (150 → 500)
            [200.0,  0.001,  1.0,  13.0, 0.98],   # Step 20
            [300.0,  0.001,  1.0,  12.0, 1.00],   # Step 21
            [400.0,  0.001,  1.0,  11.0, 1.00],   # Step 22
            [500.0,  0.001,  1.0,  10.0, 1.00],   # Step 23

            # Section D — Strong wC (500 → 1000)
            [600.0,  0.001,  1.0,   9.5, 1.00],   # Step 24
            [750.0,  0.001,  1.0,   9.0, 1.00],   # Step 25
            [900.0,  0.001,  1.0,   8.5, 1.00],   # Step 26
            [1000.0, 0.001,  1.0,   8.0, 1.00],   # Step 27

            # Section E — Very strong wC (1000 → 2500)
            [1200.0, 0.001,  1.0,   7.5, 1.00],   # Step 28
            [1500.0, 0.001,  1.0,   7.0, 1.00],   # Step 29
            [1800.0, 0.001,  1.0,   6.5, 1.00],   # Step 30
            [2200.0, 0.001,  1.0,   6.0, 1.00],   # Step 31
            [2500.0, 0.001,  1.0,   5.5, 1.00],   # Step 32

            # Section F — Final aggressive fitting (2500 → 5000)
            # But smooth because we keep strong normals & smoothness.
            [3000.0, 0.001,  1.0,   5.0, 1.00],   # Step 33
            [3500.0, 0.001,  1.0,   4.8, 1.00],   # Step 34
            [4000.0, 0.001,  1.0,   4.6, 1.00],   # Step 35
            [4500.0, 0.001,  1.0,   4.4, 1.00],   # Step 36
            [5000.0, 0.001,  1.0,   4.2, 1.00],   # Step 37
            [5000.0, 0.001,  1.0,   4.0, 1.00],   # Step 38
            [5000.0, 0.001,  1.0,   3.8, 1.00],   # Step 39
            [5000.0, 0.001,  1.0,   3.6, 1.00],   # Step 40
            [5000.0, 0.001,  1.0,   3.4, 1.00],   # Step 41
            [5000.0, 0.001,  1.0,   3.2, 1.00],   # Step 42
        ]

        self.stepsSumnerTable.clearContents()
        self.stepsSumnerTable.setRowCount(len(defaultSteps))
        for i, step in enumerate(defaultSteps):
            for j, value in enumerate(step):
                item = qt.QTableWidgetItem(str(value))
                self.stepsSumnerTable.setItem(i, j, item)

        self.distanceThresholdSpinBox.value = 1.0

    # ------------------------------------------------------------------
    # Amberg table handlers
    # ------------------------------------------------------------------
    def onAddRowAmberg(self):
        currentRowCount = self.stepsAmbergTable.rowCount
        self.stepsAmbergTable.insertRow(currentRowCount)
        for i in range(self.stepsAmbergTable.columnCount):
            self.stepsAmbergTable.setItem(currentRowCount, i, qt.QTableWidgetItem(""))

    def onRemoveRowAmberg(self):
        selectedRow = self.stepsAmbergTable.currentRow()
        if selectedRow != -1:
            self.stepsAmbergTable.removeRow(selectedRow)


    def onRestoreDefaultStepsAmberg(self):
        defaultSteps = [
            # [ws,     wl,   wn,   max_iter]
            # High smoothness + high landmark + high normals
            [0.20, 20.0, 0.90, 16],   # 1
            [0.18, 18.0, 0.92, 16],   # 2
            [0.16, 16.0, 0.93, 16],   # 3
            [0.14, 14.0, 0.94, 18],   # 4
            [0.12, 12.0, 0.95, 18],   # 5

            # Lowering smoothness progressively
            [0.10, 10.0, 0.96, 20],   # 6
            [0.09,  8.0, 0.97, 20],   # 7
            [0.08,  6.0, 0.97, 22],   # 8
            [0.07,  5.0, 0.98, 22],   # 9
            [0.06,  4.0, 0.98, 24],   # 10

            # Mid-low smoothness
            [0.05,  3.0, 0.99, 24],   # 11
            [0.045, 2.5, 0.99, 26],   # 12
            [0.040, 2.0, 1.00, 26],   # 13
            [0.035, 1.5, 1.00, 28],   # 14
            [0.030, 1.2, 1.00, 30],   # 15

            # Final refinement
            [0.028, 1.0, 1.00, 30],   # 16
            [0.026, 0.8, 1.00, 32],   # 17
            [0.024, 0.6, 1.00, 32],   # 18
            [0.022, 0.5, 1.00, 34],   # 19
            [0.020, 0.4, 1.00, 34],   # 20
        ]

        self.stepsAmbergTable.clearContents()
        self.stepsAmbergTable.setRowCount(len(defaultSteps))
        for i, step in enumerate(defaultSteps):
            for j, value in enumerate(step):
                item = qt.QTableWidgetItem(str(value))
                self.stepsAmbergTable.setItem(i, j, item)

        self.distanceThresholdSpinBox.value = 1.0


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class ETSE_UV__SumnerAmbergRegistrationLogic(ScriptedLoadableModuleLogic):
    def run(
        self,
        sourceModel,
        targetModel,
        sourceFiducial,
        targetFiducial,
        useSumner,
        stepsSumner,
        useAmberg,
        stepsAmberg,
        distance_threshold,
        registeredModelName="RegisteredModel",
        projectedFiducialsName="ProjectedFiducials",
        useLandmarks=True,
    ):
        if not sourceModel or not targetModel:
            raise RuntimeError("Source and Target models are required.")
        # Fiducials are optional but must be provided in pairs
        if useLandmarks and ((sourceFiducial is None) != (targetFiducial is None)):
            raise RuntimeError("Either provide both SOURCE and TARGET fiducials or none.")


        total_start = time.time()

        # Get meshes
        num_vertex_source = sourceModel.GetPolyData().GetNumberOfPoints()
        num_vertex_target = targetModel.GetPolyData().GetNumberOfPoints()
        sourceMesh = self.vtkToTrimesh(sourceModel.GetPolyData())
        targetMesh = self.vtkToTrimesh(targetModel.GetPolyData())

        # Fiducials → numpy (optional)
        if useLandmarks and sourceFiducial and targetFiducial:
            num_marks_source = sourceFiducial.GetNumberOfControlPoints()
            num_marks_target = targetFiducial.GetNumberOfControlPoints()

            sourceLandmarks = self.fiducialsToArray(sourceFiducial)
            targetLandmarks = self.fiducialsToArray(targetFiducial)

            # Map source landmarks to nearest vertex indices
            sourceLandmarksIndices = np.array(
                [
                    np.argmin(np.linalg.norm(sourceMesh.vertices - landmark, axis=1))
                    for landmark in sourceLandmarks
                ],
                dtype=np.int64,
            )
        else:
            # No landmarks: disable landmark term and run purely geometry-based NRICP
            num_marks_source = 0
            num_marks_target = 0
            sourceLandmarks = None
            targetLandmarks = None
            sourceLandmarksIndices = None

            # Zero-out landmark weights in schedules (keep other weights as user defined)
            if stepsSumner:
                stepsSumner = [
                    [wc, wi, ws, 0.0, wn] for (wc, wi, ws, wl, wn) in stepsSumner
                ]
            if stepsAmberg:
                stepsAmberg = [
                    [ws, 0.0, wn, max_iter] for (ws, wl, wn, max_iter) in stepsAmberg
                ]



        # Normalize steps None/empty
        if len(stepsSumner) == 0:
            stepsSumner = None
        if len(stepsAmberg) == 0:
            stepsAmberg = None

        templateModelName = sourceModel.GetName() if sourceModel else "Unknown"
        targetModelName = targetModel.GetName() if targetModel else "Unknown"

        # metrics dict that we will return
        metrics = {
            "template_model": templateModelName,
            "target_model": targetModelName,
            "num_source_points": int(num_vertex_source),
            "num_target_points": int(num_vertex_target),
            "num_source_landmarks": int(num_marks_source),
            "num_target_landmarks": int(num_marks_target),
            "use_sumner": bool(useSumner),
            "use_amberg": bool(useAmberg),
            "distance_threshold": float(distance_threshold),
            "steps_sumner": stepsSumner,
            "steps_amberg": stepsAmberg,
            "sumner_time": None,
            "amberg_time": None,
            "total_time": None,
            "sumner_total_distance": None,
            "sumner_mean_distance": None,
            "final_total_distance": None,
            "final_mean_distance": None,
        }

        print("[TrimeshReg] Calculating registration...")
        print(f"[TrimeshReg] Template model: {templateModelName}")
        print(f"[TrimeshReg] Template points: {num_vertex_source}")
        print(f"[TrimeshReg] Template markups: {num_marks_source}")
        print(f"[TrimeshReg] Target model: {targetModelName}")
        print(f"[TrimeshReg] Target points: {num_vertex_target}")
        print(f"[TrimeshReg] Target markups: {num_marks_target}")
        print("")
        print(f"[TrimeshReg] distance_threshold: {distance_threshold}")
        print(f"[TrimeshReg] Projected model base name: {registeredModelName}")
        print("")

        projectedVertices = None

        # ---------------- Sumner ----------------
        if useSumner:
            print("[TrimeshReg] -- Running Sumner NRICP --")
            print(f"[TrimeshReg] stepsSumner: {stepsSumner}")
            start_time = time.time()
            projectedVertices = trimesh.registration.nricp_sumner(
                source_mesh=sourceMesh,
                target_geometry=targetMesh,
                source_landmarks=sourceLandmarksIndices,
                target_positions=targetLandmarks,
                steps=stepsSumner,
                distance_threshold=distance_threshold,
                use_faces=True,
                use_vertex_normals=True,
                neighbors_count=8,
                face_pairs_type="vertex",
            )
            elapsed_time = time.time() - start_time
            metrics["sumner_time"] = float(elapsed_time)
            print(f"[TrimeshReg] Elapsed time (Sumner) = {elapsed_time:.3f} s")
            slicer.app.processEvents()

            if useAmberg:
                # Create an intermediate Sumner result node
                sourceMesh.vertices = projectedVertices
                projectedModelNodeSumner = self.createProjectedModelNode(
                    sourceMesh, registeredModelName + "_sumner"
                )

                # If landmarks exist, propagate them to the Sumner result;
                # otherwise, keep Amberg fully landmark-free.
                if useLandmarks and sourceLandmarksIndices is not None and sourceFiducial is not None:
                    sourceFiducialSumner = self.createProjectedFiducials(
                        sourceLandmarksIndices,
                        projectedVertices,
                        projectedFiducialsName + "_sumner",
                        sourceFiducials=sourceFiducial,
                    )

                    # Update sourceMesh and landmarks for Amberg stage
                    sourceMesh = self.vtkToTrimesh(projectedModelNodeSumner.GetPolyData())
                    sourceLandmarks = self.fiducialsToArray(sourceFiducialSumner)
                    sourceLandmarksIndices = np.array(
                        [
                            np.argmin(np.linalg.norm(sourceMesh.vertices - landmark, axis=1))
                            for landmark in sourceLandmarks
                        ],
                        dtype=np.int64,
                    )
                else:
                    sourceMesh = self.vtkToTrimesh(projectedModelNodeSumner.GetPolyData())
                    sourceLandmarks = None
                    sourceLandmarksIndices = None

                # Distance to target after Sumner
                if (
                    projectedModelNodeSumner.GetPolyData()
                    and projectedModelNodeSumner.GetPolyData().GetNumberOfPoints() > 0
                    and targetModel.GetPolyData()
                    and targetModel.GetPolyData().GetNumberOfPoints() > 0
                ):

                    distanceFilter = vtk.vtkDistancePolyDataFilter()
                    distanceFilter.SetInputData(0, projectedModelNodeSumner.GetPolyData())
                    distanceFilter.SetInputData(1, targetModel.GetPolyData())
                    distanceFilter.SignedDistanceOff()
                    distanceFilter.Update()

                    distanceData = distanceFilter.GetOutput().GetPointData().GetScalars()
                    distArray = vtk_np.vtk_to_numpy(distanceData)
                    totalDistance = float(np.sum(distArray))
                    meanDistance = totalDistance / projectedModelNodeSumner.GetPolyData().GetNumberOfPoints()

                    metrics["sumner_total_distance"] = totalDistance
                    metrics["sumner_mean_distance"] = meanDistance

                    print(
                        f"[TrimeshReg] Sumner: total distance to target = {totalDistance}, "
                        f"mean distance per point = {meanDistance}"
                    )

        # ---------------- Amberg ----------------
        if useAmberg:
            print("[TrimeshReg] -- Running Amberg NRICP --")
            print(f"[TrimeshReg] stepsAmberg: {stepsAmberg}")
            start_time = time.time()
            projectedVertices = trimesh.registration.nricp_amberg(
                source_mesh=sourceMesh,
                target_geometry=targetMesh,
                source_landmarks=sourceLandmarksIndices,
                target_positions=targetLandmarks,
                steps=stepsAmberg,
                distance_threshold=distance_threshold,
            )
            elapsed_time = time.time() - start_time
            metrics["amberg_time"] = float(elapsed_time)
            print(f"[TrimeshReg] Elapsed time (Amberg) = {elapsed_time:.3f} s")
            slicer.app.processEvents()

        if projectedVertices is None:
            raise RuntimeError("No registration method was executed successfully.")

        # Update the mesh with final projected vertices
        sourceMesh.vertices = projectedVertices

        # Final projected model
        projectedModelNode = self.createProjectedModelNode(sourceMesh, registeredModelName)

        # Final projected fiducials (only if landmarks are available)
        if useLandmarks and sourceLandmarksIndices is not None and sourceFiducial is not None:
            self.createProjectedFiducials(
                sourceLandmarksIndices,
                projectedVertices,
                projectedFiducialsName,
                sourceFiducials=sourceFiducial,
            )

        # Distance between final projection and target
        if (
            projectedModelNode.GetPolyData()
            and projectedModelNode.GetPolyData().GetNumberOfPoints() > 0
            and targetModel.GetPolyData()
            and targetModel.GetPolyData().GetNumberOfPoints() > 0
        ):
            distanceFilter = vtk.vtkDistancePolyDataFilter()
            distanceFilter.SetInputData(0, projectedModelNode.GetPolyData())
            distanceFilter.SetInputData(1, targetModel.GetPolyData())
            distanceFilter.SignedDistanceOff()
            distanceFilter.Update()

            distanceData = distanceFilter.GetOutput().GetPointData().GetScalars()
            distArray = vtk_np.vtk_to_numpy(distanceData)
            totalDistance = float(np.sum(distArray))
            meanDistance = totalDistance / projectedModelNode.GetPolyData().GetNumberOfPoints()

            metrics["final_total_distance"] = totalDistance
            metrics["final_mean_distance"] = meanDistance

            print("")
            print(
                f"[TrimeshReg] Final: total distance to target = {totalDistance}, "
                f"mean distance per point = {meanDistance}"
            )

        metrics["total_time"] = float(time.time() - total_start)

        return projectedModelNode, metrics

    # ------------------------------------------------------------------
    # VTK / Trimesh helpers
    # ------------------------------------------------------------------
    def vtkToTrimesh(self, vtkMesh):
        # Vertices
        vertices = vtk_np.vtk_to_numpy(vtkMesh.GetPoints().GetData())

        # Faces (triangles): polys array is [npts, id0, id1, id2, npts, id0, ...]
        vtkFaces = vtk_np.vtk_to_numpy(vtkMesh.GetPolys().GetData())
        if vtkFaces.size == 0:
            faces = np.zeros((0, 3), dtype=np.int64)
        else:
            faces = vtkFaces.reshape(-1, 4)[:, 1:]

        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def fiducialsToArray(self, fiducialNode):
        numFiducials = fiducialNode.GetNumberOfControlPoints()
        points = np.zeros((numFiducials, 3), dtype=float)
        for i in range(numFiducials):
            coord = [0.0, 0.0, 0.0]
            fiducialNode.GetNthControlPointPosition(i, coord)
            points[i] = np.array(coord)
        return points

    def createProjectedModelNode(self, mesh, name):
        vtkMesh = vtk.vtkPolyData()

        # Points
        vtkPoints = vtk.vtkPoints()
        vtkPoints.SetData(vtk_np.numpy_to_vtk(mesh.vertices))
        vtkMesh.SetPoints(vtkPoints)

        # Faces
        vtkFaces = vtk.vtkCellArray()
        if mesh.faces.size > 0:
            faces = np.hstack(
                [np.full((mesh.faces.shape[0], 1), 3, dtype=np.int64), mesh.faces.astype(np.int64)]
            )
            vtkFaces.SetCells(
                mesh.faces.shape[0],
                vtk_np.numpy_to_vtkIdTypeArray(faces.flatten()),
            )
        vtkMesh.SetPolys(vtkFaces)

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        modelNode.SetAndObservePolyData(vtkMesh)
        modelNode.CreateDefaultDisplayNodes()

        return modelNode

    def createProjectedFiducials(self, landmarkIndices, projectedVertices, name, sourceFiducials=None):
        if landmarkIndices is None or len(landmarkIndices) == 0:
            raise RuntimeError("No landmark indices provided for projected fiducials.")

        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)

        for fiducial_index, vertex_index in enumerate(landmarkIndices):
            if vertex_index < 0 or vertex_index >= projectedVertices.shape[0]:
                raise RuntimeError(
                    f"Landmark index {vertex_index} out of bounds for projected vertices."
                )
            position = projectedVertices[vertex_index]
            fiducialNode.AddControlPoint(position.tolist(), f"{fiducial_index}")
            if sourceFiducials is not None:
                desc = sourceFiducials.GetNthControlPointDescription(fiducial_index)
                fiducialNode.SetNthControlPointDescription(fiducial_index, desc)

        return fiducialNode
