# ETSE_UV__TrimeshRegistration.py
# 3D Slicer scripted module
#
# Generic registration front-end for trimesh.registration methods:
#   - icp
#   - mesh_other
#   - nricp_amberg
#   - nricp_sumner
#   - procrustes
#
# Single-run + batch mode; logs per-mesh and global stats to a CSV in the
# output folder. Fiducials are required only for Procrustes, optional for
# NRICP methods, and not used by rigid ICP / mesh_other.

import os
import time
import csv
import sys

import numpy as np
import vtk
import vtkmodules.util.numpy_support as vtk_np
import slicer
import qt
import ctk

from slicer.ScriptedLoadableModule import *


needInstall = False
try:
    import trimesh
except ModuleNotFoundError:
    needInstall = True

if needInstall:
    progressDialog = slicer.util.createProgressDialog(
        windowTitle="Installing...",
        labelText="Installing Python packages. This may take a minute...",
        maximum=0,
    )
    slicer.app.processEvents()
    try:
        slicer.util.pip_install(["trimesh"])
    except:
        slicer.util.infoDisplay("Issue while installing the Trimesh Python packages")
        progressDialog.close()

    try:
        import trimesh
    except ModuleNotFoundError as e:
        print("Module Not found. Please restart Slicer to load packages.")


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class ETSE_UV__TrimeshRegistration(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)



        parent.title = "ETSE-UV Trimesh Registration"
        parent.categories = ["ETSE_UV"]
        parent.dependencies = []
        parent.contributors = ["UV"]
        parent.helpText = (
            "Register a SOURCE (template) mesh to a TARGET mesh using methods from\n"
            "trimesh.registration: icp, mesh_other, nricp_amberg, nricp_sumner, and procrustes.\n\n"
            "Fiducials:\n"
            "  - Procrustes: requires corresponding SOURCE and TARGET fiducials.\n"
            "  - NRICP (Amberg/Sumner): can use fiducials if provided, but they are optional.\n"
            "  - ICP / mesh_other: ignore fiducials.\n\n"
            "Batch mode allows registering one SOURCE template to many TARGETS, with logs\n"
            "saved in the output folder as CSV."
        )
        parent.acknowledgementText = (
            "Thanks to the 3D Slicer and SlicerMorph communities, and to the Trimesh authors."
        )


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class ETSE_UV__TrimeshRegistrationWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()

        self.logic = ETSE_UV__TrimeshRegistrationLogic()

        # ---------------- Main form layout ----------------
        form = qt.QFormLayout()
        self.layout.addLayout(form)

        # Source model (template)
        self.sourceSelector = slicer.qMRMLNodeComboBox()
        self.sourceSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.sourceSelector.selectNodeUponCreation = True
        self.sourceSelector.addEnabled = False
        self.sourceSelector.removeEnabled = False
        self.sourceSelector.noneEnabled = False
        self.sourceSelector.setMRMLScene(slicer.mrmlScene)
        self.sourceSelector.setToolTip("Template mesh that will be deformed (SOURCE).")
        form.addRow("Source model:", self.sourceSelector)

        # Source fiducials (template landmarks)
        self.sourceFiducialSelector = slicer.qMRMLNodeComboBox()
        self.sourceFiducialSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.sourceFiducialSelector.selectNodeUponCreation = True
        self.sourceFiducialSelector.addEnabled = False
        self.sourceFiducialSelector.removeEnabled = False
        self.sourceFiducialSelector.noneEnabled = True
        self.sourceFiducialSelector.setMRMLScene(slicer.mrmlScene)
        self.sourceFiducialSelector.setToolTip(
            "Fiducial set on the SOURCE model. Required only for Procrustes; "
            "used as landmarks for NRICP methods if available."
        )
        form.addRow("Source fiducials:", self.sourceFiducialSelector)

        # Target model
        self.targetSelector = slicer.qMRMLNodeComboBox()
        self.targetSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.targetSelector.selectNodeUponCreation = True
        self.targetSelector.addEnabled = False
        self.targetSelector.removeEnabled = False
        self.targetSelector.noneEnabled = False
        self.targetSelector.setMRMLScene(slicer.mrmlScene)
        self.targetSelector.setToolTip(
            "Target mesh. The SOURCE model will be registered to match this shape."
        )
        form.addRow("Target model:", self.targetSelector)

        # Target fiducials
        self.targetFiducialSelector = slicer.qMRMLNodeComboBox()
        self.targetFiducialSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.targetFiducialSelector.selectNodeUponCreation = True
        self.targetFiducialSelector.addEnabled = False
        self.targetFiducialSelector.removeEnabled = False
        self.targetFiducialSelector.noneEnabled = True
        self.targetFiducialSelector.setMRMLScene(slicer.mrmlScene)
        self.targetFiducialSelector.setToolTip(
            "Fiducial set on the TARGET model. Required only for Procrustes; "
            "for NRICP it is used if available."
        )
        form.addRow("Target fiducials:", self.targetFiducialSelector)

        # Output names
        self.registeredModelNameEdit = qt.QLineEdit()
        self.registeredModelNameEdit.placeholderText = "Use target model name"
        self.registeredModelNameEdit.setToolTip(
            "Name for the registered (deformed) model node.\n"
            "If left empty, the TARGET model name will be used."
        )
        form.addRow("Registered model name:", self.registeredModelNameEdit)

        self.projectedFiducialsNameEdit = qt.QLineEdit()
        self.projectedFiducialsNameEdit.placeholderText = "Use source fiducials name or derived"
        self.projectedFiducialsNameEdit.setToolTip(
            "Name for projected fiducials on the registered model.\n"
            "If left empty, the SOURCE fiducials name will be used (or '<target>_Fiducials' in batch)."
        )
        form.addRow("Projected fiducials name:", self.projectedFiducialsNameEdit)

        # Method selection
        self.methodCombo = qt.QComboBox()
        self.methodCombo.addItem("Rigid ICP (icp)", "icp")
        self.methodCombo.addItem("Rigid ICP with PCA init (mesh_other)", "mesh_other")
        self.methodCombo.addItem("Nonrigid NRICP (Amberg)", "nricp_amberg")
        self.methodCombo.addItem("Nonrigid NRICP (Sumner)", "nricp_sumner")
        self.methodCombo.addItem("Procrustes (landmark-based)", "procrustes")
        self.methodCombo.setToolTip(
            "Choose one registration method from trimesh.registration.\n"
            "Procrustes requires fiducials; others do not, but NRICP can optionally use them."
        )
        form.addRow("Registration method:", self.methodCombo)

        # Method help label
        self.methodHelpLabel = qt.QLabel("")
        self.methodHelpLabel.setWordWrap(True)
        form.addRow(self.methodHelpLabel)
        self.methodCombo.currentIndexChanged.connect(self.updateMethodHelp)
        self.updateMethodHelp()  # set initial help text

        # Apply button (single run)
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Run the selected registration method on the SOURCE and TARGET."
        self.layout.addWidget(self.applyButton)
        self.applyButton.clicked.connect(self.onApplyButton)

        # ---------------- Batch registration group ----------------
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

        # Target fiducials folder (optional for most methods, required for Procrustes)
        self.targetMarkupDirectoryButton = ctk.ctkDirectoryButton()
        self.targetMarkupDirectoryButton.setToolTip(
            "Folder containing TARGET fiducials (.mrk.json). Required only for Procrustes; "
            "for NRICP they are used if available."
        )
        self.targetMarkupDirectoryButton.setMaximumWidth(400)
        batchForm.addRow("Target fiducials folder:", self.targetMarkupDirectoryButton)

        # Output folder
        self.outputDirectoryButton = ctk.ctkDirectoryButton()
        self.outputDirectoryButton.setToolTip(
            "Folder to save registered models and projected fiducials.\n"
            "A CSV log with per-mesh metrics will also be written here."
        )
        self.outputDirectoryButton.setMaximumWidth(400)
        batchForm.addRow("Output folder:", self.outputDirectoryButton)

        # Batch button
        self.batchProcessButton = qt.QPushButton("Run batch registration")
        self.batchProcessButton.toolTip = (
            "For each TARGET mesh in the models folder (and its fiducials if required),\n"
            "register the SOURCE template using the selected method and save results."
        )
        batchForm.addRow(self.batchProcessButton)
        self.batchProcessButton.clicked.connect(self.onBatchProcessButtonClicked)

        self.layout.addStretch(1)

    # ------------------------------------------------------------------
    # Method help text
    # ------------------------------------------------------------------
    def updateMethodHelp(self):
        key = self.methodCombo.currentData
        if key == "icp":
            txt = (
                "icp: iterative closest point on point clouds.\n"
                "Uses vertices of SOURCE and TARGET; requires a reasonable initial alignment."
            )
        elif key == "mesh_other":
            txt = (
                "mesh_other: aligns a mesh to another mesh/points using principal axes of\n"
                "inertia as a starting point, refined with ICP."
            )
        elif key == "nricp_amberg":
            txt = (
                "nricp_amberg: non-rigid ICP (Amberg 2007). Fits SOURCE to TARGET in fewer\n"
                "steps, can create sharper edges. Landmarks are optional, used if provided."
            )
        elif key == "nricp_sumner":
            txt = (
                "nricp_sumner: non-rigid ICP (Sumner & Popovic 2004). Tends to preserve\n"
                "original shape more; landmarks are optional, used if provided."
            )
        elif key == "procrustes":
            txt = (
                "procrustes: rigid alignment using corresponding fiducial points.\n"
                "Requires SOURCE and TARGET fiducials with the same ordering."
            )
        else:
            txt = ""
        self.methodHelpLabel.setText(txt)

    # ------------------------------------------------------------------
    # Single registration
    # ------------------------------------------------------------------
    def onApplyButton(self):
        methodKey = self.methodCombo.currentData
        methodName = self.methodCombo.currentText

        sourceModel = self.sourceSelector.currentNode()
        targetModel = self.targetSelector.currentNode()
        sourceFiducial = self.sourceFiducialSelector.currentNode()
        targetFiducial = self.targetFiducialSelector.currentNode()

        if not sourceModel or not targetModel:
            slicer.util.errorDisplay("Please select both SOURCE and TARGET model nodes.")
            return

        # Procrustes requires fiducials
        if methodKey == "procrustes":
            if not sourceFiducial or not targetFiducial:
                slicer.util.errorDisplay(
                    "Procrustes requires SOURCE and TARGET fiducials with corresponding points."
                )
                return

        # Default output names
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
            projectedModelNode, metrics = self.logic.run(
                method=methodKey,
                method_name=methodName,
                sourceModel=sourceModel,
                targetModel=targetModel,
                sourceFiducial=sourceFiducial,
                targetFiducial=targetFiducial,
                registeredModelName=registeredModelName,
                projectedFiducialsName=projectedFiducialsName,
            )
        except Exception as e:
            slicer.app.restoreOverrideCursor()
            slicer.util.errorDisplay(f"Registration failed:\n{e}")
            raise
        finally:
            slicer.app.restoreOverrideCursor()

        if projectedModelNode:
            print("[TrimeshRegAny] Registration done.")
            print(f"[TrimeshRegAny] Method: {metrics.get('method')}")
            print(f"[TrimeshRegAny] Total time: {metrics.get('total_time')}")
            print(f"[TrimeshRegAny] Final mean distance: {metrics.get('final_mean_distance')}")

    # ------------------------------------------------------------------
    # Batch registration
    # ------------------------------------------------------------------
    def onBatchProcessButtonClicked(self):
        methodKey = self.methodCombo.currentData
        methodName = self.methodCombo.currentText

        sourceModel = self.sourceSelector.currentNode()
        sourceFiducial = self.sourceFiducialSelector.currentNode()

        if not sourceModel:
            slicer.util.errorDisplay("Please select a SOURCE model (template) for batch registration.")
            return

        modelDirectory = self.targetModelDirectoryButton.directory
        markupDirectory = self.targetMarkupDirectoryButton.directory
        outputDirectory = self.outputDirectoryButton.directory

        # Method requirements wrt fiducials
        method_requires_fids = (methodKey == "procrustes")

        if not modelDirectory or not os.path.isdir(modelDirectory):
            slicer.util.errorDisplay("Please select a valid TARGET models folder.")
            return
        if method_requires_fids:
            if not markupDirectory or not os.path.isdir(markupDirectory):
                slicer.util.errorDisplay(
                    "Procrustes requires a valid TARGET fiducials folder (.mrk.json)."
                )
                return
        if not outputDirectory:
            slicer.util.errorDisplay("Please select an output folder.")
            return
        if not os.path.isdir(outputDirectory):
            os.makedirs(outputDirectory, exist_ok=True)

        print("[TrimeshRegAny][Batch] Starting batch registration...")
        print(f"[TrimeshRegAny][Batch] Method: {methodName} ({methodKey})")
        print(f"[TrimeshRegAny][Batch] Template: {sourceModel.GetName()}")
        print(f"[TrimeshRegAny][Batch] Models folder:  {modelDirectory}")
        print(f"[TrimeshRegAny][Batch] Markups folder: {markupDirectory if markupDirectory else '(none)'}")
        print(f"[TrimeshRegAny][Batch] Output folder:  {outputDirectory}")

        total = 0
        success = 0
        failures = []
        results = []

        exts = (".ply", ".obj", ".vtk")

        total_times = []
        final_mean_distances = []
        costs = []

        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)

            for modelFileName in os.listdir(modelDirectory):
                if not modelFileName.lower().endswith(exts):
                    continue

                total += 1
                modelFilePath = os.path.join(modelDirectory, modelFileName)
                baseName = os.path.splitext(modelFileName)[0]

                # registered model: same name as target
                registeredModelName = baseName
                # projected fiducials: avoid name clash
                projectedFiducialsName = baseName + "_Fiducials"

                # Target fiducial file (if we have a markup folder)
                markupFilePath = None
                if markupDirectory:
                    markupFileName = baseName + ".mrk.json"
                    markupFilePath = os.path.join(markupDirectory, markupFileName)

                if method_requires_fids and (not markupFilePath or not os.path.exists(markupFilePath)):
                    msg = f"Missing TARGET fiducials for {modelFileName} (required by Procrustes)."
                    print(f"[TrimeshRegAny][Batch][Skip] {msg}")
                    failures.append((modelFileName, "markup not found"))
                    results.append(self._emptyResultRow(modelFileName, methodKey, methodName, msg))
                    continue

                print(f"[TrimeshRegAny][Batch] Processing {modelFileName} ...")
                slicer.app.processEvents()
                sys.stdout.flush()
                try:
                    # Load target model
                    targetModelNode = slicer.util.loadModel(modelFilePath)
                    if not targetModelNode:
                        msg = "Failed to load target model."
                        print(f"[TrimeshRegAny][Batch][Error] {msg}")
                        failures.append((modelFileName, msg))
                        results.append(self._emptyResultRow(modelFileName, methodKey, methodName, msg))
                        continue

                    # Load target fiducials if needed/available
                    targetFiducialNode = None
                    if markupFilePath and os.path.exists(markupFilePath):
                        targetFiducialNode = slicer.util.loadMarkups(markupFilePath)
                        if not targetFiducialNode:
                            msg = "Failed to load target fiducials."
                            print(f"[TrimeshRegAny][Batch][Error] {msg}")
                            failures.append((modelFileName, msg))
                            results.append(self._emptyResultRow(modelFileName, methodKey, methodName, msg))
                            slicer.mrmlScene.RemoveNode(targetModelNode)
                            continue
                    else:
                        # If method requires fiducials, this was already checked above.
                        targetFiducialNode = None

                    # Hide target model
                    if targetModelNode.GetDisplayNode():
                        targetModelNode.GetDisplayNode().SetVisibility(False)

                    projectedModelNode, metrics = self.logic.run(
                        method=methodKey,
                        method_name=methodName,
                        sourceModel=sourceModel,
                        targetModel=targetModelNode,
                        sourceFiducial=sourceFiducial,
                        targetFiducial=targetFiducialNode,
                        registeredModelName=registeredModelName,
                        projectedFiducialsName=projectedFiducialsName,
                    )

                    # Clean target nodes
                    slicer.mrmlScene.RemoveNode(targetModelNode)
                    if targetFiducialNode is not None:
                        slicer.mrmlScene.RemoveNode(targetFiducialNode)

                    if projectedModelNode is None:
                        msg = "Logic returned no projected model."
                        print(f"[TrimeshRegAny][Batch][Error] {msg}")
                        failures.append((modelFileName, msg))
                        results.append(self._emptyResultRow(modelFileName, methodKey, methodName, msg))
                        continue

                    # Save registered model
                    registeredModelPath = os.path.join(outputDirectory, registeredModelName + ".vtk")
                    if not slicer.util.saveNode(projectedModelNode, registeredModelPath):
                        msg = "Could not save registered model."
                        print(f"[TrimeshRegAny][Batch][Error] {msg}")
                        failures.append((modelFileName, msg))
                        results.append(self._metricsRowFromMetrics(modelFileName, metrics, status="fail", error=msg))
                        continue

                    # Save projected fiducials if they exist
                    try:
                        projectedFiducialsNode = slicer.util.getNode(projectedFiducialsName)
                        projectedFiducialsPath = os.path.join(outputDirectory, projectedFiducialsName + ".json")
                        slicer.util.saveNode(projectedFiducialsNode, projectedFiducialsPath)
                    except Exception:
                        # If no fiducials were created, that's OK for methods without landmarks
                        pass

                    success += 1
                    print(f"[TrimeshRegAny][Batch] Saved results for {modelFileName}")

                    total_times.append(metrics.get("total_time") or 0.0)
                    if metrics.get("final_mean_distance") is not None:
                        final_mean_distances.append(metrics["final_mean_distance"])
                    if metrics.get("cost") is not None:
                        costs.append(metrics["cost"])

                    results.append(self._metricsRowFromMetrics(modelFileName, metrics, status="ok", error=""))

                except Exception as e:
                    msg = str(e)
                    print(f"[TrimeshRegAny][Batch][Exception] {modelFileName}: {msg}")
                    failures.append((modelFileName, msg))
                    results.append(self._emptyResultRow(modelFileName, methodKey, methodName, msg))
                    # Try to clean up known nodes
                    for nodeName in [registeredModelName, projectedFiducialsName]:
                        try:
                            node = slicer.util.getNode(nodeName)
                            slicer.mrmlScene.RemoveNode(node)
                        except Exception:
                            pass
                    continue

        finally:
            slicer.app.restoreOverrideCursor()

        # Global stats
        total_time_all = sum(total_times) if total_times else 0.0
        mean_time = (total_time_all / len(total_times)) if total_times else 0.0
        mean_final_mean_distance = (
            sum(final_mean_distances) / len(final_mean_distances)
            if final_mean_distances
            else 0.0
        )
        mean_cost = (sum(costs) / len(costs)) if costs else 0.0

        # Write CSV log
        csv_path = os.path.join(outputDirectory, "batch_log_trimesh_registration.csv")
        try:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)

                # Configuration header
                writer.writerow(["CONFIGURATION"])
                writer.writerow(["method_key", methodKey])
                writer.writerow(["method_name", methodName])
                writer.writerow(["requires_fiducials", method_requires_fids])
                writer.writerow([])
                # Per-mesh header
                writer.writerow([
                    "filename",
                    "status",
                    "error",
                    "method",
                    "cost",
                    "total_time",
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
                        row["method"],
                        row["cost"],
                        row["total_time"],
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
                writer.writerow(["mean_cost", mean_cost])

            print(f"[TrimeshRegAny][Batch] Wrote log CSV: {csv_path}")
        except Exception as e:
            print(f"[TrimeshRegAny][Batch] Could not write log CSV: {e}")

        summary = (
            f"Batch registration finished.\n\n"
            f"Method: {methodName} ({methodKey})\n\n"
            f"Total target models found: {total}\n"
            f"Successfully processed:    {success}\n"
            f"Failed or skipped:         {len(failures)}\n\n"
            f"Total time (all meshes):   {total_time_all:.3f} s\n"
            f"Mean time per mesh:        {mean_time:.3f} s\n"
            f"Mean final mean distance:  {mean_final_mean_distance:.6g}\n"
            f"Mean cost (if defined):    {mean_cost:.6g}"
        )
        slicer.util.infoDisplay(summary, "ETSE_UV__TrimeshRegistration batch")

        if failures:
            print("[TrimeshRegAny][Batch] Failures:")
            for fname, reason in failures:
                print(f"  - {fname}: {reason}")

    # Helpers for batch logging
    def _emptyResultRow(self, filename, methodKey, methodName, error):
        return {
            "filename": filename,
            "status": "fail",
            "error": error,
            "method": f"{methodName} ({methodKey})",
            "cost": "",
            "total_time": "",
            "final_total_distance": "",
            "final_mean_distance": "",
            "num_source_points": "",
            "num_target_points": "",
            "num_source_landmarks": "",
            "num_target_landmarks": "",
        }

    def _metricsRowFromMetrics(self, filename, metrics, status, error):
        return {
            "filename": filename,
            "status": status,
            "error": error,
            "method": metrics.get("method"),
            "cost": metrics.get("cost") if metrics.get("cost") is not None else "",
            "total_time": metrics.get("total_time") if metrics.get("total_time") is not None else "",
            "final_total_distance": metrics.get("final_total_distance") if metrics.get("final_total_distance") is not None else "",
            "final_mean_distance": metrics.get("final_mean_distance") if metrics.get("final_mean_distance") is not None else "",
            "num_source_points": metrics.get("num_source_points") if metrics.get("num_source_points") is not None else "",
            "num_target_points": metrics.get("num_target_points") if metrics.get("num_target_points") is not None else "",
            "num_source_landmarks": metrics.get("num_source_landmarks") if metrics.get("num_source_landmarks") is not None else "",
            "num_target_landmarks": metrics.get("num_target_landmarks") if metrics.get("num_target_landmarks") is not None else "",
        }


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class ETSE_UV__TrimeshRegistrationLogic(ScriptedLoadableModuleLogic):
    def run(
        self,
        method,
        method_name,
        sourceModel,
        targetModel,
        sourceFiducial=None,
        targetFiducial=None,
        registeredModelName="RegisteredModel",
        projectedFiducialsName="ProjectedFiducials",
    ):
        """
        Generic dispatcher for trimesh.registration methods.
        Returns (projectedModelNode, metrics_dict).
        """
        if not sourceModel or not targetModel:
            raise RuntimeError("Source and Target models are required.")

        total_start = time.time()

        src_pd = sourceModel.GetPolyData()
        tgt_pd = targetModel.GetPolyData()
        num_src_pts = src_pd.GetNumberOfPoints()
        num_tgt_pts = tgt_pd.GetNumberOfPoints()

        sourceMesh = self.vtkToTrimesh(src_pd)
        targetMesh = self.vtkToTrimesh(tgt_pd)

        # Fiducials
        num_src_fids = sourceFiducial.GetNumberOfControlPoints() if sourceFiducial else 0
        num_tgt_fids = targetFiducial.GetNumberOfControlPoints() if targetFiducial else 0

        # Procrustes requires fiducials
        if method == "procrustes":
            if not sourceFiducial or not targetFiducial:
                raise RuntimeError("Procrustes requires SOURCE and TARGET fiducials.")
            if num_src_fids == 0 or num_tgt_fids == 0:
                raise RuntimeError("Procrustes: fiducials must not be empty.")
            if num_src_fids != num_tgt_fids:
                raise RuntimeError("Procrustes: SOURCE and TARGET must have the same number of fiducials.")

        # Prepare metrics dict
        metrics = {
            "method": f"{method_name} ({method})",
            "cost": None,
            "total_time": None,
            "final_total_distance": None,
            "final_mean_distance": None,
            "num_source_points": int(num_src_pts),
            "num_target_points": int(num_tgt_pts),
            "num_source_landmarks": int(num_src_fids),
            "num_target_landmarks": int(num_tgt_fids),
        }

        print("[TrimeshRegAny] Running registration...")
        print(f"[TrimeshRegAny] Method: {metrics['method']}")
        print(f"[TrimeshRegAny] Source model: {sourceModel.GetName()} ({num_src_pts} points)")
        print(f"[TrimeshRegAny] Target model: {targetModel.GetName()} ({num_tgt_pts} points)")
        print(f"[TrimeshRegAny] Source fiducials: {num_src_fids}")
        print(f"[TrimeshRegAny] Target fiducials: {num_tgt_fids}")
        print("")

        projectedVertices = None
        transformMatrix = None  # 4x4, for rigid methods

        # ---------- Landmarks for NRICP if available ----------
        sourceLandmarks = None
        targetLandmarks = None
        sourceLandmarkIndices = None
        if (method in ("nricp_amberg", "nricp_sumner")) and sourceFiducial and targetFiducial:
            sourceLandmarks = self.fiducialsToArray(sourceFiducial)
            targetLandmarks = self.fiducialsToArray(targetFiducial)
            # Map each source landmark to nearest vertex index
            sourceLandmarkIndices = np.array(
                [
                    np.argmin(np.linalg.norm(sourceMesh.vertices - landmark, axis=1))
                    for landmark in sourceLandmarks
                ],
                dtype=np.int64,
            )

        # Keep a copy of original vertices for nonrigid landmark projection
        originalVertices = sourceMesh.vertices.copy()

        # ---------- Method dispatch ----------
        if method == "icp":
            # Use all vertices as point clouds
            a = sourceMesh.vertices
            b = targetMesh.vertices
            start = time.time()
            matrix, transformed, cost = trimesh.registration.icp(a, b)
            elapsed = time.time() - start
            metrics["cost"] = float(cost)
            transformMatrix = matrix
            projectedVertices = transformed
            print(f"[TrimeshRegAny] icp cost: {cost}, time: {elapsed:.3f} s")

        elif method == "mesh_other":
            start = time.time()
            matrix, cost = trimesh.registration.mesh_other(
                mesh=sourceMesh, other=targetMesh
            )
            elapsed = time.time() - start
            metrics["cost"] = float(cost)
            transformMatrix = matrix
            # Apply transform to vertices
            projectedVertices = self.apply4x4ToPoints(matrix, sourceMesh.vertices)
            print(f"[TrimeshRegAny] mesh_other cost: {cost}, time: {elapsed:.3f} s")

        elif method == "nricp_amberg":
            start = time.time()
            projectedVertices = trimesh.registration.nricp_amberg(
                source_mesh=sourceMesh,
                target_geometry=targetMesh,
                source_landmarks=sourceLandmarkIndices,
                target_positions=targetLandmarks,
                # use trimesh defaults for steps, eps, gamma, etc
            )
            elapsed = time.time() - start
            print(f"[TrimeshRegAny] nricp_amberg time: {elapsed:.3f} s")

        elif method == "nricp_sumner":
            start = time.time()
            projectedVertices = trimesh.registration.nricp_sumner(
                source_mesh=sourceMesh,
                target_geometry=targetMesh,
                source_landmarks=sourceLandmarkIndices,
                target_positions=targetLandmarks,
                # use trimesh defaults for steps, distance_threshold, etc
            )
            elapsed = time.time() - start
            print(f"[TrimeshRegAny] nricp_sumner time: {elapsed:.3f} s")

        elif method == "procrustes":
            # Fiducials define the transform
            srcF = self.fiducialsToArray(sourceFiducial)
            tgtF = self.fiducialsToArray(targetFiducial)
            start = time.time()
            matrix, transformedF, cost = trimesh.registration.procrustes(
                a=srcF,
                b=tgtF,
                # default reflection=True, translation=True, scale=True, return_cost=True
            )
            elapsed = time.time() - start
            metrics["cost"] = float(cost)
            transformMatrix = matrix
            projectedVertices = self.apply4x4ToPoints(matrix, sourceMesh.vertices)
            print(f"[TrimeshRegAny] procrustes cost: {cost}, time: {elapsed:.3f} s")

        else:
            raise RuntimeError(f"Unknown method key: {method}")

        if projectedVertices is None:
            raise RuntimeError("Registration did not produce projected vertices.")

        # Update the mesh with the new vertex positions
        sourceMesh.vertices = projectedVertices

        # Create registered model node
        projectedModelNode = self.createProjectedModelNode(sourceMesh, registeredModelName)

        # Create projected fiducials if we can
        if sourceFiducial:
            try:
                if transformMatrix is not None:
                    # Rigid methods: apply transform to fiducial coordinates
                    srcF = self.fiducialsToArray(sourceFiducial)
                    projF = self.apply4x4ToPoints(transformMatrix, srcF)
                else:
                    # Nonrigid methods: use nearest vertex indices from original mesh
                    srcF = self.fiducialsToArray(sourceFiducial)
                    # Map once from original vertices
                    landmarkIndices = np.array(
                        [
                            np.argmin(np.linalg.norm(originalVertices - landmark, axis=1))
                            for landmark in srcF
                        ],
                        dtype=np.int64,
                    )
                    projF = projectedVertices[landmarkIndices]

                self.createProjectedFiducials(
                    projectedPositions=projF,
                    name=projectedFiducialsName,
                    sourceFiducials=sourceFiducial,
                )
            except Exception as e:
                print(f"[TrimeshRegAny] Warning: could not create projected fiducials: {e}")

        # Compute distance metrics between registered model and target
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
                f"[TrimeshRegAny] Final: total distance to target = {totalDistance}, "
                f"mean distance per point = {meanDistance}"
            )

        metrics["total_time"] = float(time.time() - total_start)

        return projectedModelNode, metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def vtkToTrimesh(self, vtkMesh):
        # Vertices
        vertices = vtk_np.vtk_to_numpy(vtkMesh.GetPoints().GetData())

        # Faces (triangles) if present
        vtkFaces = vtk_np.vtk_to_numpy(vtkMesh.GetPolys().GetData())
        if vtkFaces.size == 0:
            faces = np.zeros((0, 3), dtype=np.int64)
        else:
            # polys: [npts, id0, id1, id2, npts, ...] for triangles
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

    def apply4x4ToPoints(self, matrix4x4, points):
        """
        Apply a 4x4 transform matrix to an (N,3) array of 3D points.
        """
        pts = np.asarray(points, dtype=float)
        ones = np.ones((pts.shape[0], 1), dtype=float)
        homo = np.concatenate([pts, ones], axis=1)  # (N,4)
        transformed = homo @ matrix4x4.T
        return transformed[:, :3]

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

    def createProjectedFiducials(self, projectedPositions, name, sourceFiducials=None):
        """
        Create a new fiducial node at projectedPositions (N,3).
        If sourceFiducials is given, copy descriptions.
        """
        pts = np.asarray(projectedPositions, dtype=float)
        num = pts.shape[0]

        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)

        for i in range(num):
            pos = pts[i].tolist()
            fiducialNode.AddControlPoint(pos, f"{i}")
            if sourceFiducials is not None and i < sourceFiducials.GetNumberOfControlPoints():
                desc = sourceFiducials.GetNthControlPointDescription(i)
                fiducialNode.SetNthControlPointDescription(i, desc)

        return fiducialNode
