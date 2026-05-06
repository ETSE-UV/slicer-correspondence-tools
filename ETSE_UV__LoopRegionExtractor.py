# ETSE_UV__LoopRegionExtractor.py
# 3D Slicer scripted module
#
# Loop-based cutter focused on ear (pinna) meshes, but usable on any
# single connected surface mesh. Given two closed fiducial loops on
# the surface (e.g. canal and otobasion) and a seed point on the tragus,
# it splits the mesh into:
#   - Pinna region (component containing the tragus seed)
#   - Base / complement
#
# It can also save the extracted region topology (vertex indices and
# exact faces) and re-apply it to other registered meshes, including
# in batch mode.

import os
import numpy as np
import slicer
import vtk
import qt
import ctk
import collections
from slicer.ScriptedLoadableModule import *
from Resources.ETSE_UV__Dependencies import ensure_packages

ensure_packages(
    [("scipy", "scipy")],
    interactive=False,
    module_name="ETSE-UV Loop Region Extractor",
)

# ------------------------------------------------------------
# Module
# ------------------------------------------------------------
class ETSE_UV__LoopRegionExtractor(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Loop Region Extractor"
        parent.categories = ["ETSE_UV"]
        parent.contributors = ["ETSE-UV"]
        parent.helpText = """
                            <p>Loop-based surface cutter designed for ear/pinna meshes, but usable on any
                            single connected surface mesh.</p>

                            <p><b>Typical ear workflow:</b></p>
                            <ol>
                              <li>Draw a closed fiducial loop around the ear canal.</li>
                              <li>Draw another closed fiducial loop around the otobasion region.</li>
                              <li>Specify which 1-based fiducial indices belong to each loop.</li>
                              <li>Select the tragus seed landmark used to identify the pinna side.</li>
                              <li>Run the cut.</li>
                            </ol>

                            <p><b>Outputs:</b></p>
                            <ul>
                              <li>Pinna model: the connected component containing the tragus seed.</li>
                              <li>Base model: the remaining surface.</li>
                            </ul>

                            <p>The module can also save extracted pinna topology and re-apply it to registered
                            meshes, either one-by-one or in batch.</p>
                            """
        parent.acknowledgementText = (
            "Developed by Juan Antonio De Rus Arance at the Escola Tècnica Superior "
            "d'Enginyeria (ETSE-UV), Universitat de València, in the context of the "
            "Signal Processing & Acoustic Technology (SPAT) research group. "
            "Thanks to the 3D Slicer, SlicerMorph, VTK, NumPy, SciPy, Trimesh, and related "
            "open-source communities."
        )


# ------------------------------------------------------------
# Widget
# ------------------------------------------------------------
class ETSE_UV__LoopRegionExtractorWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # create one logic instance for the widget lifecycle
        self.logic = ETSE_UV__LoopRegionExtractorLogic()

        # ----------------------------------------------------
        # Section: Loop-based cut on current mesh
        # ----------------------------------------------------
        cutBox = ctk.ctkCollapsibleButton()
        cutBox.text = "Cut current mesh using loops"
        self.layout.addWidget(cutBox)
        cutLayout = qt.QVBoxLayout(cutBox)

        form = qt.QFormLayout()
        cutLayout.addLayout(form)

        # Model selector
        self.modelSelector = slicer.qMRMLNodeComboBox()
        self.modelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.modelSelector.selectNodeUponCreation = True
        self.modelSelector.addEnabled = False
        self.modelSelector.removeEnabled = False
        self.modelSelector.noneEnabled = False
        self.modelSelector.setMRMLScene(slicer.mrmlScene)
        self.modelSelector.setToolTip("Pick the surface mesh to cut (e.g. ear mold + base).")
        form.addRow("Input mesh:", self.modelSelector)

        # Fiducial selector
        self.fiducialSelector = slicer.qMRMLNodeComboBox()
        self.fiducialSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.fiducialSelector.selectNodeUponCreation = True
        self.fiducialSelector.addEnabled = False
        self.fiducialSelector.removeEnabled = False
        self.fiducialSelector.noneEnabled = False
        self.fiducialSelector.setMRMLScene(slicer.mrmlScene)
        self.fiducialSelector.setToolTip(
            "Pick the fiducial node containing all loop points.\n"
            "Canal and otobasion loops are defined by index ranges in this node."
        )
        form.addRow("Loop markups:", self.fiducialSelector)

        # Canal indices (1–4)
        canalBox = qt.QGroupBox("Canal loop indices (1-based)")
        canalLayout = qt.QHBoxLayout(canalBox)
        self.canalLineEdit = qt.QLineEdit("1-4")
        self.canalLineEdit.setToolTip(
            "Indices (1-based) of fiducials that belong to the canal loop.\n"
            "Supports ranges like '1-4, 7, 10-12'."
        )
        canalLayout.addWidget(self.canalLineEdit)
        cutLayout.addWidget(canalBox)

        # Otobasion indices (246–261)
        otoBox = qt.QGroupBox("Otobasion loop indices (1-based)")
        otoLayout = qt.QHBoxLayout(otoBox)
        self.otoLineEdit = qt.QLineEdit("246-261")
        self.otoLineEdit.setToolTip(
            "Indices (1-based) of fiducials that belong to the otobasion loop."
        )
        otoLayout.addWidget(self.otoLineEdit)
        cutLayout.addWidget(otoBox)

        # Ear seed index (tragus, usually 30)
        earBox = qt.QGroupBox("Seed landmark (tragus index, 1-based)")
        earLayout = qt.QHBoxLayout(earBox)
        self.earSpinBox = qt.QSpinBox()
        self.earSpinBox.setRange(1, 9999)
        self.earSpinBox.setValue(31)  # default 31 in 1-based -> index 30
        self.earSpinBox.setToolTip(
            "Index (1-based) of the tragus landmark in the same fiducial node.\n"
            "The connected component containing this point is labeled as the Pinna."
        )
        earLayout.addWidget(self.earSpinBox)
        cutLayout.addWidget(earBox)

        # Options
        self.cleanCheck = qt.QCheckBox("Clean input meshes (vtkCleanPolyData)")
        self.cleanCheck.checked = False
        self.cleanCheck.setToolTip("If enabled, run vtkCleanPolyData on the input mesh before processing.")
        cutLayout.addWidget(self.cleanCheck)

        self.showLoopsCheck = qt.QCheckBox("Create visual loop polylines")
        self.showLoopsCheck.checked = True
        self.showLoopsCheck.setToolTip("If enabled, creates model nodes that show the canal and otobasion loops.")
        cutLayout.addWidget(self.showLoopsCheck)

        self.preserveConnCheck = qt.QCheckBox(
            "Preserve original vertex indices & connectivity when re-applying (no cleaning)"
        )
        self.preserveConnCheck.checked = True
        self.preserveConnCheck.setToolTip(
            "Affects topology re-application tools below. When enabled, tries to keep original vertex\n"
            "indexing when reconstructing the pinna on other meshes."
        )
        cutLayout.addWidget(self.preserveConnCheck)

        self.debugCheck = qt.QCheckBox("Debug (verbose)")
        self.debugCheck.checked = False
        self.debugCheck.setToolTip("Print additional internal info to the Python console.")
        cutLayout.addWidget(self.debugCheck)

        # Apply
        self.applyButton = qt.QPushButton("Cut mesh by loops")
        self.applyButton.toolTip = "Compute geodesic loops from fiducials and split mesh into Pinna and Base."
        self.applyButton.connect('clicked(bool)', self.onApplyButton)
        cutLayout.addWidget(self.applyButton)

        # ----------------------------------------------------
        # Section: Save / load / apply stored topology
        # ----------------------------------------------------
        topoBox = ctk.ctkCollapsibleButton()
        topoBox.text = "Save, load and apply stored pinna topology"
        self.layout.addWidget(topoBox)
        topoLayout = qt.QVBoxLayout(topoBox)

        # stored indices label
        self.storedLabel = qt.QLabel("Stored ear indices: none")
        topoLayout.addWidget(self.storedLabel)

        # Target model to apply saved indices
        self.targetModelSelector = slicer.qMRMLNodeComboBox()
        self.targetModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.targetModelSelector.selectNodeUponCreation = True
        self.targetModelSelector.addEnabled = False
        self.targetModelSelector.removeEnabled = False
        self.targetModelSelector.noneEnabled = False
        self.targetModelSelector.setMRMLScene(slicer.mrmlScene)
        self.targetModelSelector.setToolTip(
            "Pick a target model to apply the stored pinna topology.\n"
            "Target meshes should have the same topology / registration as the source used for the cut."
        )
        topoLayout.addWidget(self.targetModelSelector)

        # Buttons row
        buttonsRow = qt.QHBoxLayout()
        self.saveIndicesButton = qt.QPushButton("Save pinna topology (.npz)")
        self.loadIndicesButton = qt.QPushButton("Load pinna topology (.npz)")
        self.applyIndicesButton = qt.QPushButton("Apply topology to target model")
        buttonsRow.addWidget(self.saveIndicesButton)
        buttonsRow.addWidget(self.loadIndicesButton)
        buttonsRow.addWidget(self.applyIndicesButton)
        topoLayout.addLayout(buttonsRow)

        self.saveIndicesButton.toolTip = "Save vertex indices + connectivity (faces) of the extracted pinna region to a .npz file."
        self.loadIndicesButton.toolTip = "Load previously saved pinna topology (.npz) with vertex indices and faces."
        self.applyIndicesButton.toolTip = "Rebuild the pinna region on the selected target model using the stored topology."

        # connect buttons
        self.saveIndicesButton.connect('clicked(bool)', self.onSaveIndices)
        self.loadIndicesButton.connect('clicked(bool)', self.onLoadIndices)
        self.applyIndicesButton.connect('clicked(bool)', self.onApplyIndicesToTarget)

        # initialize stored label from logic (if any)
        self._updateStoredLabel(self.logic)

        # --- Batch apply to folder ---
        self.batchButton = qt.QPushButton("Batch apply stored topology to folder…")
        self.batchButton.toolTip = (
            "Apply the stored pinna topology to all models in a folder and save the results to an output folder."
        )
        topoLayout.addWidget(self.batchButton)
        self.batchButton.connect('clicked(bool)', self.onBatchApplyToFolder)

        self.layout.addStretch(1)

    # ----------------------------------------------------
    # Helpers (UI)
    # ----------------------------------------------------
    def parseIndexSpec(self, text):
        # e.g. "1-4,7,10-12" -> [0,1,2,3,6,9,10,11] (0-based)
        out = []
        for chunk in text.replace(' ', '').split(','):
            if not chunk:
                continue
            if '-' in chunk:
                a, b = chunk.split('-')
                a, b = int(a), int(b)
                step = 1 if b >= a else -1
                out.extend(list(range(a, b + step, step)))
            else:
                out.append(int(chunk))
        # convert to 0-based for VTK access later
        return [i-1 for i in out]

    def ensureDisplayNode(self, node, color=(1.0, 1.0, 1.0), opacity=1.0, pointSize=None, lineWidth=None):
        if not node.GetDisplayNode():
            node.CreateDefaultDisplayNodes()
        dn = node.GetDisplayNode()
        if color is not None:
            dn.SetColor(*color)
        dn.SetOpacity(opacity)
        if hasattr(dn, "SetPointSize") and pointSize is not None:
            dn.SetPointSize(pointSize)
        if hasattr(dn, "SetLineWidth") and lineWidth is not None:
            dn.SetLineWidth(lineWidth)
        dn.SetVisibility(True)
        return dn

    # ----------------------------------------------------
    # Actions
    # ----------------------------------------------------
    def onApplyButton(self):
        meshNode = self.modelSelector.currentNode()
        fidNode  = self.fiducialSelector.currentNode()

        if not meshNode or not fidNode:
            slicer.util.errorDisplay("Please select both a mesh and a fiducial node.")
            return

        canalIdx = self.parseIndexSpec(self.canalLineEdit.text)
        otoIdx   = self.parseIndexSpec(self.otoLineEdit.text)

        if max(canalIdx + otoIdx) >= fidNode.GetNumberOfControlPoints():
            slicer.util.errorDisplay("Fiducial node does not contain that many points.")
            return

        tragusIdx = self.earSpinBox.value - 1

        res = self.logic.cutByLoops(
            meshNode,
            fidNode,
            canalIdx,
            otoIdx,
            tragus_index=tragusIdx,
            createVisualLoops=self.showLoopsCheck.checked,
            debug=self.debugCheck.checked,
            clean=self.cleanCheck.checked
        )

        # update stored label from logic (cutByLoops stores indices in self.logic)
        self._updateStoredLabel(self.logic)

        if res.get("pinnaNode"):
            self.ensureDisplayNode(res["pinnaNode"], color=(0.2, 0.8, 0.2), opacity=1.0)
        if res.get("baseNode"):
            self.ensureDisplayNode(res["baseNode"], color=(0.8, 0.2, 0.2), opacity=0.6)
        slicer.util.infoDisplay("Loop-based cut finished.", "ETSE_UV__LoopRegionExtractor")

    def _updateStoredLabel(self, logic):
        arr = logic.get_last_indices()
        if arr is None or len(arr) == 0:
            self.storedLabel.setText("Stored ear indices: none")
        else:
            self.storedLabel.setText(f"Stored ear indices: {len(arr)}")

    def onSaveIndices(self):
        # We save BOTH vertex indices and connectivity (faces) in one .npz file
        if not self.logic.get_last_indices():
            slicer.util.errorDisplay("No stored pinna indices to save. Run a cut first.")
            return
        filePath = qt.QFileDialog.getSaveFileName(self.parent, "Save pinna topology", "", "NPZ files (*.npz)")
        if filePath:
            if not filePath.lower().endswith(".npz"):
                filePath += ".npz"
            try:
                self.logic.save_indices(filePath)
                vi = self.logic.get_last_indices() or []
                ci = self.logic.get_last_cell_ids() or []
                slicer.util.infoDisplay(
                    f"Saved topology to:\n{filePath}\n"
                    f"- vertex indices: {len(vi)}\n"
                    f"- stored cells: {len(ci)}"
                )
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to save: {e}")

    def onLoadIndices(self):
        filePath = qt.QFileDialog.getOpenFileName(self.parent, "Load pinna topology", "", "NPZ files (*.npz)")
        if filePath:
            try:
                self.logic.load_indices(filePath)
                self._updateStoredLabel(self.logic)
                vi = self.logic.get_last_indices() or []
                ci = self.logic.get_last_cell_ids() or []
                fv = self.logic.get_last_faces_vtk()
                fv_count = 0 if fv is None else int(np.count_nonzero(fv[::1] >= 0))  # informative only
                slicer.util.infoDisplay(
                    f"Loaded topology from:\n{filePath}\n"
                    f"- vertex indices: {len(vi)}\n"
                    f"- stored cells: {len(ci)}\n"
                    f"- faces_vtk present: {'yes' if fv is not None else 'no'}"
                )
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to load: {e}")

    def onApplyIndicesToTarget(self):
        target = self.targetModelSelector.currentNode()
        if not target:
            slicer.util.errorDisplay("Please select a target model.")
            return
        if not self.logic.get_last_indices():
            slicer.util.errorDisplay("No stored pinna topology. Run a cut or load a topology first.")
            return

        outname = target.GetName() + "_EarFromIndices"

        try:
            # 1) First: if we have faces_vtk + vertex_indices saved, use the exact topology path
            outNode = self.logic.apply_saved_topology_to_model(
                targetModelNode=target,
                output_name=outname,
                debug=self.debugCheck.checked
            )

            # 2) Fallbacks if needed
            if outNode is None and self.logic.get_last_cell_ids():
                outNode = self.logic.apply_stored_cells_to_model(
                    targetModelNode=target,
                    output_name=outname,
                    debug=self.debugCheck.checked,
                    clean=self.cleanCheck.checked
                )
            if outNode is None:
                outNode = self.logic.apply_indices_to_model(
                    targetModelNode=target,
                    vertex_indices=self.logic.get_last_indices(),
                    include_partial=False,
                    output_name=outname,
                    debug=self.debugCheck.checked,
                    clean=self.cleanCheck.checked
                )

            if outNode:
                self.ensureDisplayNode(outNode, color=(0.2, 0.8, 0.2), opacity=1.0)
                slicer.util.infoDisplay(f"Created node: {outNode.GetName()}")
            else:
                slicer.util.errorDisplay("Failed to apply stored topology to target model.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to apply indices: {e}")

    def onBatchApplyToFolder(self):
        # Require saved topology
        if not self.logic.get_last_indices():
            slicer.util.errorDisplay("No stored pinna topology. Run a cut or load a topology first.")
            return

        inputDir = qt.QFileDialog.getExistingDirectory(self.parent, "Select input folder with models")
        if not inputDir:
            return
        outputDir = qt.QFileDialog.getExistingDirectory(self.parent, "Select output folder for results")
        if not outputDir:
            return

        exts = {".vtk"}
        files = [f for f in os.listdir(inputDir) if os.path.splitext(f)[1].lower() in exts]
        if not files:
            slicer.util.errorDisplay("No model files found in the selected input folder.")
            return

        debug = bool(self.debugCheck.checked)
        clean = bool(self.cleanCheck.checked)

        processed, saved = 0, 0
        for fname in files:
            try:
                inPath = os.path.join(inputDir, fname)
                base = os.path.splitext(os.path.basename(inPath))[0]

                ok, inNode = slicer.util.loadModel(inPath, returnNode=True)
                if not ok or inNode is None:
                    if debug:
                        print(f"[PinnaLoopBatch] Could not load: {inPath}")
                    continue

                outName = base
                outNode = self.logic.apply_saved_topology_to_model(
                    targetModelNode=inNode,
                    output_name=outName,
                    debug=debug
                )

                if outNode is None and self.logic.get_last_cell_ids():
                    outNode = self.logic.apply_stored_cells_to_model(
                        targetModelNode=inNode,
                        output_name=outName,
                        debug=debug,
                        clean=clean
                    )
                if outNode is None:
                    outNode = self.logic.apply_indices_to_model(
                        targetModelNode=inNode,
                        vertex_indices=self.logic.get_last_indices(),
                        include_partial=False,
                        output_name=outName,
                        debug=debug,
                        clean=clean
                    )

                if outNode:
                    outPath = os.path.join(outputDir, outName + ".vtk")
                    ok = slicer.util.saveNode(outNode, outPath)
                    if ok:
                        saved += 1
                        if debug:
                            print(f"[PinnaLoopBatch] Saved: {outPath}")
                    else:
                        if debug:
                            print(f"[PinnaLoopBatch] Failed to save: {outPath}")

                    slicer.mrmlScene.RemoveNode(outNode)

                slicer.mrmlScene.RemoveNode(inNode)
                processed += 1
            except Exception as e:
                if debug:
                    print(f"[PinnaLoopBatch] Error on {fname}: {e}")

        slicer.util.infoDisplay(
            f"Batch finished.\nProcessed: {processed}\nSaved: {saved}\nOutput: {outputDir}",
            "ETSE_UV__LoopRegionExtractor"
        )


# ------------------------------------------------------------
# Logic
# ------------------------------------------------------------
class ETSE_UV__LoopRegionExtractorLogic(ScriptedLoadableModuleLogic):

    # ---------- Stored indices handling ----------
    def __init__(self):
        try:
            super(ETSE_UV__LoopRegionExtractorLogic, self).__init__()
        except Exception:
            pass
        self.last_ear_vertex_indices = None
        self.last_ear_cell_ids = None
        self.last_faces_vtk = None  # flattened VTK-style cell array (n v0 ... v{n-1} ...)
        self.last_source_num_points = None
        self.last_source_num_cells = None

    def set_last_vertex_indices(self, indices):
        # convenience alias
        self.set_last_indices(indices)

    def set_last_indices(self, indices):
        if indices is None or len(indices) == 0:
            self.last_ear_vertex_indices = None
        else:
            self.last_ear_vertex_indices = list(map(int, sorted(set(int(i) for i in indices))))

    def get_last_indices(self):
        return self.last_ear_vertex_indices

    def set_last_faces_vtk(self, faces_flat_array):
        """faces_flat_array is a 1D np.int64 array in VTK 'Polys' format: [n, v0, v1, ... , n, ...] using ORIGINAL vertex ids."""
        if faces_flat_array is None:
            self.last_faces_vtk = None
        else:
            arr = np.asarray(faces_flat_array, dtype=np.int64).ravel()
            self.last_faces_vtk = arr

    def get_last_faces_vtk(self):
        return self.last_faces_vtk

    # ---------- Cell-storage utilities ----------
    def set_last_cell_ids(self, cell_ids):
        """Store last pinna original cell ids (list or array)."""
        if cell_ids is None:
            self.last_ear_cell_ids = None
        else:
            self.last_ear_cell_ids = list(map(int, sorted(set(int(i) for i in cell_ids))))

    def get_last_cell_ids(self):
        return getattr(self, "last_ear_cell_ids", None)

    def save_cell_ids(self, filepath):
        arr = self.get_last_cell_ids()
        if not arr:
            raise ValueError("No cell ids stored.")
        np.save(filepath, np.array(arr, dtype=np.int64))

    def load_cell_ids(self, filepath):
        arr = np.load(filepath)
        self.set_last_cell_ids(arr)

    def save_indices(self, filepath):
        """
        Save vertex indices + exact connectivity (faces_vtk) and source mesh sizes into a .npz
        """
        if not self.last_ear_vertex_indices:
            raise ValueError("No vertex indices stored.")

        if not filepath.lower().endswith(".npz"):
            filepath += ".npz"

        vertex_indices = np.array(self.last_ear_vertex_indices, dtype=np.int64)
        faces_vtk = np.array(self.last_faces_vtk, dtype=np.int64) if self.last_faces_vtk is not None else np.array([], dtype=np.int64)
        src_np = np.int64(self.last_source_num_points if self.last_source_num_points is not None else -1)
        src_nc = np.int64(self.last_source_num_cells if self.last_source_num_cells is not None else -1)

        np.savez_compressed(
            filepath,
            vertex_indices=vertex_indices,
            faces_vtk=faces_vtk,
            cell_ids=np.array(self.last_ear_cell_ids, dtype=np.int64) if self.last_ear_cell_ids is not None else np.array([], dtype=np.int64),
            source_num_points=src_np,
            source_num_cells=src_nc
        )

    def load_indices(self, filepath):
        """
        Load vertex indices + connectivity and metadata from .npz
        """
        z = np.load(filepath, allow_pickle=False)
        v = z.get("vertex_indices", None)
        f = z.get("faces_vtk", None)
        c = z.get("cell_ids", None)
        snp = int(z.get("source_num_points", -1))
        snc = int(z.get("source_num_cells", -1))

        self.set_last_indices([] if v is None else v)
        self.set_last_faces_vtk(None if f is None or f.size == 0 else f)
        self.set_last_cell_ids([] if c is None or c.size == 0 else c)
        self.last_source_num_points = None if snp < 0 else snp
        self.last_source_num_cells = None if snc < 0 else snc

    # --------------------------- Topology re-application ---------------------------
    def apply_stored_cells_to_model(self, targetModelNode, output_name=None, debug=False, clean=True):
        """
        Create a compacted submesh from `targetModelNode` using stored original cell ids.
        """
        stored_cells = self.get_last_cell_ids()
        if not stored_cells:
            if debug:
                print("apply_stored_cells_to_model: no stored cell ids.")
            return None

        poly = vtk.vtkPolyData()
        poly.DeepCopy(targetModelNode.GetPolyData())
        if clean:
            poly = self._vtkClean(poly)
        if debug:
            print(f"apply_stored_cells_to_model: target points={poly.GetNumberOfPoints()}, cells={poly.GetNumberOfCells()}, using clean={clean}")

        nOrigCells = poly.GetNumberOfCells()
        bad = [c for c in stored_cells if c < 0 or c >= nOrigCells]
        if bad:
            if debug:
                print("apply_stored_cells_to_model: some stored cell ids out of range for target model:", bad[:10])
            return None

        used_point_ids = []
        used_point_ids_set = set()
        for cid in stored_cells:
            cell = poly.GetCell(int(cid))
            if cell is None:
                continue
            for k in range(cell.GetNumberOfPoints()):
                pid = int(cell.GetPointId(k))
                if pid not in used_point_ids_set:
                    used_point_ids_set.add(pid)
                    used_point_ids.append(pid)

        if len(used_point_ids) == 0:
            if debug:
                print("apply_stored_cells_to_model: no points used by stored cells.")
            return None

        old_to_new = {old: new for new, old in enumerate(used_point_ids)}

        newPoints = vtk.vtkPoints()
        newPoints.SetNumberOfPoints(len(used_point_ids))
        for new_idx, old_pid in enumerate(used_point_ids):
            p = poly.GetPoint(old_pid)
            newPoints.SetPoint(new_idx, p)

        newPolys = vtk.vtkCellArray()
        for cid in stored_cells:
            cell = poly.GetCell(int(cid))
            if cell is None:
                continue
            npts = cell.GetNumberOfPoints()
            idList = vtk.vtkIdList()
            for k in range(npts):
                old_pid = int(cell.GetPointId(k))
                new_pid = old_to_new.get(old_pid, None)
                if new_pid is None:
                    new_pid = -1
                idList.InsertNextId(int(new_pid))
            newPolys.InsertNextCell(idList)

        out = vtk.vtkPolyData()
        out.SetPoints(newPoints)
        out.SetPolys(newPolys)
        out.Modified()
        out = self._vtkClean(out)

        outName = output_name if output_name else (targetModelNode.GetName() + "_Ear_from_cells")
        outNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', outName)
        outNode.SetAndObservePolyData(out)
        outNode.CreateDefaultDisplayNodes()
        if debug:
            print("apply_stored_cells_to_model: created node", outName, "points=", out.GetNumberOfPoints(), "cells=", out.GetNumberOfCells())
        return outNode

    def apply_indices_to_model(self, targetModelNode, vertex_indices=None, include_partial=False,
                               output_name=None, debug=False, clean=True,
                               preserve_connectivity=False, use_stored_cells=True):
        """
        Create a new model node that contains cells from targetModelNode according to stored vertex/cell info.
        """
        if vertex_indices is None:
            vertex_indices = self.get_last_indices()
        if not vertex_indices:
            raise ValueError("No vertex indices provided or stored.")

        vid_set = set(int(i) for i in vertex_indices)

        poly = vtk.vtkPolyData()
        poly.DeepCopy(targetModelNode.GetPolyData())
        if not preserve_connectivity and clean:
            poly = self._vtkClean(poly)

        if debug:
            print(f"apply_indices_to_model: target points={poly.GetNumberOfPoints()}, cells={poly.GetNumberOfCells()}, preserve_conn={preserve_connectivity}, use_stored_cells={use_stored_cells}, include_partial={include_partial}")

        stored_cells = self.get_last_cell_ids()
        if use_stored_cells and stored_cells and self.last_source_num_points is not None and self.last_source_num_cells is not None:
            if poly.GetNumberOfPoints() == self.last_source_num_points and poly.GetNumberOfCells() == self.last_source_num_cells:
                if debug:
                    print("apply_indices_to_model: Using stored cell ids (topology matches).")
                out_poly = self._build_submesh_from_cell_ids(poly, stored_cells, debug=debug)
                if preserve_connectivity:
                    outPD = vtk.vtkPolyData()
                    outPoints = vtk.vtkPoints()
                    outPoints.DeepCopy(poly.GetPoints())
                    outPD.SetPoints(outPoints)
                    polys = vtk.vtkCellArray()
                    for cid in stored_cells:
                        cell = poly.GetCell(cid)
                        idList = vtk.vtkIdList()
                        for k in range(cell.GetNumberOfPoints()):
                            idList.InsertNextId(int(cell.GetPointId(k)))
                        polys.InsertNextCell(idList)
                    outPD.SetPolys(polys)
                    out_poly = self._vtkClean(outPD) if not preserve_connectivity else outPD
                if out_poly.GetNumberOfPoints() == 0:
                    if debug:
                        print("apply_indices_to_model: stored-cell extraction produced empty poly (unexpected). Falling back.")
                else:
                    outName = output_name if output_name else (targetModelNode.GetName() + "_EarFromIndices")
                    outNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', outName)
                    outNode.SetAndObservePolyData(out_poly)
                    outNode.CreateDefaultDisplayNodes()
                    return outNode
            else:
                if debug:
                    print("apply_indices_to_model: target topology differs from stored source -> cannot use stored cell ids.")

        keep_cell_ids = []
        nCells = poly.GetNumberOfCells()
        for cid in range(nCells):
            cell = poly.GetCell(cid)
            npts = cell.GetNumberOfPoints()
            pts = [cell.GetPointId(k) for k in range(npts)]
            in_count = sum(1 for pid in pts if pid in vid_set)
            if include_partial:
                if in_count > 0:
                    keep_cell_ids.append(cid)
            else:
                if in_count == npts:
                    keep_cell_ids.append(cid)

        if len(keep_cell_ids) == 0:
            if debug:
                print("apply_indices_to_model: no cells matched the provided vertex index set (fallback).")
            return None

        out_poly = self._build_submesh_from_cell_ids(poly, keep_cell_ids, debug=debug)
        if out_poly.GetNumberOfPoints() == 0:
            return None

        outName = output_name if output_name else (targetModelNode.GetName() + "_EarFromIndices")
        outNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', outName)
        outNode.SetAndObservePolyData(out_poly)
        outNode.CreateDefaultDisplayNodes()
        return outNode

    # --------------------------- Utilities ---------------------------
    def _vtkClean(self, polydata):
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        cleaner.Update()
        return cleaner.GetOutput()

    def _cell_centroid(self, polydata, cellId):
        cell = polydata.GetCell(cellId)
        pts = cell.GetPoints()
        centroid = np.mean([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())], axis=0)
        return centroid

    def _closestVertexId(self, coords, point):
        diffs = coords - point
        d2 = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(d2))

    def _build_submesh_from_cell_ids(self, polydata, cell_ids, debug=False):
        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        ids = vtk.vtkIdTypeArray()
        for cid in cell_ids:
            ids.InsertNextValue(cid)
        selectionNode.SetSelectionList(ids)
        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)
        extractSelection = vtk.vtkExtractSelection()
        extractSelection.SetInputData(0, polydata)
        extractSelection.SetInputData(1, selection)
        extractSelection.Update()

        geomFilter = vtk.vtkGeometryFilter()
        geomFilter.SetInputData(extractSelection.GetOutput())
        geomFilter.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(geomFilter.GetOutput())
        return self._vtkClean(out)

    def _buildAdjacencyGraph(self, polydata, debug=False):
        nPoints = polydata.GetNumberOfPoints()
        coords = np.array([polydata.GetPoint(i) for i in range(nPoints)])
        edges = set()
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            ids = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
            for a, b in zip(ids, ids[1:] + [ids[0]]):
                edges.add(tuple(sorted((a, b))))
        from scipy.sparse import csr_matrix
        row, col, data = [], [], []
        for a, b in edges:
            pa, pb = coords[a], coords[b]
            d = np.linalg.norm(pa - pb)
            row.extend([a, b])
            col.extend([b, a])
            data.extend([d, d])
        graph = csr_matrix((data, (row, col)), shape=(nPoints, nPoints))
        if debug:
            print(f"_buildAdjacencyGraph: built graph with {nPoints} nodes and {len(edges)} unique edges.")
        return coords, graph

    def _loopPointsFromFids(self, polydata, fidNode, indices, graph, coords, debug=False):
        from scipy.sparse.csgraph import dijkstra
        loopPts = vtk.vtkPoints()
        ids = [self._closestVertexId(coords, np.array(fidNode.GetNthControlPointPosition(i))) for i in indices]
        fullPath = []
        for a, b in zip(ids, ids[1:] + [ids[0]]):
            dist, predecessors = dijkstra(graph, directed=False, indices=a, return_predecessors=True)
            path = []
            v = b
            if predecessors[v] == -9999:
                raise RuntimeError("No path found on mesh between fiducials.")
            while v != a:
                path.append(v)
                v = predecessors[v]
            path.append(a)
            path.reverse()
            fullPath.extend(path[:-1])
        fullPath.append(fullPath[0])
        for vid in fullPath:
            loopPts.InsertNextPoint(coords[vid])
        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(len(fullPath))
        for i, vid in enumerate(range(len(fullPath))):
            polyLine.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyLine)
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(loopPts)
        polyData.SetLines(cells)
        return loopPts, polyData

    def _split_poly_by_loop(self, poly, loop_vid_seq, debug=False):
        """
        Split poly into connected components where adjacency across loop edges is blocked.
        """
        edge_set = set()
        for k in range(len(loop_vid_seq) - 1):
            a, b = int(loop_vid_seq[k]), int(loop_vid_seq[k+1])
            if a == b:
                continue
            edge_set.add((min(a, b), max(a, b)))

        nCells = poly.GetNumberOfCells()
        edge2cells = {}
        for cid in range(nCells):
            cell = poly.GetCell(cid)
            np_cell = cell.GetNumberOfPoints()
            pts = [cell.GetPointId(i) for i in range(np_cell)]
            for i in range(np_cell):
                a, b = pts[i], pts[(i+1) % np_cell]
                e = (min(a, b), max(a, b))
                edge2cells.setdefault(e, []).append(cid)

        adj = {cid: set() for cid in range(nCells)}
        for e, cells in edge2cells.items():
            if e in edge_set:
                continue
            for i in range(len(cells)):
                for j in range(i+1, len(cells)):
                    a, b = cells[i], cells[j]
                    adj[a].add(b)
                    adj[b].add(a)

        visited = [False] * nCells
        components = []
        for cid in range(nCells):
            if visited[cid]:
                continue
            stack = [cid]
            comp = []
            while stack:
                c = stack.pop()
                if visited[c]:
                    continue
                visited[c] = True
                comp.append(c)
                for nb in adj[c]:
                    if not visited[nb]:
                        stack.append(nb)
            if comp:
                components.append(comp)

        if debug:
            print(f"_split_poly_by_loop: found {len(components)} components (cells) using loop with {len(edge_set)} edges.")

        parts = []
        for comp in components:
            if comp:
                part = self._build_submesh_from_cell_ids(poly, comp, debug=debug)
                parts.append(part)
            else:
                parts.append(vtk.vtkPolyData())

        return parts, components

    # --------------------------- Main cutting function ---------------------------
    def cutByLoops(self, meshNode, fidNode, canalIndices, otobasionIndices, tragus_index=30, createVisualLoops=True, debug=False, clean=True):
        """
        Cut the mesh using two closed loops (otobasion then canal) and return
        two separate model nodes: the Pinna (component containing tragus fiducial)
        and the Base (all other components merged).
        """

        # Try Dynamic Modeler-based cutter first if available in your environment
        try:
            if debug:
                print("cutByLoops: attempting perform_curve_cuts (DynamicModeler Curve cut)...")
            dm_res = self.perform_curve_cuts(meshNode, fidNode, canalIndices, otobasionIndices,
                                             tragus_index=tragus_index, createVisualLoops=createVisualLoops, debug=debug)
            if dm_res and (dm_res.get("pinnaNode") or dm_res.get("baseNode")):
                if debug:
                    print("cutByLoops: perform_curve_cuts succeeded — returning DM results.")
                return dm_res
            else:
                if debug:
                    print("cutByLoops: perform_curve_cuts did not produce outputs; falling back to manual method.")
        except Exception as e:
            if debug:
                print("cutByLoops: perform_curve_cuts raised exception:", e)

        result = {}

        polyOrig = vtk.vtkPolyData()
        polyOrig.DeepCopy(meshNode.GetPolyData())
        if clean:
            polyOrig = self._vtkClean(polyOrig)
        if debug:
            print(f"cutByLoops: input mesh points={polyOrig.GetNumberOfPoints()}, cells={polyOrig.GetNumberOfCells()}, clean={clean}")

        try:
            coords, graph = self._buildAdjacencyGraph(polyOrig, debug=debug)
        except Exception:
            coords, graph = None, None

        canalLoopPts, canalLoopPoly = self._loopPointsFromFids(polyOrig, fidNode, canalIndices, graph=graph, coords=coords, debug=debug)
        otoLoopPts,   otoLoopPoly   = self._loopPointsFromFids(polyOrig, fidNode, otobasionIndices, graph=graph, coords=coords, debug=debug)

        if debug:
            print("cutByLoops: canalLoopPts points:", canalLoopPts.GetNumberOfPoints())
            print("cutByLoops: otoLoopPts points:", otoLoopPts.GetNumberOfPoints())

        if createVisualLoops:
            canalNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', meshNode.GetName()+"_CanalLoop")
            canalNode.SetAndObservePolyData(canalLoopPoly)
            result["canalLoopNode"] = canalNode
            otoNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', meshNode.GetName()+"_OtobasionLoop")
            otoNode.SetAndObservePolyData(otoLoopPoly)
            result["otobasionLoopNode"] = otoNode

        def _loop_points_to_vertex_ids(poly, loopPts):
            locator = vtk.vtkPointLocator()
            locator.SetDataSet(poly)
            locator.BuildLocator()
            vid_seq = []
            for i in range(loopPts.GetNumberOfPoints()):
                p = loopPts.GetPoint(i)
                vid = locator.FindClosestPoint(p)
                vid_seq.append(int(vid))
            filtered = []
            for v in vid_seq:
                if len(filtered) == 0 or filtered[-1] != v:
                    filtered.append(v)
            if len(filtered) > 1 and filtered[0] != filtered[-1]:
                filtered.append(filtered[0])
            return filtered

        def _vertex_seq_to_edge_set(vid_seq):
            edges = set()
            for k in range(len(vid_seq)-1):
                a, b = int(vid_seq[k]), int(vid_seq[k+1])
                if a == b:
                    continue
                edges.add((min(a,b), max(a,b)))
            return edges

        def _remove_cells_with_edges(poly, edge_set, debug=False):
            nCells = poly.GetNumberOfCells()
            keep_cell_ids = []
            for cid in range(nCells):
                cell = poly.GetCell(cid)
                np_cell = cell.GetNumberOfPoints()
                pts = [cell.GetPointId(k) for k in range(np_cell)]
                remove_cell = False
                for k in range(np_cell):
                    a = pts[k]
                    b = pts[(k+1) % np_cell]
                    if (min(a,b), max(a,b)) in edge_set:
                        remove_cell = True
                        break
                if not remove_cell:
                    keep_cell_ids.append(cid)
            if debug:
                print(f"_remove_cells_with_edges: removed {nCells - len(keep_cell_ids)} / {nCells} cells")
            return self._build_submesh_from_cell_ids(poly, keep_cell_ids, debug=debug)

        oto_vid_seq = _loop_points_to_vertex_ids(polyOrig, otoLoopPts)
        oto_edges = _vertex_seq_to_edge_set(oto_vid_seq)
        if debug:
            print(f"cutByLoops: otobasion vertex-seq length {len(oto_vid_seq)}, edges {len(oto_edges)}")

        parts_after_oto, comps_after_oto = self._split_poly_by_loop(polyOrig, oto_vid_seq, debug=debug)

        if len(parts_after_oto) <= 1:
            if debug:
                print("cutByLoops: otobasion loop did not separate mesh (fallback to delete-cells).")
            poly_after_otobasion = _remove_cells_with_edges(polyOrig, oto_edges, debug=debug)
        else:
            tragus_pt = [0.0, 0.0, 0.0]
            t_idx = min(max(0, 30), fidNode.GetNumberOfControlPoints()-1)
            fidNode.GetNthControlPointPosition(t_idx, tragus_pt)
            inside_idx = None
            for i, part in enumerate(parts_after_oto):
                if part.GetNumberOfPoints() == 0:
                    continue
                locator = vtk.vtkPointLocator(); locator.SetDataSet(part); locator.BuildLocator()
                pid = locator.FindClosestPoint(tragus_pt)
                if pid >= 0:
                    p = [0.0, 0.0, 0.0]; part.GetPoint(pid, p)
                    d2 = vtk.vtkMath.Distance2BetweenPoints(p, tragus_pt)
                    if d2 <= (2.0 * 2.0):
                        inside_idx = i
                        if debug:
                            print(f"cutByLoops: tragus falls in oto-component {i} (d2={d2})")
                        break
            if inside_idx is None:
                dists = []
                for part in parts_after_oto:
                    if part.GetNumberOfCells() == 0:
                        dists.append((1e9, None))
                        continue
                    nC = part.GetNumberOfCells()
                    accum = np.zeros(3, dtype=float)
                    for cid in range(nC):
                        accum += np.array(self._cell_centroid(part, cid))
                    c = accum / float(max(1, nC))
                    dists.append((np.sum((c - np.array(tragus_pt))**2), c))
                inside_idx = int(min(range(len(dists)), key=lambda k: dists[k][0]))
                if debug:
                    print(f"cutByLoops: chosen inside_idx by centroid fallback = {inside_idx}")

            poly_after_otobasion = parts_after_oto[inside_idx]

            other_parts = [parts_after_oto[i] for i in range(len(parts_after_oto)) if i != inside_idx and parts_after_oto[i].GetNumberOfPoints() > 0]
            if len(other_parts) == 0:
                outside_part = vtk.vtkPolyData()
            elif len(other_parts) == 1:
                outside_part = other_parts[0]
            else:
                app = vtk.vtkAppendPolyData()
                for p in other_parts:
                    app.AddInputData(p)
                app.Update()
                outside_part = self._vtkClean(app.GetOutput())

        poly_after_otobasion = self._vtkClean(poly_after_otobasion)
        if debug:
            print(f"cutByLoops: after otobasion points={poly_after_otobasion.GetNumberOfPoints()}, cells={poly_after_otobasion.GetNumberOfCells()}")

        canal_vid_seq_on_open = _loop_points_to_vertex_ids(poly_after_otobasion, canalLoopPts)
        canal_edges_on_open = _vertex_seq_to_edge_set(canal_vid_seq_on_open)
        if debug:
            print(f"cutByLoops: canal vertex-seq-on-open length {len(canal_vid_seq_on_open)}, edges {len(canal_edges_on_open)}")

        parts_after_canal, comps_after_canal = self._split_poly_by_loop(poly_after_otobasion, canal_vid_seq_on_open, debug=debug)

        if len(parts_after_canal) <= 1:
            if debug:
                print("cutByLoops: canal loop did not separate inside piece (fallback to delete-cells).")
            poly_after_canal = _remove_cells_with_edges(poly_after_otobasion, canal_edges_on_open, debug=debug)
            poly_after_canal = self._vtkClean(poly_after_canal)
        else:
            tragus_pt = [0.0, 0.0, 0.0]
            t_idx = min(max(0, 30), fidNode.GetNumberOfControlPoints()-1)
            fidNode.GetNthControlPointPosition(t_idx, tragus_pt)

            pinna_idx = None
            for i, part in enumerate(parts_after_canal):
                if part.GetNumberOfPoints() == 0:
                    continue
                locator = vtk.vtkPointLocator(); locator.SetDataSet(part); locator.BuildLocator()
                pid = locator.FindClosestPoint(tragus_pt)
                if pid >= 0:
                    p = [0.0, 0.0, 0.0]; part.GetPoint(pid, p)
                    d2 = vtk.vtkMath.Distance2BetweenPoints(p, tragus_pt)
                    if d2 <= (2.0 * 2.0):
                        pinna_idx = i
                        if debug:
                            print(f"cutByLoops: tragus falls in canal-component {i} (d2={d2})")
                        break
            if pinna_idx is None:
                dists = []
                for part in parts_after_canal:
                    if part.GetNumberOfCells() == 0:
                        dists.append((1e9, None))
                        continue
                    nC = part.GetNumberOfCells()
                    accum = np.zeros(3, dtype=float)
                    for cid in range(nC):
                        accum += np.array(self._cell_centroid(part, cid))
                    c = accum / float(max(1, nC))
                    dists.append((np.sum((c - np.array(tragus_pt))**2), c))
                pinna_idx = int(min(range(len(dists)), key=lambda k: dists[k][0]))
                if debug:
                    print(f"cutByLoops: chosen pinna_idx by centroid fallback = {pinna_idx}")

            pinna_part = parts_after_canal[pinna_idx]

            # store vertex indices + cells + faces_vtk for later reuse
            try:
                orig_locator = vtk.vtkPointLocator()
                orig_locator.SetDataSet(polyOrig)
                orig_locator.BuildLocator()
                pinna_pts = pinna_part.GetPoints()
                stored_vids = set()
                for pi in range(pinna_pts.GetNumberOfPoints()):
                    p = pinna_pts.GetPoint(pi)
                    oid = orig_locator.FindClosestPoint(p)
                    stored_vids.add(int(oid))
                self.set_last_indices(sorted(stored_vids))

                try:
                    nOrigCells = polyOrig.GetNumberOfCells()
                    orig_centroids = np.zeros((nOrigCells, 3), dtype=float)
                    for cid in range(nOrigCells):
                        orig_centroids[cid, :] = self._cell_centroid(polyOrig, cid)
                    nPinnaCells = pinna_part.GetNumberOfCells()
                    pinna_centroids = np.zeros((nPinnaCells, 3), dtype=float)
                    for k in range(nPinnaCells):
                        pinna_centroids[k, :] = self._cell_centroid(pinna_part, k)
                    try:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(orig_centroids)
                        _, idxs = tree.query(pinna_centroids, k=1)
                        stored_cids = sorted(set(int(i) for i in idxs))
                    except Exception:
                        stored_cids = set()
                        for pc in pinna_centroids:
                            diffs = orig_centroids - pc
                            d2 = np.sum(diffs * diffs, axis=1)
                            stored_cids.add(int(np.argmin(d2)))
                        stored_cids = sorted(stored_cids)
                    self.set_last_cell_ids(stored_cids)

                    try:
                        faces_list = []
                        for cid in stored_cids:
                            cell = polyOrig.GetCell(int(cid))
                            npts = cell.GetNumberOfPoints()
                            faces_list.append(npts)
                            for k in range(npts):
                                faces_list.append(int(cell.GetPointId(k)))
                        faces_vtk = np.array(faces_list, dtype=np.int64)
                        self.set_last_faces_vtk(faces_vtk)
                        if debug:
                            print(f"cutByLoops: stored faces_vtk with {len(stored_cids)} cells (flattened length={faces_vtk.size}).")
                    except Exception as e:
                        if debug:
                            print("cutByLoops: could not build faces_vtk:", e)
                        self.set_last_faces_vtk(None)

                    self.last_source_num_points = polyOrig.GetNumberOfPoints()
                    self.last_source_num_cells = polyOrig.GetNumberOfCells()
                    if debug:
                        print(f"cutByLoops: stored {len(stored_vids)} vertex indices and {len(stored_cids)} cell ids for pinna. source Npoints={self.last_source_num_points}, Ncells={self.last_source_num_cells}")
                except Exception as e:
                    if debug:
                        print("cutByLoops: could not compute/store pinna cell ids:", e)
                    self.set_last_cell_ids(None)

            except Exception as e:
                if debug:
                    print("cutByLoops: could not compute/store pinna indices:", e)

            other_parts = [parts_after_canal[i] for i in range(len(parts_after_canal)) if i != pinna_idx and parts_after_canal[i].GetNumberOfPoints() > 0]
            if len(other_parts) == 0:
                base_inside_part = vtk.vtkPolyData()
            elif len(other_parts) == 1:
                base_inside_part = other_parts[0]
            else:
                app = vtk.vtkAppendPolyData()
                for p in other_parts:
                    app.AddInputData(p)
                app.Update()
                base_inside_part = self._vtkClean(app.GetOutput())

            if 'outside_part' in locals():
                if base_inside_part.GetNumberOfPoints() == 0:
                    base_part = outside_part
                else:
                    app = vtk.vtkAppendPolyData()
                    app.AddInputData(base_inside_part)
                    app.AddInputData(outside_part)
                    app.Update()
                    base_part = self._vtkClean(app.GetOutput())
            else:
                base_part = base_inside_part

            pinnaNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', meshNode.GetName()+"_Pinna")
            pinnaNode.SetAndObservePolyData(pinna_part)
            result["pinnaNode"] = pinnaNode

            baseNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', meshNode.GetName()+"_Base")
            baseNode.SetAndObservePolyData(base_part)
            result["baseNode"] = baseNode

            if debug:
                print("cutByLoops: produced pinna points=", pinna_part.GetNumberOfPoints(), "cells=", pinna_part.GetNumberOfCells())
                print("cutByLoops: produced base  points=", base_part.GetNumberOfPoints(), "cells=", base_part.GetNumberOfCells())

            return result

    # --------------------------- Exact topology apply ---------------------------
    def apply_saved_topology_to_model(self, targetModelNode, output_name=None, debug=False):
        """
        Rebuild the pinna on `targetModelNode` using EXACT saved topology.
        """
        vertex_indices = self.get_last_indices()
        faces_vtk = self.get_last_faces_vtk()
        if not vertex_indices or faces_vtk is None or faces_vtk.size == 0:
            if debug:
                print("apply_saved_topology_to_model: missing vertex_indices or faces_vtk.")
            return None

        poly = vtk.vtkPolyData()
        poly.DeepCopy(targetModelNode.GetPolyData())
        nTargetPts = poly.GetNumberOfPoints()
        if self.last_source_num_points is not None and nTargetPts != int(self.last_source_num_points):
            if debug:
                print(f"apply_saved_topology_to_model: target has {nTargetPts} points but saved source had {self.last_source_num_points}.")

        vi = np.array(vertex_indices, dtype=np.int64)
        if np.any(vi < 0) or np.any(vi >= nTargetPts):
            if debug:
                bad = vi[(vi < 0) | (vi >= nTargetPts)]
                print(f"apply_saved_topology_to_model: vertex indices out of range for target (e.g. {bad[:10]})")
            return None

        newPoints = vtk.vtkPoints()
        newPoints.SetNumberOfPoints(vi.size)
        for new_id, old_id in enumerate(vi):
            newPoints.SetPoint(new_id, poly.GetPoint(int(old_id)))

        old_to_new = {int(old): int(i) for i, old in enumerate(vi)}

        faces = np.asarray(faces_vtk, dtype=np.int64).ravel()
        newPolys = vtk.vtkCellArray()
        i = 0
        while i < faces.size:
            n = int(faces[i]); i += 1
            idList = vtk.vtkIdList()
            for k in range(n):
                old_pid = int(faces[i + k])
                new_pid = old_to_new.get(old_pid, None)
                if new_pid is None:
                    idList = None
                    break
                idList.InsertNextId(new_pid)
            i += n
            if idList is not None and idList.GetNumberOfIds() > 0:
                newPolys.InsertNextCell(idList)

        outPD = vtk.vtkPolyData()
        outPD.SetPoints(newPoints)
        outPD.SetPolys(newPolys)
        outPD.Modified()

        outName = output_name if output_name else (targetModelNode.GetName())
        outNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', outName)
        outNode.SetAndObservePolyData(outPD)
        outNode.CreateDefaultDisplayNodes()

        if debug:
            print(f"apply_saved_topology_to_model: created {outName}  points={outPD.GetNumberOfPoints()}  cells={outPD.GetNumberOfCells()}")

        return outNode
