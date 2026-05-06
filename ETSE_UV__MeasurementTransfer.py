import os, json, csv
import vtk, qt, ctk, slicer
import numpy as np
from slicer.ScriptedLoadableModule import *


#
# Module
#
class ETSE_UV__MeasurementTransfer(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "ETSE-UV Measurement Transfer"
        parent.categories = ["ETSE_UV"]
        parent.dependencies = []
        parent.contributors = ["ETSE-UV"]
        parent.helpText = """
        <p>Transfer distance measurements from one registered ear mesh to another.</p>

        <p><b>Supported input annotations:</b></p>
        <ul>
          <li>A fiducial point list where points are interpreted as pairs.</li>
          <li>Multiple line markups, where each line has exactly two control points.</li>
        </ul>

        <p>The module stores nearest mesh vertex indices, line names, descriptions, and
        source lengths in JSON. The same measurements can then be recreated on another
        registered mesh or applied in batch to a folder of meshes.</p>
        """
        parent.acknowledgementText = (
            "Developed by Juan Antonio De Rus Arance at the Escola Tècnica Superior "
            "d'Enginyeria (ETSE-UV), Universitat de València, in the context of the "
            "Signal Processing & Acoustic Technology (SPAT) research group. "
            "Thanks to the 3D Slicer, SlicerMorph, VTK, NumPy, SciPy, Trimesh, and related "
            "open-source communities."
        )


#
# Widget
#
class ETSE_UV__MeasurementTransferWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        super().setup()
        self.logic = ETSE_UV__MeasurementTransferLogic()

        #
        # SECTION 1: Manual annotation (source model)
        #
        boxA = ctk.ctkCollapsibleButton()
        boxA.text = "1) Annotate distances (point list OR line nodes)"
        self.layout.addWidget(boxA)
        layA = qt.QFormLayout(boxA)

        # Source model
        self.modelSelectorA = slicer.qMRMLNodeComboBox()
        self.modelSelectorA.nodeTypes = ["vtkMRMLModelNode"]
        self.modelSelectorA.setMRMLScene(slicer.mrmlScene)
        self.modelSelectorA.noneEnabled = False
        layA.addRow("Source model:", self.modelSelectorA)

        # Mode selection
        self.modePointsRadio = qt.QRadioButton("Use point list (pairs)")
        self.modeLinesRadio = qt.QRadioButton("Use line markups (multiple)")
        self.modePointsRadio.checked = True
        layA.addRow(self.modePointsRadio)
        layA.addRow(self.modeLinesRadio)

        self.modeGroup = qt.QButtonGroup()
        self.modeGroup.addButton(self.modePointsRadio)
        self.modeGroup.addButton(self.modeLinesRadio)

        # Point list selector (single node)
        self.fiducialSelectorA = slicer.qMRMLNodeComboBox()
        self.fiducialSelectorA.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.fiducialSelectorA.setMRMLScene(slicer.mrmlScene)
        self.fiducialSelectorA.noneEnabled = True
        layA.addRow("Point list (pairs):", self.fiducialSelectorA)

        # Line selector (multi-selection)
        self.lineSelectorA = slicer.qMRMLCheckableNodeComboBox()
        self.lineSelectorA.nodeTypes = ["vtkMRMLMarkupsLineNode"]
        self.lineSelectorA.setMRMLScene(slicer.mrmlScene)
        self.lineSelectorA.noneEnabled = True
        layA.addRow("Line markups (select several):", self.lineSelectorA)

        infoLabelA = qt.QLabel(
            "Point list mode:\n"
            "  • Use ONE Fiducial node with an EVEN number of points.\n"
            "  • Pairs (0–1), (2–3), ... are distances.\n\n"
            "Line mode:\n"
            "  • Check the Line nodes you want to use.\n"
            "  • Each Line node must have exactly 2 control points."
        )
        infoLabelA.setWordWrap(True)
        layA.addRow(infoLabelA)

        self.modePointsRadio.toggled.connect(self._updateModeWidgets)
        self._updateModeWidgets(self.modePointsRadio.checked)

        self.saveButton = qt.QPushButton("Extract nearest vertices + Save JSON")
        self.saveButton.toolTip = (
            "For each distance (pair or line), find nearest mesh vertices and "
            "store vertex indices, line name, and length into JSON."
        )
        self.saveButton.connect("clicked(bool)", self.onSave)
        layA.addRow(self.saveButton)

        #
        # SECTION 2: Reconstruction (single target model)
        #
        boxB = ctk.ctkCollapsibleButton()
        boxB.text = "2) Load indices and recreate Line markups"
        self.layout.addWidget(boxB)
        layB = qt.QFormLayout(boxB)

        self.modelSelectorB = slicer.qMRMLNodeComboBox()
        self.modelSelectorB.nodeTypes = ["vtkMRMLModelNode"]
        self.modelSelectorB.setMRMLScene(slicer.mrmlScene)
        self.modelSelectorB.noneEnabled = False
        layB.addRow("Target model:", self.modelSelectorB)

        self.prefixEdit = qt.QLineEdit("T_")
        self.prefixEdit.setToolTip("Prefix for NEW Line nodes (e.g. 'T_').")
        layB.addRow("Prefix for new lines:", self.prefixEdit)

        self.loadButton = qt.QPushButton("Load JSON and create lines")
        self.loadButton.toolTip = (
            "Load JSON with vertex indices and create NEW Line markups on "
            "the target model. New node names = prefix + original name."
        )
        self.loadButton.connect("clicked(bool)", self.onLoad)
        layB.addRow(self.loadButton)

        self.resultLabel = qt.QLabel("")
        self.resultLabel.setWordWrap(True)
        layB.addRow("New distances:", self.resultLabel)

        #
        # SECTION 3: Batch
        #
        boxC = ctk.ctkCollapsibleButton()
        boxC.text = "3) Batch apply JSON to folder of meshes → CSV"
        self.layout.addWidget(boxC)
        layC = qt.QFormLayout(boxC)

        self.batchButton = qt.QPushButton("Run batch (JSON → folder → CSV)")
        self.batchButton.toolTip = (
            "Select a JSON file (vertex indices), a folder of meshes, and a CSV "
            "output file. Each mesh is loaded, distances are computed, and one\n"
            "row per mesh is written to the CSV. No markups are created."
        )
        self.batchButton.connect("clicked(bool)", self.onBatch)
        layC.addRow(self.batchButton)

        self.layout.addStretch(1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _updateModeWidgets(self, usePoints):
        self.fiducialSelectorA.enabled = usePoints
        self.lineSelectorA.enabled = not usePoints

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def onSave(self):
        modelNode = self.modelSelectorA.currentNode()
        if not modelNode:
            slicer.util.errorDisplay("Select a source model.")
            return

        usePoints = self.modePointsRadio.checked

        try:
            if usePoints:
                fidNode = self.fiducialSelectorA.currentNode()
                if not fidNode:
                    slicer.util.errorDisplay("Select a point list node.")
                    return
                data = self.logic.computeFromFiducials(modelNode, fidNode)
            else:
                checked = list(self.lineSelectorA.checkedNodes())
                lineNodes = [n for n in checked if n is not None]
                if not lineNodes:
                    slicer.util.errorDisplay("Select (check) at least one Line markup.")
                    return
                data = self.logic.computeFromLineNodes(modelNode, lineNodes)
        except Exception as e:
            slicer.util.errorDisplay(str(e))
            return

        fn = qt.QFileDialog.getSaveFileName(
            None, "Save vertex index JSON", "", "JSON files (*.json)"
        )
        if not fn:
            return

        try:
            with open(fn, "w") as f:
                json.dump(data, f, indent=2)
            slicer.util.infoDisplay(
                f"Vertex indices saved to:\n{fn}\n"
                f"Pairs: {len(data.get('pairs', []))}"
            )
        except Exception as e:
            slicer.util.errorDisplay(f"Error writing JSON:\n{e}")

    def onLoad(self):
        modelNode = self.modelSelectorB.currentNode()
        if not modelNode:
            slicer.util.errorDisplay("Select a target model first.")
            return

        fn = qt.QFileDialog.getOpenFileName(
            None, "Open vertex index JSON", "", "JSON files (*.json)"
        )
        if not fn:
            return

        try:
            with open(fn, "r") as f:
                data = json.load(f)
        except Exception as e:
            slicer.util.errorDisplay(f"Cannot read JSON:\n{e}")
            return

        prefix = self.prefixEdit.text if self.prefixEdit.text is not None else ""

        try:
            distances = self.logic.createLinesFromVertexPairs(modelNode, data, prefix)
        except Exception as e:
            slicer.util.errorDisplay(str(e))
            return

        # Only new distances, no "(saved: ...)"
        lines = []
        for name, d_calc in distances:
            line = f"{name}: {d_calc:.3f} mm"
            lines.append(line)
            print(f"[EarDistanceTest] {line}")

        text = "\n".join(lines)
        self.resultLabel.setText(text)
        slicer.util.infoDisplay("New distances:\n" + text,
                                windowTitle="Recreated distances")

    def onBatch(self):
        # 1) JSON
        jsonPath = qt.QFileDialog.getOpenFileName(
            None, "Open vertex index JSON", "", "JSON files (*.json)"
        )
        if not jsonPath:
            return

        try:
            with open(jsonPath, "r") as f:
                data = json.load(f)
        except Exception as e:
            slicer.util.errorDisplay(f"Cannot read JSON:\n{e}")
            return

        # 2) Folder of meshes
        folder = qt.QFileDialog.getExistingDirectory(
            None, "Select folder with meshes"
        )
        if not folder:
            return

        # 3) CSV output
        csvPath = qt.QFileDialog.getSaveFileName(
            None, "Save CSV with distances", "", "CSV files (*.csv)"
        )
        if not csvPath:
            return

        try:
            self.logic.runBatch(data, folder, csvPath)
            slicer.util.infoDisplay(
                f"Batch completed.\nCSV saved to:\n{csvPath}",
                windowTitle="Batch distances"
            )
        except Exception as e:
            slicer.util.errorDisplay(str(e))


#
# Logic
#
class ETSE_UV__MeasurementTransferLogic(ScriptedLoadableModuleLogic):

    # --- helpers ---
    def _mergeDescriptions(self, d0, d1):
        """Return a single description string from two control-point descriptions.

        - If both are equal and non-empty: that value.
        - If only one is non-empty: that one.
        - If both non-empty and different: 'd0 | d1'.
        - If both empty/None: ''.
        """
        d0 = (d0 or "").strip()
        d1 = (d1 or "").strip()
        if d0 and d1:
            if d0 == d1:
                return d0
            else:
                return f"{d0} | {d1}"
        return d0 or d1 or ""

    def _polyDataInWorld(self, modelNode):
        """Return model polydata in WORLD coordinates."""
        poly = modelNode.GetPolyData()
        if poly is None:
            raise RuntimeError("Model has no polydata.")

        parentTx = modelNode.GetParentTransformNode()
        if not parentTx:
            return poly

        transformModelToWorld = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            parentTx, None, transformModelToWorld
        )
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetTransform(transformModelToWorld)
        tf.SetInputData(poly)
        tf.Update()
        return tf.GetOutput()

    def _worldPairsFromFiducials(self, fidNode):
        """Return list of (name, p0, p1, description) from ONE Fiducial node."""
        n = fidNode.GetNumberOfControlPoints()
        if n < 2 or (n % 2) != 0:
            raise RuntimeError(
                f"Fiducial node '{fidNode.GetName()}' must have an even number "
                f"of points (>=2). Now: {n}"
            )

        pts = []
        tmp = [0.0, 0.0, 0.0]
        for i in range(n):
            fidNode.GetNthControlPointPositionWorld(i, tmp)
            pts.append(list(tmp))

        pairs = []
        idx = 1
        for i in range(0, n, 2):
            p0 = pts[i]
            p1 = pts[i + 1]

            d0 = fidNode.GetNthControlPointDescription(i) or ""
            d1 = fidNode.GetNthControlPointDescription(i + 1) or ""
            desc = self._mergeDescriptions(d0, d1)

            name = f"Dist_{idx}"
            idx += 1
            pairs.append((name, p0, p1, desc))
        return pairs


    def _worldPairsFromLineNodes(self, lineNodes):
        """Return list of (name, p0, p1, description) from a LIST of Line markups."""
        if not lineNodes:
            raise RuntimeError("No Line nodes provided.")

        pairs = []
        lineNodes = list(lineNodes)
        lineNodes.sort(key=lambda n: n.GetName())

        for ln in lineNodes:
            ncp = ln.GetNumberOfControlPoints()
            if ncp != 2:
                raise RuntimeError(
                    f"Line node '{ln.GetName()}' must have exactly 2 points, has {ncp}."
                )

            p0 = [0.0, 0.0, 0.0]
            p1 = [0.0, 0.0, 0.0]
            ln.GetNthControlPointPositionWorld(0, p0)
            ln.GetNthControlPointPositionWorld(1, p1)

            d0 = ln.GetNthControlPointDescription(0) or ""
            d1 = ln.GetNthControlPointDescription(1) or ""
            desc = self._mergeDescriptions(d0, d1)

            name = ln.GetName()  # preserve original line node name (e.g. D3)
            pairs.append((name, p0, p1, desc))

        return pairs


    def _buildJSON(self, modelNode, markupName, worldPairs):
        """From world pairs -> vertex indices + length + description + JSON."""
        polyWorld = self._polyDataInWorld(modelNode)

        locator = vtk.vtkPointLocator()
        locator.SetDataSet(polyWorld)
        locator.BuildLocator()

        outPairs = []
        for item in worldPairs:
            # worldPairs elements: (name, p0, p1, description)
            name, p0, p1, desc = item

            v0 = int(locator.FindClosestPoint(p0))
            v1 = int(locator.FindClosestPoint(p1))
            length = float(np.linalg.norm(np.array(p0) - np.array(p1)))

            outPairs.append(
                {
                    "name": name,
                    "v0": v0,
                    "v1": v1,
                    "length": length,
                    "description": desc,
                }
            )

        data = {
            "version": 4,
            "modelName": modelNode.GetName(),
            "markupNodeName": markupName,
            "pairs": outPairs,
        }
        return data


    # --- public API: saving ---

    def computeFromFiducials(self, modelNode, fidNode):
        worldPairs = self._worldPairsFromFiducials(fidNode)
        return self._buildJSON(modelNode, fidNode.GetName(), worldPairs)

    def computeFromLineNodes(self, modelNode, lineNodes):
        worldPairs = self._worldPairsFromLineNodes(lineNodes)
        return self._buildJSON(modelNode, "LineNodes", worldPairs)

    # --- public API: interactive loading (creates line nodes) ---

    def createLinesFromVertexPairs(self, modelNode, data, prefix=""):
        """Create one Line markup per pair, return [(newName, dist_calc_mm), ...]."""
        if "pairs" not in data or not data["pairs"]:
            raise RuntimeError("JSON has no 'pairs' field with vertex indices.")

        polyWorld = self._polyDataInWorld(modelNode)
        pointsWorld = polyWorld.GetPoints()
        if pointsWorld is None:
            raise RuntimeError("Model polydata has no points.")

        nPoints = pointsWorld.GetNumberOfPoints()
        distances = []

        for idx, item in enumerate(data["pairs"]):
            baseName = item.get("name", f"Dist_{idx + 1}")
            v0 = int(item["v0"])
            v1 = int(item["v1"])

            if v0 < 0 or v0 >= nPoints or v1 < 0 or v1 >= nPoints:
                raise RuntimeError(
                    f"Vertex indices out of range for model '{modelNode.GetName()}': "
                    f"({v0}, {v1}) with {nPoints} points."
                )

            p0 = [0.0, 0.0, 0.0]
            p1 = [0.0, 0.0, 0.0]
            pointsWorld.GetPoint(v0, p0)
            pointsWorld.GetPoint(v1, p1)

            newName = f"{prefix}{baseName}" if prefix else baseName

            lineNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsLineNode", newName
            )
            lineNode.AddControlPointWorld(vtk.vtkVector3d(*p0))
            lineNode.AddControlPointWorld(vtk.vtkVector3d(*p1))

            d_calc = float(np.linalg.norm(np.array(p0) - np.array(p1)))
            distances.append((newName, d_calc))

        return distances

    # --- helper for batch (no markups created) ---

    def computeDistancesFromVertexPairs(self, modelNode, data):
        """Compute distances from vertex pairs WITHOUT creating markups.

        Returns: [(name, dist_mm), ...] with names from JSON.
        """
        if "pairs" not in data or not data["pairs"]:
            raise RuntimeError("JSON has no 'pairs' field with vertex indices.")

        polyWorld = self._polyDataInWorld(modelNode)
        pointsWorld = polyWorld.GetPoints()
        if pointsWorld is None:
            raise RuntimeError("Model polydata has no points.")

        nPoints = pointsWorld.GetNumberOfPoints()
        distances = []

        for idx, item in enumerate(data["pairs"]):
            name = item.get("name", f"Dist_{idx + 1}")
            v0 = int(item["v0"])
            v1 = int(item["v1"])

            if v0 < 0 or v0 >= nPoints or v1 < 0 or v1 >= nPoints:
                raise RuntimeError(
                    f"Vertex indices out of range for model '{modelNode.GetName()}': "
                    f"({v0}, {v1}) with {nPoints} points."
                )

            p0 = [0.0, 0.0, 0.0]
            p1 = [0.0, 0.0, 0.0]
            pointsWorld.GetPoint(v0, p0)
            pointsWorld.GetPoint(v1, p1)

            d_calc = float(np.linalg.norm(np.array(p0) - np.array(p1)))
            distances.append((name, d_calc))

        return distances

    # --- batch processing ---

    def runBatch(self, data, folder, csvPath):
        """Apply JSON to all meshes in folder and write CSV with distances."""

        # Collect mesh files
        exts = {".stl", ".ply", ".vtk", ".vtp", ".obj", ".gltf", ".glb"}
        files = [
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ]
        files.sort()

        if not files:
            raise RuntimeError("No mesh files with supported extensions in folder.")

        # Determine distance names (columns)
        if "pairs" not in data or not data["pairs"]:
            raise RuntimeError("JSON has no 'pairs' field with vertex indices.")
        distNames = [item.get("name", f"Dist_{i+1}") for i, item in enumerate(data["pairs"])]

        # CSV header: Model, dist1, dist2, ...
        header = ["Model"] + distNames

        with open(csvPath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for fname in files:
                fullPath = os.path.join(folder, fname)
                print(f"[EarDistanceTest] Processing: {fullPath}")

                modelNode = slicer.util.loadModel(fullPath)
                if modelNode is None:
                    print(f"[EarDistanceTest]  -> Failed to load as model, skipping.")
                    continue

                try:
                    dists = self.computeDistancesFromVertexPairs(modelNode, data)
                except Exception as e:
                    print(f"[EarDistanceTest]  -> Error computing distances: {e}")
                    slicer.mrmlScene.RemoveNode(modelNode)
                    continue

                # Reorder distances by distNames to be safe
                distDict = {name: value for name, value in dists}
                row = [os.path.splitext(fname)[0]]
                for dn in distNames:
                    v = distDict.get(dn, "")
                    row.append(f"{v:.6f}" if isinstance(v, float) else v)
                writer.writerow(row)

                slicer.mrmlScene.RemoveNode(modelNode)

        print(f"[EarDistanceTest] Batch done. CSV: {csvPath}")
