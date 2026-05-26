import os
import json
import csv
import heapq
import math

import vtk
import qt
import ctk
import slicer
import numpy as np
from slicer.ScriptedLoadableModule import *


# =============================================================================
# Module
# =============================================================================
class ETSE_UV__MeasurementTransfer(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "ETSE-UV Measurement Transfer"
        parent.categories = ["ETSE_UV"]
        parent.dependencies = []
        parent.contributors = ["ETSE-UV"]
        parent.helpText = """
        <p>Transfer distance measurements between registered ear meshes.</p>

        <p><b>Input measurement definitions:</b></p>
        <ul>
          <li>A fiducial point list where consecutive points are interpreted as pairs.</li>
          <li>Multiple Line markups, where each line has exactly two control points.</li>
          <li>A NumPy index file (.npy/.npz) containing vertex-index pairs.</li>
          <li>An ETSE-UV measurement config JSON with names, descriptions and vertex pairs.</li>
        </ul>

        <p>The module saves a measurement config JSON that preserves the expected
        distance names/descriptions (for example D1-D7) instead of depending on the
        names of temporary Line nodes in the scene.</p>

        <p>For registered meshes with small local errors, distances can be computed
        point-to-point or zone-to-zone: each endpoint vertex is expanded to a local
        mesh neighbourhood and the reported value is the mean of all cross distances
        between both zones.</p>
        """
        parent.acknowledgementText = (
            "Developed by Juan Antonio De Rus Arance at the Escola Tècnica Superior "
            "d'Enginyeria (ETSE-UV), Universitat de València, in the context of the "
            "Signal Processing & Acoustic Technology (SPAT) research group. "
            "Thanks to the 3D Slicer, SlicerMorph, VTK, NumPy, SciPy, Trimesh, and related "
            "open-source communities."
        )


# =============================================================================
# Widget
# =============================================================================
class ETSE_UV__MeasurementTransferWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        super().setup()
        self.logic = ETSE_UV__MeasurementTransferLogic()

        # ------------------------------------------------------------------
        # SECTION 1: Build / save measurement config
        # ------------------------------------------------------------------
        boxA = ctk.ctkCollapsibleButton()
        boxA.text = "1) Build measurement config"
        self.layout.addWidget(boxA)
        layA = qt.QFormLayout(boxA)

        self.modelSelectorA = slicer.qMRMLNodeComboBox()
        self.modelSelectorA.nodeTypes = ["vtkMRMLModelNode"]
        self.modelSelectorA.setMRMLScene(slicer.mrmlScene)
        self.modelSelectorA.noneEnabled = False
        layA.addRow("Source model:", self.modelSelectorA)

        self.modePointsRadio = qt.QRadioButton("Use point list (pairs)")
        self.modeLinesRadio = qt.QRadioButton("Use line markups (multiple)")
        self.modePointsRadio.checked = True
        layA.addRow(self.modePointsRadio)
        layA.addRow(self.modeLinesRadio)

        self.modeGroup = qt.QButtonGroup()
        self.modeGroup.addButton(self.modePointsRadio)
        self.modeGroup.addButton(self.modeLinesRadio)

        self.fiducialSelectorA = slicer.qMRMLNodeComboBox()
        self.fiducialSelectorA.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.fiducialSelectorA.setMRMLScene(slicer.mrmlScene)
        self.fiducialSelectorA.noneEnabled = True
        layA.addRow("Point list (pairs):", self.fiducialSelectorA)

        self.lineSelectorA = slicer.qMRMLCheckableNodeComboBox()
        self.lineSelectorA.nodeTypes = ["vtkMRMLMarkupsLineNode"]
        self.lineSelectorA.setMRMLScene(slicer.mrmlScene)
        self.lineSelectorA.noneEnabled = True
        layA.addRow("Line markups (select several):", self.lineSelectorA)

        infoLabelA = qt.QLabel(
            "Point list mode:\n"
            "  • Use ONE Fiducial node with an EVEN number of points.\n"
            "  • Pairs (0-1), (2-3), ... are distances.\n\n"
            "Line mode:\n"
            "  • Check the Line nodes you want to use.\n"
            "  • Each Line node must have exactly 2 control points.\n\n"
            "The saved config stores fixed measurement names/descriptions, so target outputs can stay D1-D7."
        )
        infoLabelA.setWordWrap(True)
        layA.addRow(infoLabelA)

        self.modePointsRadio.toggled.connect(self._updateModeWidgets)
        self._updateModeWidgets(self.modePointsRadio.checked)

        self.saveButton = qt.QPushButton("Extract nearest vertices + Save config JSON")
        self.saveButton.toolTip = (
            "For each distance, find nearest mesh vertices and store vertex indices, "
            "measurement name, description, and source length into a config JSON."
        )
        self.saveButton.connect("clicked(bool)", self.onSave)
        layA.addRow(self.saveButton)

        # ------------------------------------------------------------------
        # SECTION 1b: Build config directly from index file
        # ------------------------------------------------------------------
        boxN = ctk.ctkCollapsibleButton()
        boxN.text = "1b) Build config from vertex-index file (.npy/.npz)"
        self.layout.addWidget(boxN)
        layN = qt.QFormLayout(boxN)

        self.npyIndexPathEdit = qt.QLineEdit("")
        layN.addRow("Index file (.npy/.npz):", self._pathRow(self.npyIndexPathEdit, self.onBrowseNpyIndexFile))

        self.npyMetadataPathEdit = qt.QLineEdit("")
        self.npyMetadataPathEdit.setToolTip(
            "Optional JSON/CSV/TXT with measurement names/descriptions.\n"
            "CSV columns may be: name, description, v0, v1.\n"
            "JSON may be {'pairs': [...]} or a list of pair dictionaries."
        )
        layN.addRow("Optional names/descriptions:", self._pathRow(self.npyMetadataPathEdit, self.onBrowseNpyMetadataFile))

        self.npyOutputConfigPathEdit = qt.QLineEdit("")
        layN.addRow("Output config JSON:", self._pathRow(self.npyOutputConfigPathEdit, self.onBrowseNpyOutputConfig, save=True))

        self.defaultNamePatternEdit = qt.QLineEdit("D{}")
        self.defaultNamePatternEdit.setToolTip("Used when the index file/metadata has no names. Example: D{} gives D1, D2, ...")
        layN.addRow("Default name pattern:", self.defaultNamePatternEdit)

        self.buildFromNpyButton = qt.QPushButton("Save measurement config from index file")
        self.buildFromNpyButton.toolTip = (
            "Load vertex index pairs from .npy/.npz, optionally merge names/descriptions, "
            "and save an ETSE-UV measurement config JSON."
        )
        self.buildFromNpyButton.connect("clicked(bool)", self.onBuildConfigFromNpy)
        layN.addRow(self.buildFromNpyButton)

        # ------------------------------------------------------------------
        # SECTION 2: Single target
        # ------------------------------------------------------------------
        boxB = ctk.ctkCollapsibleButton()
        boxB.text = "2) Load config and recreate measurements on one target"
        self.layout.addWidget(boxB)
        layB = qt.QFormLayout(boxB)

        self.modelSelectorB = slicer.qMRMLNodeComboBox()
        self.modelSelectorB.nodeTypes = ["vtkMRMLModelNode"]
        self.modelSelectorB.setMRMLScene(slicer.mrmlScene)
        self.modelSelectorB.noneEnabled = False
        layB.addRow("Target model:", self.modelSelectorB)

        self.singleConfigPathEdit = qt.QLineEdit("")
        layB.addRow("Config / index file:", self._pathRow(self.singleConfigPathEdit, self.onBrowseSingleConfig))

        self.singleMetadataPathEdit = qt.QLineEdit("")
        self.singleMetadataPathEdit.setToolTip("Optional names/descriptions file when Config is .npy/.npz.")
        layB.addRow("Optional metadata:", self._pathRow(self.singleMetadataPathEdit, self.onBrowseSingleMetadata))

        self.prefixEdit = qt.QLineEdit("T_")
        self.prefixEdit.setToolTip("Prefix for NEW Line nodes (e.g. 'T_').")
        layB.addRow("Prefix for new nodes:", self.prefixEdit)

        self.singleUseZoneMeanCheck = qt.QCheckBox("Compute zone-to-zone statistics")
        self.singleUseZoneMeanCheck.checked = False
        self.singleUseZoneMeanCheck.toolTip = (
            "If enabled, each endpoint vertex is expanded to a connected mesh neighbourhood.\n"
            "The main reported distance remains point-to-point; zone mean and centroid distance are shown in parentheses."
        )
        layB.addRow(self.singleUseZoneMeanCheck)

        self.singleRadiusSpin = ctk.ctkDoubleSpinBox()
        self.singleRadiusSpin.decimals = 2
        self.singleRadiusSpin.minimum = 0.0
        self.singleRadiusSpin.maximum = 1000.0
        self.singleRadiusSpin.singleStep = 0.5
        self.singleRadiusSpin.value = 2.0
        self.singleRadiusSpin.setToolTip("Geodesic neighbourhood radius in model units, usually mm. Default = 2.0. 0 = point-to-point only.")
        layB.addRow("Zone geodesic radius:", self.singleRadiusSpin)

        self.singleHighlightZonesCheck = qt.QCheckBox("Highlight all zone points used")
        self.singleHighlightZonesCheck.checked = True
        self.singleHighlightZonesCheck.setToolTip("Create two point-cloud model nodes showing all endpoint-zone points used by the measurements.")
        layB.addRow(self.singleHighlightZonesCheck)

        self.singleShowCentroidLinesCheck = qt.QCheckBox("Also draw centroid-to-centroid lines")
        self.singleShowCentroidLinesCheck.checked = False
        self.singleShowCentroidLinesCheck.setToolTip(
            "By default the module draws the original point-to-point distance lines.\n"
            "Enable this to also draw additional lines between the two zone centroids.\n"
            "Those centroid lines are only visual helpers; the main output remains point-to-point."
        )
        layB.addRow(self.singleShowCentroidLinesCheck)

        self.loadButton = qt.QPushButton("Load config and create measurements")
        self.loadButton.toolTip = (
            "Create Line markups on the target model. By default, lines are drawn between the original saved vertices. "
            "In zone mode, zone statistics are computed and can optionally be visualized with centroid-to-centroid lines."
        )
        self.loadButton.connect("clicked(bool)", self.onLoad)
        layB.addRow(self.loadButton)

        self.resultLabel = qt.QLabel("")
        self.resultLabel.setWordWrap(True)
        layB.addRow("New distances:", self.resultLabel)

        # ------------------------------------------------------------------
        # SECTION 3: Batch - visible folder/path fields instead of pop-ups
        # ------------------------------------------------------------------
        boxC = ctk.ctkCollapsibleButton()
        boxC.text = "3) Batch apply config to folder of meshes -> CSV"
        self.layout.addWidget(boxC)
        layC = qt.QFormLayout(boxC)

        self.batchConfigPathEdit = qt.QLineEdit("")
        layC.addRow("Config / index file:", self._pathRow(self.batchConfigPathEdit, self.onBrowseBatchConfig))

        self.batchMetadataPathEdit = qt.QLineEdit("")
        self.batchMetadataPathEdit.setToolTip("Optional names/descriptions file when Config is .npy/.npz.")
        layC.addRow("Optional metadata:", self._pathRow(self.batchMetadataPathEdit, self.onBrowseBatchMetadata))

        self.batchFolderEdit = qt.QLineEdit("")
        layC.addRow("Folder with meshes:", self._pathRow(self.batchFolderEdit, self.onBrowseBatchFolder, folder=True))

        self.batchCsvPathEdit = qt.QLineEdit("")
        layC.addRow("Output CSV:", self._pathRow(self.batchCsvPathEdit, self.onBrowseBatchCsv, save=True))

        self.batchUseZoneMeanCheck = qt.QCheckBox("Compute zone-to-zone statistics")
        self.batchUseZoneMeanCheck.checked = False
        layC.addRow(self.batchUseZoneMeanCheck)

        self.batchRadiusSpin = ctk.ctkDoubleSpinBox()
        self.batchRadiusSpin.decimals = 2
        self.batchRadiusSpin.minimum = 0.0
        self.batchRadiusSpin.maximum = 1000.0
        self.batchRadiusSpin.singleStep = 0.5
        self.batchRadiusSpin.value = 2.0
        self.batchRadiusSpin.setToolTip("Geodesic neighbourhood radius in model units, usually mm. Default = 2.0. 0 = point-to-point only.")
        layC.addRow("Zone geodesic radius:", self.batchRadiusSpin)

        self.batchExtraStatsCheck = qt.QCheckBox("Also write point-to-point values and zone sizes")
        self.batchExtraStatsCheck.checked = True
        self.batchExtraStatsCheck.toolTip = (
            "Adds zone/debug columns: <name>__zone_mean_mm, <name>__centroid_mm, <name>__zone0_n, <name>__zone1_n. The main <name>__mm column is always point-to-point."
        )
        layC.addRow(self.batchExtraStatsCheck)

        self.batchButton = qt.QPushButton("Run batch")
        self.batchButton.toolTip = "Process all supported mesh files in the selected folder and write one CSV row per mesh."
        self.batchButton.connect("clicked(bool)", self.onBatch)
        layC.addRow(self.batchButton)

        self.layout.addStretch(1)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _updateModeWidgets(self, usePoints):
        self.fiducialSelectorA.enabled = usePoints
        self.lineSelectorA.enabled = not usePoints

    def _text(self, widget):
        try:
            txt = widget.text
            if callable(txt):
                txt = txt()
            return str(txt or "").strip()
        except Exception:
            return ""

    def _pathRow(self, lineEdit, callback, folder=False, save=False):
        row = qt.QHBoxLayout()
        lineEdit.setMinimumWidth(300)
        row.addWidget(lineEdit)
        btn = qt.QPushButton("Browse...")
        btn.setMaximumWidth(90)
        btn.clicked.connect(callback)
        row.addWidget(btn)
        w = qt.QWidget()
        w.setLayout(row)
        return w

    def _openFile(self, title, lineEdit, filters="All files (*)"):
        path = qt.QFileDialog.getOpenFileName(self.parent, title, "", filters)
        if path:
            lineEdit.setText(str(path))

    def _saveFile(self, title, lineEdit, filters="All files (*)", defaultExt=None):
        path = qt.QFileDialog.getSaveFileName(self.parent, title, "", filters)
        if path:
            path = str(path)
            if defaultExt and not path.lower().endswith(defaultExt.lower()):
                path += defaultExt
            lineEdit.setText(path)

    def _openFolder(self, title, lineEdit):
        path = qt.QFileDialog.getExistingDirectory(self.parent, title)
        if path:
            lineEdit.setText(str(path))

    # Browse callbacks: config build
    def onBrowseNpyIndexFile(self):
        self._openFile("Select vertex index file", self.npyIndexPathEdit, "NumPy files (*.npy *.npz);;All files (*)")

    def onBrowseNpyMetadataFile(self):
        self._openFile("Select optional names/descriptions file", self.npyMetadataPathEdit, "Metadata files (*.json *.csv *.txt);;All files (*)")

    def onBrowseNpyOutputConfig(self):
        self._saveFile("Save measurement config JSON", self.npyOutputConfigPathEdit, "JSON files (*.json);;All files (*)", ".json")

    # Browse callbacks: single
    def onBrowseSingleConfig(self):
        self._openFile("Select measurement config or index file", self.singleConfigPathEdit, "Measurement files (*.json *.npy *.npz);;All files (*)")

    def onBrowseSingleMetadata(self):
        self._openFile("Select optional names/descriptions file", self.singleMetadataPathEdit, "Metadata files (*.json *.csv *.txt);;All files (*)")

    # Browse callbacks: batch
    def onBrowseBatchConfig(self):
        self._openFile("Select measurement config or index file", self.batchConfigPathEdit, "Measurement files (*.json *.npy *.npz);;All files (*)")

    def onBrowseBatchMetadata(self):
        self._openFile("Select optional names/descriptions file", self.batchMetadataPathEdit, "Metadata files (*.json *.csv *.txt);;All files (*)")

    def onBrowseBatchFolder(self):
        self._openFolder("Select folder with meshes", self.batchFolderEdit)

    def onBrowseBatchCsv(self):
        self._saveFile("Save CSV with distances", self.batchCsvPathEdit, "CSV files (*.csv);;All files (*)", ".csv")

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

        fn = qt.QFileDialog.getSaveFileName(None, "Save measurement config JSON", "", "JSON files (*.json)")
        if not fn:
            return
        fn = str(fn)
        if not fn.lower().endswith(".json"):
            fn += ".json"

        try:
            self.logic.saveMeasurementConfig(data, fn)
            slicer.util.infoDisplay(f"Measurement config saved to:\n{fn}\nPairs: {len(data.get('pairs', []))}")
        except Exception as e:
            slicer.util.errorDisplay(f"Error writing JSON:\n{e}")

    def onBuildConfigFromNpy(self):
        indexPath = self._text(self.npyIndexPathEdit)
        metadataPath = self._text(self.npyMetadataPathEdit) or None
        outputPath = self._text(self.npyOutputConfigPathEdit)
        defaultPattern = self._text(self.defaultNamePatternEdit) or "D{}"

        if not indexPath or not os.path.isfile(indexPath):
            slicer.util.errorDisplay("Select a valid .npy/.npz index file.")
            return
        if metadataPath and not os.path.isfile(metadataPath):
            slicer.util.errorDisplay("The optional metadata file does not exist.")
            return
        if not outputPath:
            slicer.util.errorDisplay("Select an output config JSON path.")
            return

        try:
            data = self.logic.configFromIndexFile(indexPath, metadataPath=metadataPath, defaultNamePattern=defaultPattern)
            self.logic.saveMeasurementConfig(data, outputPath)
            slicer.util.infoDisplay(
                f"Saved measurement config:\n{outputPath}\n"
                f"Pairs: {len(data.get('pairs', []))}"
            )
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def onLoad(self):
        modelNode = self.modelSelectorB.currentNode()
        if not modelNode:
            slicer.util.errorDisplay("Select a target model first.")
            return

        configPath = self._text(self.singleConfigPathEdit)
        metadataPath = self._text(self.singleMetadataPathEdit) or None
        if not configPath or not os.path.isfile(configPath):
            slicer.util.errorDisplay("Select a valid config / index file.")
            return
        if metadataPath and not os.path.isfile(metadataPath):
            slicer.util.errorDisplay("The optional metadata file does not exist.")
            return

        try:
            data = self.logic.loadMeasurementConfig(configPath, metadataPath=metadataPath)
        except Exception as e:
            slicer.util.errorDisplay(f"Cannot read measurement config:\n{e}")
            return

        prefix = self._text(self.prefixEdit)
        useZoneMean = bool(self.singleUseZoneMeanCheck.checked)
        radiusMm = float(self.singleRadiusSpin.value)
        highlightZones = bool(self.singleHighlightZonesCheck.checked)
        showCentroidLines = bool(self.singleShowCentroidLinesCheck.checked)

        try:
            results = self.logic.createMeasurementsFromConfig(
                modelNode=modelNode,
                data=data,
                prefix=prefix,
                useZoneMean=useZoneMean,
                radiusMm=radiusMm,
                highlightZones=highlightZones,
                showCentroidLines=showCentroidLines,
            )
        except Exception as e:
            slicer.util.errorDisplay(str(e))
            return

        lines = []
        for r in results:
            # Main value is always the original saved vertex-to-vertex distance.
            if r.get("mode") == "zone_mean":
                line = (
                    f"{r['name']}: {r['point_to_point_mm']:.3f} mm "
                    f"(zone mean={r['distance_mm']:.3f}; "
                    f"centroid={r['centroid_distance_mm']:.3f}; "
                    f"n0={r['zone0_n']}, n1={r['zone1_n']}; "
                    f"radius={r['zone_radius_mm']:.2f})"
                )
            else:
                line = f"{r['name']}: {r['point_to_point_mm']:.3f} mm"
            lines.append(line)
            print(f"[ETSE_UV_MeasurementTransfer] {line}")

        text = "\n".join(lines)
        self.resultLabel.setText(text)
        slicer.util.infoDisplay("New distances:\n" + text, windowTitle="Recreated distances")

    def onBatch(self):
        configPath = self._text(self.batchConfigPathEdit)
        metadataPath = self._text(self.batchMetadataPathEdit) or None
        folder = self._text(self.batchFolderEdit)
        csvPath = self._text(self.batchCsvPathEdit)

        if not configPath or not os.path.isfile(configPath):
            slicer.util.errorDisplay("Select a valid config / index file.")
            return
        if metadataPath and not os.path.isfile(metadataPath):
            slicer.util.errorDisplay("The optional metadata file does not exist.")
            return
        if not folder or not os.path.isdir(folder):
            slicer.util.errorDisplay("Select a valid mesh folder.")
            return
        if not csvPath:
            slicer.util.errorDisplay("Select an output CSV path.")
            return
        if not csvPath.lower().endswith(".csv"):
            csvPath += ".csv"
            self.batchCsvPathEdit.setText(csvPath)

        try:
            data = self.logic.loadMeasurementConfig(configPath, metadataPath=metadataPath)
            processed, written = self.logic.runBatch(
                data=data,
                folder=folder,
                csvPath=csvPath,
                useZoneMean=bool(self.batchUseZoneMeanCheck.checked),
                radiusMm=float(self.batchRadiusSpin.value),
                includeExtraStats=bool(self.batchExtraStatsCheck.checked),
            )
            slicer.util.infoDisplay(
                f"Batch completed.\nProcessed meshes: {processed}\nRows written: {written}\nCSV saved to:\n{csvPath}",
                windowTitle="Batch distances",
            )
        except Exception as e:
            slicer.util.errorDisplay(str(e))


# =============================================================================
# Logic
# =============================================================================
class ETSE_UV__MeasurementTransferLogic(ScriptedLoadableModuleLogic):

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------
    def _mergeDescriptions(self, d0, d1):
        d0 = (d0 or "").strip()
        d1 = (d1 or "").strip()
        if d0 and d1:
            if d0 == d1:
                return d0
            return f"{d0} | {d1}"
        return d0 or d1 or ""

    def _polyDataInWorld(self, modelNode):
        """Return model polydata in WORLD coordinates. Point IDs are preserved."""
        poly = modelNode.GetPolyData()
        if poly is None:
            raise RuntimeError("Model has no polydata.")

        parentTx = modelNode.GetParentTransformNode()
        if not parentTx:
            return poly

        transformModelToWorld = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(parentTx, None, transformModelToWorld)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetTransform(transformModelToWorld)
        tf.SetInputData(poly)
        tf.Update()
        return tf.GetOutput()

    def _pointsToNumpy(self, vtkPoints):
        if vtkPoints is None:
            raise RuntimeError("Polydata has no points.")
        n = vtkPoints.GetNumberOfPoints()
        arr = np.zeros((n, 3), dtype=float)
        p = [0.0, 0.0, 0.0]
        for i in range(n):
            vtkPoints.GetPoint(i, p)
            arr[i, :] = p
        return arr

    def _makeDisplayNode(self, modelNode, color=(1.0, 0.2, 0.2), opacity=1.0, pointSize=None, lineWidth=None):
        modelNode.CreateDefaultDisplayNodes()
        dn = modelNode.GetDisplayNode()
        if dn:
            if color is not None:
                dn.SetColor(*color)
            dn.SetOpacity(float(opacity))
            if pointSize is not None and hasattr(dn, "SetPointSize"):
                dn.SetPointSize(int(pointSize))
            if lineWidth is not None and hasattr(dn, "SetLineWidth"):
                dn.SetLineWidth(int(lineWidth))
            dn.SetVisibility(True)
        return dn

    # ------------------------------------------------------------------
    # Measurement definition from markups
    # ------------------------------------------------------------------
    def _worldPairsFromFiducials(self, fidNode):
        """Return list of (name, p0, p1, description) from ONE Fiducial node."""
        n = fidNode.GetNumberOfControlPoints()
        if n < 2 or (n % 2) != 0:
            raise RuntimeError(
                f"Fiducial node '{fidNode.GetName()}' must have an even number of points (>=2). Now: {n}"
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

            # If both endpoint labels are identical/non-empty, use that as the measurement name;
            # otherwise use D1, D2... This avoids accidental node names like AAAAD1.
            l0 = (fidNode.GetNthControlPointLabel(i) or "").strip()
            l1 = (fidNode.GetNthControlPointLabel(i + 1) or "").strip()
            if l0 and l1 and l0 == l1:
                name = l0
            else:
                name = f"D{idx}"
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
                raise RuntimeError(f"Line node '{ln.GetName()}' must have exactly 2 points, has {ncp}.")

            p0 = [0.0, 0.0, 0.0]
            p1 = [0.0, 0.0, 0.0]
            ln.GetNthControlPointPositionWorld(0, p0)
            ln.GetNthControlPointPositionWorld(1, p1)

            d0 = ln.GetNthControlPointDescription(0) or ""
            d1 = ln.GetNthControlPointDescription(1) or ""
            desc = self._mergeDescriptions(d0, d1)

            # Preserve the original line name. This is the safest path for D1-D7.
            name = ln.GetName()
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
            name, p0, p1, desc = item
            v0 = int(locator.FindClosestPoint(p0))
            v1 = int(locator.FindClosestPoint(p1))
            length = float(np.linalg.norm(np.array(p0, dtype=float) - np.array(p1, dtype=float)))

            outPairs.append(
                {
                    "name": str(name),
                    "v0": v0,
                    "v1": v1,
                    "length": length,
                    "description": str(desc or ""),
                }
            )

        return {
            "version": 5,
            "type": "ETSE_UV_measurement_config",
            "modelName": modelNode.GetName(),
            "markupNodeName": markupName,
            "distanceMode": "point_to_point",
            "pairs": outPairs,
        }

    def computeFromFiducials(self, modelNode, fidNode):
        worldPairs = self._worldPairsFromFiducials(fidNode)
        return self._buildJSON(modelNode, fidNode.GetName(), worldPairs)

    def computeFromLineNodes(self, modelNode, lineNodes):
        worldPairs = self._worldPairsFromLineNodes(lineNodes)
        return self._buildJSON(modelNode, "LineNodes", worldPairs)

    # ------------------------------------------------------------------
    # Config loading / saving, including NPY/NPZ index files
    # ------------------------------------------------------------------
    def saveMeasurementConfig(self, data, path):
        self._validateConfig(data)
        if not path.lower().endswith(".json"):
            path += ".json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def loadMeasurementConfig(self, path, metadataPath=None, defaultNamePattern="D{}"):
        if not path:
            raise RuntimeError("No config path provided.")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._validateConfig(data)
            return self._normalizeConfig(data)
        if ext in (".npy", ".npz"):
            return self.configFromIndexFile(path, metadataPath=metadataPath, defaultNamePattern=defaultNamePattern)
        raise RuntimeError(f"Unsupported measurement file extension: {ext}")

    def configFromIndexFile(self, indexPath, metadataPath=None, defaultNamePattern="D{}"):
        pairRecords = self._readIndexFile(indexPath)
        metadata = self._readMetadataFile(metadataPath) if metadataPath else []

        # Metadata rows may include v0/v1. If so, allow them to override or fully define pairs.
        if metadata and all(("v0" in m and "v1" in m) for m in metadata):
            pairRecords = [
                {"v0": int(m["v0"]), "v1": int(m["v1"])}
                for m in metadata
            ]

        pairs = []
        for i, pr in enumerate(pairRecords):
            meta = metadata[i] if i < len(metadata) else {}
            name = meta.get("name") or pr.get("name") or self._formatDefaultName(defaultNamePattern, i + 1)
            desc = meta.get("description") or pr.get("description") or ""
            pairs.append(
                {
                    "name": str(name),
                    "v0": int(pr["v0"]),
                    "v1": int(pr["v1"]),
                    "length": None,
                    "description": str(desc or ""),
                }
            )

        data = {
            "version": 5,
            "type": "ETSE_UV_measurement_config",
            "modelName": "(from index file)",
            "markupNodeName": os.path.basename(indexPath),
            "sourceIndexFile": indexPath,
            "sourceMetadataFile": metadataPath or "",
            "distanceMode": "point_to_point",
            "pairs": pairs,
        }
        self._validateConfig(data)
        return data

    def _formatDefaultName(self, pattern, idx):
        try:
            if "{}" in pattern:
                return pattern.format(idx)
            if "{i}" in pattern:
                return pattern.format(i=idx)
            return f"{pattern}{idx}"
        except Exception:
            return f"D{idx}"

    def _readIndexFile(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            z = np.load(path, allow_pickle=True)
            keys = list(z.keys())
            # Common formats: pairs/indices Nx2, or v0/v1 arrays.
            if "pairs" in z:
                arr = z["pairs"]
            elif "indices" in z:
                arr = z["indices"]
            elif "vertex_pairs" in z:
                arr = z["vertex_pairs"]
            elif "v0" in z and "v1" in z:
                v0 = np.asarray(z["v0"]).ravel()
                v1 = np.asarray(z["v1"]).ravel()
                if len(v0) != len(v1):
                    raise RuntimeError("NPZ v0 and v1 arrays have different lengths.")
                return [{"v0": int(a), "v1": int(b)} for a, b in zip(v0, v1)]
            elif len(keys) == 1:
                arr = z[keys[0]]
            else:
                raise RuntimeError(
                    "NPZ must contain 'pairs', 'indices', 'vertex_pairs', v0/v1 arrays, or a single Nx2 array."
                )
            return self._arrayToPairRecords(arr)

        if ext == ".npy":
            arr = np.load(path, allow_pickle=True)
            # np.save(dict) gives a 0-d object array.
            if isinstance(arr, np.ndarray) and arr.shape == () and isinstance(arr.item(), dict):
                obj = arr.item()
                if "pairs" in obj:
                    return self._recordsFromPairList(obj["pairs"])
                if "indices" in obj:
                    return self._arrayToPairRecords(obj["indices"])
                if "v0" in obj and "v1" in obj:
                    return [
                        {"v0": int(a), "v1": int(b)}
                        for a, b in zip(np.asarray(obj["v0"]).ravel(), np.asarray(obj["v1"]).ravel())
                    ]
                raise RuntimeError("Dictionary .npy has no supported keys: expected pairs, indices, or v0/v1.")
            return self._arrayToPairRecords(arr)

        raise RuntimeError(f"Unsupported index file extension: {ext}")

    def _arrayToPairRecords(self, arr):
        arr = np.asarray(arr)

        if arr.dtype.fields:
            names = arr.dtype.names or []
            if "v0" in names and "v1" in names:
                records = []
                for row in arr:
                    rec = {"v0": int(row["v0"]), "v1": int(row["v1"])}
                    if "name" in names:
                        rec["name"] = str(row["name"])
                    if "description" in names:
                        rec["description"] = str(row["description"])
                    records.append(rec)
                return records

        if arr.dtype == object:
            # Could be list of dicts/tuples.
            try:
                return self._recordsFromPairList(arr.tolist())
            except Exception:
                pass

        arr = np.asarray(arr, dtype=np.int64)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [{"v0": int(row[0]), "v1": int(row[1])} for row in arr]
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
            return [{"v0": int(row[0]), "v1": int(row[1])} for row in arr]
        flat = arr.ravel()
        if len(flat) >= 2 and len(flat) % 2 == 0:
            return [{"v0": int(flat[i]), "v1": int(flat[i + 1])} for i in range(0, len(flat), 2)]

        raise RuntimeError("Index array must be Nx2, 2xN, or a flat even-length list of vertex IDs.")

    def _recordsFromPairList(self, pairList):
        records = []
        for item in pairList:
            if isinstance(item, dict):
                if "v0" in item and "v1" in item:
                    rec = {"v0": int(item["v0"]), "v1": int(item["v1"])}
                elif "indices" in item and len(item["indices"]) >= 2:
                    rec = {"v0": int(item["indices"][0]), "v1": int(item["indices"][1])}
                else:
                    raise RuntimeError("Pair dictionary needs v0/v1 or indices.")
                if item.get("name"):
                    rec["name"] = str(item.get("name"))
                if item.get("description"):
                    rec["description"] = str(item.get("description"))
                records.append(rec)
            else:
                seq = list(item)
                if len(seq) < 2:
                    raise RuntimeError("Every pair entry must contain at least two vertex IDs.")
                records.append({"v0": int(seq[0]), "v1": int(seq[1])})
        return records

    def _readMetadataFile(self, path):
        if not path:
            return []
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and "pairs" in obj:
                rows = obj["pairs"]
            elif isinstance(obj, list):
                rows = obj
            else:
                raise RuntimeError("Metadata JSON must be a list or a dict with a 'pairs' list.")
            out = []
            for r in rows:
                if isinstance(r, dict):
                    out.append({
                        "name": str(r.get("name", "")),
                        "description": str(r.get("description", "")),
                        **({"v0": int(r["v0"])} if "v0" in r else {}),
                        **({"v1": int(r["v1"])} if "v1" in r else {}),
                    })
                else:
                    seq = list(r)
                    out.append({"name": str(seq[0]) if len(seq) > 0 else "", "description": str(seq[1]) if len(seq) > 1 else ""})
            return out

        if ext in (".csv", ".txt"):
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t") if sample.strip() else csv.excel
                hasHeader = csv.Sniffer().has_header(sample) if sample.strip() else False
                rows = []
                if hasHeader:
                    reader = csv.DictReader(f, dialect=dialect)
                    for row in reader:
                        lower = {str(k).strip().lower(): (v or "") for k, v in row.items()}
                        rec = {
                            "name": lower.get("name", lower.get("distance", lower.get("label", ""))).strip(),
                            "description": lower.get("description", lower.get("desc", "")).strip(),
                        }
                        if lower.get("v0", "").strip() != "":
                            rec["v0"] = int(float(lower["v0"]))
                        if lower.get("v1", "").strip() != "":
                            rec["v1"] = int(float(lower["v1"]))
                        rows.append(rec)
                else:
                    reader = csv.reader(f, dialect=dialect)
                    for row in reader:
                        if not row:
                            continue
                        rec = {
                            "name": str(row[0]).strip() if len(row) > 0 else "",
                            "description": str(row[1]).strip() if len(row) > 1 else "",
                        }
                        if len(row) > 2 and str(row[2]).strip() != "":
                            rec["v0"] = int(float(row[2]))
                        if len(row) > 3 and str(row[3]).strip() != "":
                            rec["v1"] = int(float(row[3]))
                        rows.append(rec)
                return rows

        raise RuntimeError(f"Unsupported metadata file extension: {ext}")

    def _validateConfig(self, data):
        if not isinstance(data, dict):
            raise RuntimeError("Measurement config must be a JSON object.")
        if "pairs" not in data or not data["pairs"]:
            raise RuntimeError("Measurement config has no 'pairs' field with vertex indices.")
        for i, item in enumerate(data["pairs"]):
            if "v0" not in item or "v1" not in item:
                raise RuntimeError(f"Pair {i + 1} is missing v0/v1.")
            int(item["v0"])
            int(item["v1"])
        return True

    def _normalizeConfig(self, data):
        # Backwards compatible with version 4 JSON generated by the old tool.
        out = dict(data)
        out.setdefault("version", 5)
        out.setdefault("type", "ETSE_UV_measurement_config")
        out.setdefault("distanceMode", "point_to_point")
        pairs = []
        for i, item in enumerate(out["pairs"]):
            pairs.append(
                {
                    "name": str(item.get("name", f"D{i + 1}")),
                    "v0": int(item["v0"]),
                    "v1": int(item["v1"]),
                    "length": item.get("length", None),
                    "description": str(item.get("description", "") or ""),
                }
            )
        out["pairs"] = pairs
        return out

    # ------------------------------------------------------------------
    # Mesh graph / geodesic neighbourhoods
    # ------------------------------------------------------------------
    def _buildAdjacency(self, polyData, pointsNp):
        """Build weighted vertex adjacency from mesh cells. Edge weights are Euclidean lengths."""
        n = int(polyData.GetNumberOfPoints())
        adjacency = [dict() for _ in range(n)]
        idList = vtk.vtkIdList()

        for ci in range(polyData.GetNumberOfCells()):
            polyData.GetCellPoints(ci, idList)
            m = idList.GetNumberOfIds()
            if m < 2:
                continue
            ids = [int(idList.GetId(k)) for k in range(m)]
            edges = []
            for k in range(m - 1):
                edges.append((ids[k], ids[k + 1]))
            # Close polygonal cells; keep 2-point line cells open.
            if m > 2:
                edges.append((ids[-1], ids[0]))

            for a, b in edges:
                if a == b or a < 0 or b < 0 or a >= n or b >= n:
                    continue
                w = float(np.linalg.norm(pointsNp[a] - pointsNp[b]))
                if b not in adjacency[a] or w < adjacency[a][b]:
                    adjacency[a][b] = w
                    adjacency[b][a] = w

        return adjacency

    def _geodesicNeighborhood(self, centerId, radiusMm, adjacency, cache=None):
        centerId = int(centerId)
        radiusMm = float(radiusMm)
        if radiusMm <= 0.0:
            return [centerId]

        key = (centerId, round(radiusMm, 6))
        if cache is not None and key in cache:
            return cache[key]

        if centerId < 0 or centerId >= len(adjacency):
            raise RuntimeError(f"Center vertex {centerId} is out of range.")

        # No connectivity available -> fall back to the center point only.
        if not adjacency or not adjacency[centerId]:
            result = [centerId]
            if cache is not None:
                cache[key] = result
            return result

        dist = {centerId: 0.0}
        heap = [(0.0, centerId)]
        result = []

        while heap:
            d, u = heapq.heappop(heap)
            if d > radiusMm:
                break
            if d != dist.get(u, None):
                continue
            result.append(u)
            for v, w in adjacency[u].items():
                nd = d + w
                if nd <= radiusMm and nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        if not result:
            result = [centerId]
        result = sorted(set(int(x) for x in result))
        if cache is not None:
            cache[key] = result
        return result

    def _meanPairwiseDistance(self, pts0, pts1):
        n0 = int(len(pts0))
        n1 = int(len(pts1))
        if n0 == 0 or n1 == 0:
            return float("nan")
        # Chunked to avoid exploding memory if a large radius is used.
        total = 0.0
        count = 0
        chunk = 512
        for i0 in range(0, n0, chunk):
            a = pts0[i0:i0 + chunk]
            d = np.linalg.norm(a[:, None, :] - pts1[None, :, :], axis=2)
            total += float(np.sum(d))
            count += int(d.size)
        return total / float(count) if count else float("nan")

    def _measureOnePair(self, pointsNp, adjacency, neighCache, item, radiusMm=0.0, useZoneMean=False):
        v0 = int(item["v0"])
        v1 = int(item["v1"])
        nPoints = int(pointsNp.shape[0])
        if v0 < 0 or v0 >= nPoints or v1 < 0 or v1 >= nPoints:
            raise RuntimeError(f"Vertex indices out of range: ({v0}, {v1}) with {nPoints} points.")

        p0 = pointsNp[v0]
        p1 = pointsNp[v1]
        p2p = float(np.linalg.norm(p0 - p1))

        if (not useZoneMean) or float(radiusMm) <= 0.0:
            return {
                "mode": "point_to_point",
                "name": str(item.get("name", "")),
                "description": str(item.get("description", "") or ""),
                "v0": v0,
                "v1": v1,
                "distance_mm": p2p,
                "point_to_point_mm": p2p,
                "centroid_distance_mm": p2p,
                "zone0_n": 1,
                "zone1_n": 1,
                "zone_radius_mm": 0.0,
                "zone0_ids": [v0],
                "zone1_ids": [v1],
                "point_p0": p0,
                "point_p1": p1,
                "centroid_p0": p0,
                "centroid_p1": p1,
                "draw_p0": p0,
                "draw_p1": p1,
            }

        z0 = self._geodesicNeighborhood(v0, radiusMm, adjacency, cache=neighCache)
        z1 = self._geodesicNeighborhood(v1, radiusMm, adjacency, cache=neighCache)
        pts0 = pointsNp[z0]
        pts1 = pointsNp[z1]
        c0 = np.mean(pts0, axis=0)
        c1 = np.mean(pts1, axis=0)
        meanDist = self._meanPairwiseDistance(pts0, pts1)
        centroidDist = float(np.linalg.norm(c0 - c1))

        return {
            "mode": "zone_mean",
            "name": str(item.get("name", "")),
            "description": str(item.get("description", "") or ""),
            "v0": v0,
            "v1": v1,
            "distance_mm": float(meanDist),
            "point_to_point_mm": p2p,
            "centroid_distance_mm": centroidDist,
            "zone0_n": int(len(z0)),
            "zone1_n": int(len(z1)),
            "zone_radius_mm": float(radiusMm),
            "zone0_ids": z0,
            "zone1_ids": z1,
            "point_p0": p0,
            "point_p1": p1,
            "centroid_p0": c0,
            "centroid_p1": c1,
            "draw_p0": c0,
            "draw_p1": c1,
        }

    def measureFromConfig(self, modelNode, data, useZoneMean=False, radiusMm=0.0):
        self._validateConfig(data)
        data = self._normalizeConfig(data)
        polyWorld = self._polyDataInWorld(modelNode)
        pointsWorld = polyWorld.GetPoints()
        pointsNp = self._pointsToNumpy(pointsWorld)

        adjacency = None
        neighCache = {}
        if useZoneMean and float(radiusMm) > 0.0:
            adjacency = self._buildAdjacency(polyWorld, pointsNp)
        else:
            adjacency = []

        results = []
        for idx, item in enumerate(data["pairs"]):
            item = dict(item)
            item.setdefault("name", f"D{idx + 1}")
            results.append(
                self._measureOnePair(
                    pointsNp=pointsNp,
                    adjacency=adjacency,
                    neighCache=neighCache,
                    item=item,
                    radiusMm=float(radiusMm),
                    useZoneMean=bool(useZoneMean),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Interactive measurement creation
    # ------------------------------------------------------------------
    def _prefixedMeasurementName(self, prefix, baseName):
        """Return a clean node name: prefix + baseName, without ugly duplicates.

        Example: prefix='D_' and baseName='D1' -> 'D1' instead of 'D_D1'.
        No measured numeric value is ever appended to the name.
        """
        baseName = str(baseName or "Distance").strip()
        prefix = str(prefix or "").strip()
        if not prefix:
            return baseName

        prefixStem = prefix[:-1] if prefix.endswith("_") else prefix
        if prefixStem and baseName.startswith(prefixStem):
            return baseName
        return f"{prefix}{baseName}"

    def _createMarkupLineNode(self, name, p0, p1, description=""):
        p0 = [float(x) for x in p0]
        p1 = [float(x) for x in p1]

        lineNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
        lineNode.AddControlPointWorld(vtk.vtkVector3d(*p0))
        lineNode.AddControlPointWorld(vtk.vtkVector3d(*p1))
        lineNode.CreateDefaultDisplayNodes()

        if description:
            try:
                lineNode.SetDescription(description)
            except Exception:
                pass
            try:
                lineNode.SetNthControlPointDescription(0, description)
                lineNode.SetNthControlPointDescription(1, description)
            except Exception:
                pass
        return lineNode

    def createMeasurementsFromConfig(
        self,
        modelNode,
        data,
        prefix="",
        useZoneMean=False,
        radiusMm=0.0,
        highlightZones=False,
        showCentroidLines=False,
    ):
        results = self.measureFromConfig(modelNode, data, useZoneMean=useZoneMean, radiusMm=radiusMm)

        for r in results:
            baseName = str(r.get("name") or "Distance")
            p2pNodeName = self._prefixedMeasurementName(prefix, baseName)

            desc = r.get("description", "") or ""
            extra = (
                f"displayed_distance=point_to_point; "
                f"point_to_point_mm={r['point_to_point_mm']:.6f}; "
                f"zone_mean_mm={r['distance_mm']:.6f}; "
                f"centroid_distance_mm={r['centroid_distance_mm']:.6f}; "
                f"mode={r['mode']}; "
                f"zone_radius_mm={r['zone_radius_mm']:.6f}; "
                f"zone0_n={r['zone0_n']}; zone1_n={r['zone1_n']}"
            )
            fullDesc = (desc + " | " + extra).strip(" |")

            # Default visual output: original saved vertex-to-vertex distance.
            p2pLineNode = self._createMarkupLineNode(
                p2pNodeName,
                r.get("point_p0", r.get("draw_p0")),
                r.get("point_p1", r.get("draw_p1")),
                description=fullDesc,
            )
            r["line_node_name"] = p2pLineNode.GetName()

            # Optional helper: draw what zone mode uses as the centroid-to-centroid visual line.
            if showCentroidLines and r.get("mode") == "zone_mean":
                centroidName = f"{p2pNodeName}_centroid"
                centroidDesc = (
                    f"centroid_to_centroid_mm={r['centroid_distance_mm']:.6f}; "
                    f"zone_mean_mm={r['distance_mm']:.6f}; "
                    f"point_to_point_mm={r['point_to_point_mm']:.6f}; "
                    f"zone_radius_mm={r['zone_radius_mm']:.6f}; "
                    f"zone0_n={r['zone0_n']}; zone1_n={r['zone1_n']}"
                )
                centroidLineNode = self._createMarkupLineNode(
                    centroidName,
                    r.get("centroid_p0", r.get("draw_p0")),
                    r.get("centroid_p1", r.get("draw_p1")),
                    description=centroidDesc,
                )
                r["centroid_line_node_name"] = centroidLineNode.GetName()
            else:
                r["centroid_line_node_name"] = ""

            # Keep r['name'] as the clean measurement name, e.g. D1, not T_D1 and not D1_mean_18.2mm.
            r["name"] = baseName

        if highlightZones and useZoneMean and float(radiusMm) > 0.0:
            self.createZoneHighlightModels(modelNode, results, prefix=prefix)

        return results

    # Backwards-compatible names used by the old UI/API
    def createLinesFromVertexPairs(self, modelNode, data, prefix=""):
        results = self.createMeasurementsFromConfig(modelNode, data, prefix=prefix, useZoneMean=False, radiusMm=0.0)
        return [(r.get("line_node_name", r["name"]), r["point_to_point_mm"]) for r in results]

    def computeDistancesFromVertexPairs(self, modelNode, data):
        results = self.measureFromConfig(modelNode, data, useZoneMean=False, radiusMm=0.0)
        return [(r["name"], r["distance_mm"]) for r in results]

    def createZoneHighlightModels(self, modelNode, results, prefix=""):
        """Create aggregate point-cloud nodes for all zone points used as endpoint-0 and endpoint-1 areas."""
        polyWorld = self._polyDataInWorld(modelNode)
        pointsNp = self._pointsToNumpy(polyWorld.GetPoints())

        ids0 = sorted(set(i for r in results for i in r.get("zone0_ids", [])))
        ids1 = sorted(set(i for r in results for i in r.get("zone1_ids", [])))

        created = []
        if ids0:
            n0 = self._createPointCloudModel(pointsNp[ids0], f"{prefix}Measurement_zone0_points", color=(0.1, 0.4, 1.0))
            created.append(n0)
        if ids1:
            n1 = self._createPointCloudModel(pointsNp[ids1], f"{prefix}Measurement_zone1_points", color=(1.0, 0.45, 0.05))
            created.append(n1)
        return created

    def _createPointCloudModel(self, ptsNp, name, color=(1.0, 0.4, 0.0)):
        vtkPts = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        for i, p in enumerate(ptsNp):
            vtkPts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)

        pd = vtk.vtkPolyData()
        pd.SetPoints(vtkPts)
        pd.SetVerts(verts)

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.SetAndObservePolyData(pd)
        self._makeDisplayNode(node, color=color, opacity=1.0, pointSize=6)
        return node

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def runBatch(self, data, folder, csvPath, useZoneMean=False, radiusMm=0.0, includeExtraStats=True):
        self._validateConfig(data)
        data = self._normalizeConfig(data)

        exts = {".stl", ".ply", ".vtk", ".vtp", ".obj", ".gltf", ".glb"}
        files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
        files.sort()

        if not files:
            raise RuntimeError("No mesh files with supported extensions in folder.")

        distNames = [item.get("name", f"D{i + 1}") for i, item in enumerate(data["pairs"])]

        header = ["Model"]
        for dn in distNames:
            # Main exported measurement is always the original saved vertex-to-vertex distance.
            header.append(f"{dn}__mm")
            if includeExtraStats:
                header.append(f"{dn}__zone_mean_mm")
                header.append(f"{dn}__centroid_mm")
                header.append(f"{dn}__zone0_n")
                header.append(f"{dn}__zone1_n")
        header.append("mode")
        header.append("zone_radius_mm")

        processed = 0
        written = 0
        with open(csvPath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for fname in files:
                fullPath = os.path.join(folder, fname)
                print(f"[ETSE_UV_MeasurementTransfer] Processing: {fullPath}")
                processed += 1

                modelNode = slicer.util.loadModel(fullPath)
                if modelNode is None:
                    print("[ETSE_UV_MeasurementTransfer]  -> Failed to load as model, skipping.")
                    continue

                try:
                    results = self.measureFromConfig(
                        modelNode=modelNode,
                        data=data,
                        useZoneMean=bool(useZoneMean),
                        radiusMm=float(radiusMm),
                    )
                    resDict = {r["name"]: r for r in results}
                    row = [os.path.splitext(fname)[0]]
                    for dn in distNames:
                        r = resDict.get(dn)
                        if r is None:
                            row.append("")
                            if includeExtraStats:
                                row.extend(["", "", "", ""])
                            continue
                        # Main CSV value is point-to-point, even when zone statistics are enabled.
                        row.append(f"{r['point_to_point_mm']:.6f}")
                        if includeExtraStats:
                            if r.get("mode") == "zone_mean":
                                row.append(f"{r['distance_mm']:.6f}")
                                row.append(f"{r['centroid_distance_mm']:.6f}")
                                row.append(str(int(r["zone0_n"])))
                                row.append(str(int(r["zone1_n"])))
                            else:
                                row.extend(["", "", "", ""])
                    row.append("zone_mean" if (useZoneMean and float(radiusMm) > 0.0) else "point_to_point")
                    row.append(f"{float(radiusMm):.6f}" if useZoneMean else "0.000000")
                    writer.writerow(row)
                    written += 1
                except Exception as e:
                    print(f"[ETSE_UV_MeasurementTransfer]  -> Error computing distances: {e}")
                finally:
                    try:
                        slicer.mrmlScene.RemoveNode(modelNode)
                    except Exception:
                        pass

        print(f"[ETSE_UV_MeasurementTransfer] Batch done. CSV: {csvPath}")
        return processed, written
