# ETSE_UV__MeshPathTools.py
# 3D Slicer scripted module
#
# Tools:
#   1) Generate markups every Nth vertex of a mesh.
#   2) Draw a straight-line path between ordered vertices of a mesh.
#
# No geodesics, just vertex sampling and straight segments.

import os
import numpy as np
import slicer
import vtk
import qt
import ctk
from slicer.ScriptedLoadableModule import *
from slicer.util import NodeModify


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------
class ETSE_UV__MeshPathTools(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Mesh Path Tools"
        parent.categories = ["ETSE_UV"]
        parent.dependencies = []
        parent.contributors = ["ETSE-UV"]
        parent.helpText = (
            "Collection of simple mesh/markup tools:\n\n"
            "  • Generate markups at every Nth vertex in a mesh.\n"
            "  • Draw a straight-line path between ordered vertices of a mesh, "
            "optionally closing the loop.\n\n"
            "No geodesics are computed: paths are purely straight segments between "
            "the selected vertices."
        )
        parent.acknowledgementText = (
            "This module combines tools developed by J.A. De Rus and the ETSE-UV team."
        )


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class ETSE_UV__MeshPathToolsWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = ETSE_UV__MeshPathToolsLogic()

        # ------------------------------------------------------------------
        # Section 1: Generate markups every Nth vertex
        # ------------------------------------------------------------------
        genBox = ctk.ctkCollapsibleButton()
        genBox.text = "Generate markups every Nth vertex"
        self.layout.addWidget(genBox)
        genLayout = qt.QFormLayout(genBox)

        # Model selector
        self.genModelSelector = slicer.qMRMLNodeComboBox()
        self.genModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.genModelSelector.selectNodeUponCreation = True
        self.genModelSelector.addEnabled = False
        self.genModelSelector.removeEnabled = False
        self.genModelSelector.noneEnabled = False
        self.genModelSelector.setMRMLScene(slicer.mrmlScene)
        self.genModelSelector.setToolTip("Pick the mesh model where markups will be generated.")
        genLayout.addRow("Model:", self.genModelSelector)

        # Step size slider (relaxed limits)
        self.stepSizeSlider = ctk.ctkSliderWidget()
        self.stepSizeSlider.singleStep = 1
        self.stepSizeSlider.minimum = 1
        self.stepSizeSlider.maximum = 100000
        self.stepSizeSlider.value = 20
        self.stepSizeSlider.setToolTip(
            "Step size N. A markup will be placed at vertices 0, N, 2N, ...\n"
            "If N is larger than the number of vertices, only the first vertex is used."
        )
        genLayout.addRow("Step size (N):", self.stepSizeSlider)

        # Apply button
        self.genApplyButton = qt.QPushButton("Generate markups")
        self.genApplyButton.toolTip = "Generate fiducials at every Nth vertex of the selected model."
        self.genApplyButton.connect("clicked(bool)", self.onGenApply)
        genLayout.addRow(self.genApplyButton)

        # ------------------------------------------------------------------
        # Section 2: Draw path between vertices
        # ------------------------------------------------------------------
        pathBox = ctk.ctkCollapsibleButton()
        pathBox.text = "Draw path between vertices"
        self.layout.addWidget(pathBox)
        pathLayout = qt.QFormLayout(pathBox)

        # Model selector
        self.pathModelSelector = slicer.qMRMLNodeComboBox()
        self.pathModelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.pathModelSelector.selectNodeUponCreation = True
        self.pathModelSelector.addEnabled = False
        self.pathModelSelector.removeEnabled = False
        self.pathModelSelector.noneEnabled = False
        self.pathModelSelector.setMRMLScene(slicer.mrmlScene)
        self.pathModelSelector.setToolTip("Pick the mesh model (triangle polydata) whose vertices you want to connect.")
        pathLayout.addRow("Input model:", self.pathModelSelector)

        # Vertex specification
        self.specEdit = qt.QLineEdit("all")
        self.specEdit.setToolTip(
            "Specify vertices to connect, using 0-based or 1-based indices.\n"
            "Examples:\n"
            "  all\n"
            "  0-10\n"
            "  1..n   (if 1-based)\n"
            "  5-50:5 (step 5)\n"
            "  0,2,4,8  or  1-20, 30, 40-50\n"
            "  1 to n\n"
        )
        pathLayout.addRow("Vertex spec:", self.specEdit)

        # Index base toggle
        self.oneBasedCheck = qt.QCheckBox("Indices are 1-based")
        self.oneBasedCheck.checked = False
        pathLayout.addRow(self.oneBasedCheck)

        # Options
        self.closeLoopCheck = qt.QCheckBox("Close path (connect last → first)")
        self.closeLoopCheck.checked = False
        pathLayout.addRow(self.closeLoopCheck)

        self.gradientCheck = qt.QCheckBox("Color path with gradient (per-segment)")
        self.gradientCheck.checked = True
        pathLayout.addRow(self.gradientCheck)

        # Apply button
        self.pathApplyButton = qt.QPushButton("Draw path")
        self.pathApplyButton.toolTip = (
            "Create a polyline model that connects the selected vertices with straight segments.\n"
            "No geodesics or connectivity information is used."
        )
        self.pathApplyButton.connect("clicked(bool)", self.onPathApply)
        pathLayout.addRow(self.pathApplyButton)

        self.layout.addStretch(1)

    # ------------------------------------------------------------------
    # Markup generator
    # ------------------------------------------------------------------
    def onGenApply(self):
        modelNode = self.genModelSelector.currentNode()
        stepSize = int(self.stepSizeSlider.value)

        if not modelNode:
            slicer.util.errorDisplay("Please select a model for markup generation.")
            return

        if stepSize < 1:
            slicer.util.errorDisplay("Step size N must be at least 1.")
            return

        try:
            self.logic.generateMarkupsEveryN(modelNode, stepSize)
        except Exception as e:
            slicer.util.errorDisplay(f"Error generating markups:\n{e}")

    # ------------------------------------------------------------------
    # Path between vertices
    # ------------------------------------------------------------------
    def _parseSpec(self, text, nPoints, one_based=False):
        """Parse vertex specification string into list of 0-based indices."""
        def to_int(tok):
            t = tok.strip().lower()
            if t == "n":
                return nPoints if one_based else (nPoints - 1)
            return int(t)

        expr = text.replace("\n", " ").replace("\t", " ")
        expr = expr.replace(" to ", "-").replace("..", "-")
        parts = [p for p in expr.replace(",", " ").split(" ") if p]

        out = []
        if len(parts) == 1 and parts[0].lower() == "all":
            out = list(range(1, nPoints + 1)) if one_based else list(range(0, nPoints))
        else:
            for tok in parts:
                t = tok.strip()
                if t.lower() == "all":
                    out.extend(list(range(1, nPoints + 1)) if one_based else list(range(0, nPoints)))
                    continue
                if "-" in t:
                    core, step_str = (t.split(":", 1) + ["1"])[:2]
                    a_str, b_str = core.split("-", 1)
                    a = to_int(a_str)
                    b = to_int(b_str)
                    step = int(step_str)
                    if step == 0:
                        raise ValueError("Step cannot be 0 in vertex spec.")
                    seq = range(a, b + (1 if b >= a else -1), step if b >= a else -step)
                    out.extend(seq)
                else:
                    out.append(to_int(t))

        if one_based:
            out = [v - 1 for v in out]

        cleaned = []
        for v in out:
            if v < 0 or v >= nPoints:
                raise ValueError(f"Vertex id {v if not one_based else v + 1} out of range (0..{nPoints - 1}  or  1..{nPoints}).")
            if len(cleaned) == 0 or cleaned[-1] != v:
                cleaned.append(int(v))
        if len(cleaned) < 2:
            raise ValueError("Need at least two vertices after parsing spec.")
        return cleaned

    def onPathApply(self):
        try:
            modelNode = self.pathModelSelector.currentNode()
            if not modelNode:
                slicer.util.errorDisplay("Please select a model for the path.")
                return

            nPoints = modelNode.GetPolyData().GetNumberOfPoints()
            if nPoints == 0:
                slicer.util.errorDisplay("Input model has no points.")
                return

            ids = self._parseSpec(self.specEdit.text, nPoints, one_based=bool(self.oneBasedCheck.checked))
            if self.closeLoopCheck.checked:
                ids = ids + [ids[0]]

            outModelNode = self.logic.drawPath(modelNode, ids, colorGradient=self.gradientCheck.checked)

            if outModelNode and outModelNode.GetDisplayNode():
                dn = outModelNode.GetDisplayNode()
                dn.SetColor(1.0, 0.2, 0.2)
                dn.SetLineWidth(3)
                dn.SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation)
                dn.SetPointSize(6)

            slicer.util.infoDisplay(
                f"Created path model: {outModelNode.GetName()}   Segments: {max(0, len(ids) - 1)}",
                windowTitle="Path Between Vertices",
            )
        except Exception as e:
            slicer.util.errorDisplay(str(e))


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class ETSE_UV__MeshPathToolsLogic(ScriptedLoadableModuleLogic):

    def generateMarkupsEveryN(self, modelNode, stepSize):
        polyData = modelNode.GetPolyData()
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise RuntimeError("Input model has no points.")

        points = polyData.GetPoints()
        numPoints = points.GetNumberOfPoints()

        markupNodeName = modelNode.GetName() + "___GeneratedMarkups"
        markupNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", markupNodeName)

        with NodeModify(markupNode):
            counter = 0
            for i in range(0, numPoints, stepSize):
                point = [0.0, 0.0, 0.0]
                points.GetPoint(i, point)
                markupNode.AddControlPoint(point, f"{counter}")
                counter += 1

        slicer.util.infoDisplay(
            f"Markups created with step size {stepSize}. Total points: {counter}",
            windowTitle="Markup Generator",
        )

    def drawPath(self, modelNode, vertexIds, colorGradient=True):
        poly = modelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Input model has no points.")

        n = poly.GetNumberOfPoints()
        cleaned = []
        for vid in vertexIds:
            if vid < 0 or vid >= n:
                raise RuntimeError(f"Vertex id {vid} out of range (0..{n - 1}).")
            if len(cleaned) == 0 or cleaned[-1] != vid:
                cleaned.append(int(vid))
        if len(cleaned) < 2:
            raise RuntimeError("After cleaning duplicates, need at least two distinct vertices.")

        pathPoints = vtk.vtkPoints()
        pathPoints.SetNumberOfPoints(len(cleaned))
        for i, pid in enumerate(cleaned):
            p = [0.0, 0.0, 0.0]
            poly.GetPoint(pid, p)
            pathPoints.SetPoint(i, p)

        outPD = vtk.vtkPolyData()

        if colorGradient:
            cells = vtk.vtkCellArray()
            scalars = vtk.vtkIntArray()
            scalars.SetName("SegmentId")
            numSeg = len(cleaned) - 1
            for i in range(numSeg):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, i + 1)
                cells.InsertNextCell(line)
                scalars.InsertNextValue(i)
            outPD.SetPoints(pathPoints)
            outPD.SetLines(cells)
            outPD.GetCellData().SetScalars(scalars)
        else:
            polyLine = vtk.vtkPolyLine()
            polyLine.GetPointIds().SetNumberOfIds(len(cleaned))
            for i in range(len(cleaned)):
                polyLine.GetPointIds().SetId(i, i)
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(polyLine)
            outPD.SetPoints(pathPoints)
            outPD.SetLines(cells)

        outPD.Modified()

        outName = f"Path_{modelNode.GetName()}"
        outModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", outName)
        outModel.SetAndObservePolyData(outPD)
        outModel.CreateDefaultDisplayNodes()

        dn = outModel.GetDisplayNode()
        if dn:
            dn.SetColor(1.0, 0.2, 0.2)
            dn.SetLineWidth(3)
            dn.SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation)
            dn.SetPointSize(6)
            if colorGradient:
                try:
                    colorNode = slicer.util.getNode("vtkMRMLColorTableNodeRainbow")
                except Exception:
                    colorNode = None
                if colorNode:
                    dn.SetAndObserveColorNodeID(colorNode.GetID())
                outPD.GetCellData().SetActiveScalars("SegmentId")
                dn.SetScalarVisibility(True)
                numSeg = max(1, (len(cleaned) - 1))
                dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
                dn.SetScalarRange(0, numSeg - 1)

        return outModel
