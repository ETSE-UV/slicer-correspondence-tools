import os
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import *
from Resources.ETSE_UV__Dependencies import ensure_packages

# ------------------------------------------------------------
# Module (metadata)
# ------------------------------------------------------------
class ETSE_UV__MeshOrderer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Mesh Orderer"
        parent.categories = ["ETSE_UV"]
        parent.contributors = ["ETSE-UV"]
        parent.helpText = (
            "Order mesh points using a 1D embedding and create: "
            "(1) a polyline path that connects points in the new order, and "
            "(2) an optional copy of the model with points renumbered and cell connectivity remapped. "
            "Methods: t-SNE, PCA, or Trimesh TSP (nearest-neighbor traversal)."
        )
        parent.acknowledgementText = "Developed by J.A. De Rus at ETSE-UV"

# ------------------------------------------------------------
# Widget (UI)
# ------------------------------------------------------------
class ETSE_UV__MeshOrdererWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = ETSE_UV__MeshOrdererLogic()

        # ------------------------------------------------------------
        # Inputs
        # ------------------------------------------------------------
        inputs = ctk.ctkCollapsibleButton()
        inputs.text = "Inputs"
        self.layout.addWidget(inputs)
        inForm = qt.QFormLayout(inputs)

        self.modelSelector = slicer.qMRMLNodeComboBox()
        self.modelSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.modelSelector.selectNodeUponCreation = True
        self.modelSelector.addEnabled = False
        self.modelSelector.removeEnabled = False
        self.modelSelector.noneEnabled = False
        self.modelSelector.setMRMLScene(slicer.mrmlScene)
        self.modelSelector.setToolTip("Pick the mesh model (polydata)")
        inForm.addRow("Input model:", self.modelSelector)

        # ------------------------------------------------------------
        # ORDER: Embedding / Ordering method
        # ------------------------------------------------------------
        methodBox = ctk.ctkCollapsibleButton()
        methodBox.text = "ORDER settings (compute vertex ordering)"
        self.layout.addWidget(methodBox)
        mForm = qt.QFormLayout(methodBox)

        self.methodCombo = qt.QComboBox()
        self.methodCombo.addItems(["t-SNE", "PCA (fallback)", "Trimesh TSP"])
        mForm.addRow("Ordering method:", self.methodCombo)

        # t-SNE parameters
        self.perplexitySpin = qt.QDoubleSpinBox()
        self.perplexitySpin.setRange(5.0, 100.0)
        self.perplexitySpin.setValue(30.0)
        self.perplexitySpin.setSingleStep(1.0)
        mForm.addRow("t-SNE perplexity:", self.perplexitySpin)

        self.nIterSpin = qt.QSpinBox()
        self.nIterSpin.setRange(250, 100000)
        self.nIterSpin.setValue(1000)
        self.nIterSpin.setSingleStep(250)
        mForm.addRow("t-SNE iterations:", self.nIterSpin)

        self.lrAutoCheck = qt.QCheckBox("Learning rate: auto")
        self.lrAutoCheck.checked = True
        mForm.addRow(self.lrAutoCheck)

        self.lrSpin = qt.QDoubleSpinBox()
        self.lrSpin.setRange(10.0, 2000.0)
        self.lrSpin.setValue(200.0)
        self.lrSpin.setSingleStep(10.0)
        self.lrSpin.setEnabled(False)
        mForm.addRow("Manual learning rate:", self.lrSpin)
        self.lrAutoCheck.toggled.connect(lambda on: self.lrSpin.setEnabled(not on))

        self.initCombo = qt.QComboBox()
        self.initCombo.addItems(["pca", "random"])
        self.initCombo.setCurrentText("pca")
        mForm.addRow("t-SNE init:", self.initCombo)

        self.seedSpin = qt.QSpinBox()
        self.seedSpin.setRange(0, 10000000)
        self.seedSpin.setValue(42)
        mForm.addRow("t-SNE random seed:", self.seedSpin)

        # Trimesh TSP parameter
        self.tspStartSpin = qt.QSpinBox()
        self.tspStartSpin.setRange(0, 100000000)
        self.tspStartSpin.setValue(0)
        mForm.addRow("TSP start index (0-based):", self.tspStartSpin)

        # ------------------------------------------------------------
        # Batch folder settings (used by ORDER batch and DISORDER batch)
        # ------------------------------------------------------------
        batchBox = ctk.ctkCollapsibleButton()
        batchBox.text = "Batch folder settings"
        self.layout.addWidget(batchBox)
        bForm = qt.QFormLayout(batchBox)

        rowIn = qt.QHBoxLayout()
        self.inputDirEdit = qt.QLineEdit("")
        btnIn = qt.QPushButton("Browse…")
        btnIn.clicked.connect(self.onBrowseInput)
        rowIn.addWidget(self.inputDirEdit)
        rowIn.addWidget(btnIn)
        wIn = qt.QWidget()
        wIn.setLayout(rowIn)
        bForm.addRow("Input folder (.vtk/.ply):", wIn)

        rowOut = qt.QHBoxLayout()
        self.outputDirEdit = qt.QLineEdit("")
        btnOut = qt.QPushButton("Browse…")
        btnOut.clicked.connect(self.onBrowseOutput)
        rowOut.addWidget(self.outputDirEdit)
        rowOut.addWidget(btnOut)
        wOut = qt.QWidget()
        wOut.setLayout(rowOut)
        bForm.addRow("Output folder:", wOut)

        self.makeSubdirCheck = qt.QCheckBox("Create subfolder inside output folder")
        self.makeSubdirCheck.checked = True
        bForm.addRow(self.makeSubdirCheck)

        self.subdirNameEdit = qt.QLineEdit("Reindexed")
        bForm.addRow("Subfolder name:", self.subdirNameEdit)

        # ------------------------------------------------------------
        # Output options (apply to both ORDER and DISORDER)
        # ------------------------------------------------------------
        outBox = ctk.ctkCollapsibleButton()
        outBox.text = "Output options (apply to ORDER and DISORDER)"
        self.layout.addWidget(outBox)
        oForm = qt.QFormLayout(outBox)

        self.makePolylineCheck = qt.QCheckBox("Create polyline path following the current order")
        self.makePolylineCheck.checked = True
        oForm.addRow(self.makePolylineCheck)

        self.gradientCheck = qt.QCheckBox("Color path with gradient (per-segment)")
        self.gradientCheck.checked = True
        oForm.addRow(self.gradientCheck)

        self.remapModelCheck = qt.QCheckBox("Create new model with reordered point IDs (connectivity remapped)")
        self.remapModelCheck.checked = True
        oForm.addRow(self.remapModelCheck)

        # ------------------------------------------------------------
        # Save / load ordering+topology (.npz)
        # ------------------------------------------------------------
        saveBox = ctk.ctkCollapsibleButton()
        saveBox.text = "Save / load ordering+topology (.npz)"
        self.layout.addWidget(saveBox)
        sLayout = qt.QVBoxLayout(saveBox)

        self.storedOrderLabel = qt.QLabel("Stored ordering: none")
        sLayout.addWidget(self.storedOrderLabel)

        btnRow = qt.QHBoxLayout()
        self.saveOrderButton = qt.QPushButton("Save ordering+topology")
        self.loadOrderButton = qt.QPushButton("Load ordering+topology")
        btnRow.addWidget(self.saveOrderButton)
        btnRow.addWidget(self.loadOrderButton)
        sLayout.addLayout(btnRow)

        self.saveOrderButton.toolTip = "Save last computed vertex order and remapped connectivity to a .npz file."
        self.loadOrderButton.toolTip = "Load a previously saved ordering+topology .npz file."

        self.saveOrderButton.connect('clicked(bool)', self.onSaveOrderingTopology)
        self.loadOrderButton.connect('clicked(bool)', self.onLoadOrderingTopology)

        self._updateStoredOrderLabel()

        # ------------------------------------------------------------
        # DISORDER settings (seed only)
        # ------------------------------------------------------------
        disBox = ctk.ctkCollapsibleButton()
        disBox.text = "DISORDER settings (random shuffle)"
        self.layout.addWidget(disBox)
        dForm = qt.QFormLayout(disBox)

        self.disSeedCheck = qt.QCheckBox("Use fixed seed (repeatable)")
        self.disSeedCheck.checked = False
        dForm.addRow(self.disSeedCheck)

        self.disSeedSpin = qt.QSpinBox()
        self.disSeedSpin.setRange(0, 2**31 - 1)
        self.disSeedSpin.setValue(12345)
        self.disSeedSpin.setEnabled(False)
        dForm.addRow("Seed:", self.disSeedSpin)

        self.disSeedCheck.toggled.connect(lambda on: self.disSeedSpin.setEnabled(bool(on)))

        # ------------------------------------------------------------
        # DECIMATE settings
        # ------------------------------------------------------------
        decBox = ctk.ctkCollapsibleButton()
        decBox.text = "DECIMATE settings (reduce mesh to N points)"
        self.layout.addWidget(decBox)
        decForm = qt.QFormLayout(decBox)

        self.decTargetPointsSpin = qt.QSpinBox()
        self.decTargetPointsSpin.setRange(10, 100000000)
        self.decTargetPointsSpin.setValue(5000)
        decForm.addRow("Target points (N):", self.decTargetPointsSpin)

        # ------------------------------------------------------------
        # Actions (buttons)
        # ------------------------------------------------------------
        actionsBox = ctk.ctkCollapsibleButton()
        actionsBox.text = "Actions"
        self.layout.addWidget(actionsBox)
        aLayout = qt.QVBoxLayout(actionsBox)

        # Single actions row
        row1 = qt.QHBoxLayout()
        self.applyButton = qt.QPushButton("ORDER selected model (compute)")
        self.disorderSingleButton = qt.QPushButton("DISORDER selected model (random shuffle)")
        row1.addWidget(self.applyButton)
        row1.addWidget(self.disorderSingleButton)
        w1 = qt.QWidget()
        w1.setLayout(row1)
        aLayout.addWidget(w1)

        # Batch actions row
        row2 = qt.QHBoxLayout()
        self.batchButton = qt.QPushButton("ORDER folder (batch)")
        self.disorderBatchButton = qt.QPushButton("DISORDER folder (batch)")
        row2.addWidget(self.batchButton)
        row2.addWidget(self.disorderBatchButton)
        w2 = qt.QWidget()
        w2.setLayout(row2)
        aLayout.addWidget(w2)

        # Tooltips
        self.applyButton.toolTip = (
            "Compute ordering using the selected method and create outputs according to Output options."
        )
        self.batchButton.toolTip = (
            "Process all meshes in the input folder and save to output.\n"
            "If an .npz topology is loaded, it applies that. Otherwise it uses the last computed ordering."
        )
        self.disorderSingleButton.toolTip = (
            "Randomly reshuffle vertex indices and remap connectivity so the mesh stays valid.\n"
            "Uses Output options."
        )
        self.disorderBatchButton.toolTip = (
            "Randomly reshuffle each mesh in the input folder (each file gets a different random order).\n"
            "Uses Output options."
        )

        # Decimate actions row
        row3 = qt.QHBoxLayout()
        self.decimateSingleButton = qt.QPushButton("DECIMATE selected model (to N points)")
        self.decimateBatchButton = qt.QPushButton("DECIMATE folder (batch, to N points)")
        row3.addWidget(self.decimateSingleButton)
        row3.addWidget(self.decimateBatchButton)
        w3 = qt.QWidget()
        w3.setLayout(row3)
        aLayout.addWidget(w3)

        self.decimateSingleButton.toolTip = "Create a decimated copy of the selected model aiming for exactly N points (as close as possible)."
        self.decimateBatchButton.toolTip = "Decimate all meshes in the input folder to N points and save them to the output folder (subfolder rules apply)."

        # Connect
        self.applyButton.connect('clicked(bool)', self.onApply)
        self.batchButton.clicked.connect(self.onBatchApply)
        self.disorderSingleButton.connect('clicked(bool)', self.onDisorderSingle)
        self.disorderBatchButton.connect('clicked(bool)', self.onDisorderBatch)
        self.decimateSingleButton.connect('clicked(bool)', self.onDecimateSingle)
        self.decimateBatchButton.connect('clicked(bool)', self.onDecimateBatch)



        # Decimation template actions row
        row4 = qt.QHBoxLayout()
        self.buildTemplateButton = qt.QPushButton("Build template (from selected model)")
        self.applyTemplateBatchButton = qt.QPushButton("Batch apply template")
        row4.addWidget(self.buildTemplateButton)
        row4.addWidget(self.applyTemplateBatchButton)
        w4 = qt.QWidget()
        w4.setLayout(row4)
        aLayout.addWidget(w4)

        self.buildTemplateButton.toolTip = (
            "Build a decimation template from the currently selected model (reference).\n"
            "This template encodes which vertex IDs to keep and the output connectivity.\n"
            "Use this when models are registered with 1:1 vertex correspondence."
        )
        self.applyTemplateBatchButton.toolTip = (
            "Apply the last built/loaded decimation template to all meshes in the input folder.\n"
            "This preserves correspondence because the SAME vertex IDs are kept in each mesh."
        )

        self.buildTemplateButton.connect('clicked(bool)', self.onBuildDecimationTemplate)
        self.applyTemplateBatchButton.connect('clicked(bool)', self.onBatchApplyDecimationTemplate)



        self.layout.addStretch(1)




    # --------- UI helpers ---------
    def onBrowseInput(self):
        d = qt.QFileDialog.getExistingDirectory(self.parent, "Select target directory with meshes (.vtk/.ply)")
        if d:
            self.inputDirEdit.setText(d)

    def onBrowseOutput(self):
        d = qt.QFileDialog.getExistingDirectory(self.parent, "Select base output directory")
        if d:
            self.outputDirEdit.setText(d)

    # --------- Single ordering ---------
    def onApply(self):
        try:
            modelNode = self.modelSelector.currentNode()
            if not modelNode:
                slicer.util.errorDisplay("Please select a model.")
                return

            method = self.methodCombo.currentText  # property -> str in Slicer/Qt
            tsneParams = {
                'perplexity': float(self.perplexitySpin.value),
                'max_iter': int(self.nIterSpin.value),
                'learning_rate': 'auto' if bool(self.lrAutoCheck.checked) else float(self.lrSpin.value),
                'init': str(self.initCombo.currentText),
                'random_state': int(self.seedSpin.value)
            }
            tspStart = int(self.tspStartSpin.value)

            order = self.logic.computeOrdering(
                modelNode,
                method=method,
                tsne_params=tsneParams,
                tsp_start=tspStart
            )
            self.logic.set_last_order(order)

            outNodes = {}
            if bool(self.makePolylineCheck.checked):
                outNodes['pathModel'] = self.logic.createOrderPath(
                    modelNode, order, colorGradient=bool(self.gradientCheck.checked)
                )
            if bool(self.remapModelCheck.checked):
                outNodes['reindexedModel'] = self.logic.reindexModel(modelNode, order)

            msg = []
            if outNodes.get('pathModel'):
                msg.append(f"Path: {outNodes['pathModel'].GetName()}")
            if outNodes.get('reindexedModel'):
                msg.append(f"Reindexed: {outNodes['reindexedModel'].GetName()}")
            slicer.util.infoDisplay("Created: " + ", ".join(msg) if msg else "No outputs created")
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def _updateStoredOrderLabel(self):
        order = self.logic.get_last_order()
        self.storedOrderLabel.setText(
            f"Stored ordering: {len(order)} points" if order else "Stored ordering: none"
        )

    def onSaveOrderingTopology(self):
        if not self.logic.get_last_order():
            slicer.util.errorDisplay("No ordering computed yet. Run 'Order points (single)' first.")
            return

        filePath = qt.QFileDialog.getSaveFileName(self.parent, "Save ordering+topology", "", "NPZ files (*.npz)")
        if filePath:
            if not filePath.lower().endswith(".npz"):
                filePath += ".npz"
            try:
                # Save using current selected model as the 'source' topology reference
                modelNode = self.modelSelector.currentNode()
                if not modelNode:
                    slicer.util.errorDisplay("Please select the source model (same used to compute ordering).")
                    return
                self.logic.save_ordering_topology(filePath, modelNode)
                self._updateStoredOrderLabel()
                slicer.util.infoDisplay(f"Saved ordering+topology to:\n{filePath}")
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to save: {e}")

    def onLoadOrderingTopology(self):
        filePath = qt.QFileDialog.getOpenFileName(self.parent, "Load ordering+topology", "", "NPZ files (*.npz)")
        if filePath:
            try:
                self.logic.load_ordering_topology(filePath)
                self._updateStoredOrderLabel()
                order = self.logic.get_last_order() or []
                slicer.util.infoDisplay(
                    f"Loaded ordering+topology from:\n{filePath}\n- points: {len(order)}"
                )
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to load: {e}")


    # --------- Batch ---------
    def onBatchApply(self):
        try:
            order = self.logic.get_last_order()
            if order is None:
                slicer.util.errorDisplay("No ordering computed yet. Run 'Order points (single)' first.")
                return

            inputDir = self.inputDirEdit.text.strip()
            outputDir = self.outputDirEdit.text.strip()
            if not inputDir or not os.path.isdir(inputDir):
                slicer.util.errorDisplay("Please choose a valid target directory.")
                return
            if not outputDir or not os.path.isdir(outputDir):
                slicer.util.errorDisplay("Please choose a valid output directory.")
                return

            makeSub = bool(self.makeSubdirCheck.checked)
            subName = self.subdirNameEdit.text.strip() or "Reindexed"
            finalOut = os.path.join(outputDir, subName) if makeSub else outputDir
            if not os.path.isdir(finalOut):
                os.makedirs(finalOut, exist_ok=True)

            count, saved = self.logic.batchReindexFolder(inputDir, finalOut, order)
            slicer.util.infoDisplay(f"Batch done. Processed: {count}  Saved: {saved}\nOutput: {finalOut}")
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def _getDisorderSeed(self):
        return int(self.disSeedSpin.value) if bool(self.disSeedCheck.checked) else None

    def onDisorderSingle(self):
        try:
            modelNode = self.modelSelector.currentNode()
            if not modelNode:
                slicer.util.errorDisplay("Please select a model.")
                return

            seed = self._getDisorderSeed()
            order = self.logic.computeRandomOrdering(modelNode, seed=seed)
            self.logic.set_last_order(order)  # optional, but consistent

            outNodes = {}
            if bool(self.makePolylineCheck.checked):
                outNodes['pathModel'] = self.logic.createOrderPath(
                    modelNode, order, colorGradient=bool(self.gradientCheck.checked)
                )
            if bool(self.remapModelCheck.checked):
                outNodes['reindexedModel'] = self.logic.reindexModel(modelNode, order)

            msg = []
            if outNodes.get('pathModel'):
                msg.append(f"Path: {outNodes['pathModel'].GetName()}")
            if outNodes.get('reindexedModel'):
                msg.append(f"Disordered: {outNodes['reindexedModel'].GetName()}")
            slicer.util.infoDisplay("Created: " + ", ".join(msg) if msg else "No outputs created")
        except Exception as e:
            slicer.util.errorDisplay(str(e))


    def onDisorderBatch(self):
        try:
            inputDir = self.inputDirEdit.text.strip()
            outputDir = self.outputDirEdit.text.strip()
            if not inputDir or not os.path.isdir(inputDir):
                slicer.util.errorDisplay("Please choose a valid target directory.")
                return
            if not outputDir or not os.path.isdir(outputDir):
                slicer.util.errorDisplay("Please choose a valid output directory.")
                return

            makeSub = bool(self.makeSubdirCheck.checked)
            subName = (self.subdirNameEdit.text.strip() or "Disordered")
            finalOut = os.path.join(outputDir, subName) if makeSub else outputDir
            if not os.path.isdir(finalOut):
                os.makedirs(finalOut, exist_ok=True)

            base_seed = self._getDisorderSeed()

            # pass UI output choices down
            opts = {
                "makePolyline": bool(self.makePolylineCheck.checked),
                "colorGradient": bool(self.gradientCheck.checked),
                "remapModel": bool(self.remapModelCheck.checked),
            }

            count, saved = self.logic.batchDisorderFolder(
                inputDir, finalOut, base_seed=base_seed, outputOptions=opts
            )
            slicer.util.infoDisplay(f"Disorder batch done. Processed: {count}  Saved: {saved}\nOutput: {finalOut}")
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def onDecimateSingle(self):
        try:
            modelNode = self.modelSelector.currentNode()
            if not modelNode:
                slicer.util.errorDisplay("Please select a model.")
                return

            targetN = int(self.decTargetPointsSpin.value)
            outNode = self.logic.decimateModelToNPoints(modelNode, targetN)

            slicer.util.infoDisplay(f"Created: {outNode.GetName()} (target N={targetN})")
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def onDecimateBatch(self):
        try:
            inputDir = self.inputDirEdit.text.strip()
            outputDir = self.outputDirEdit.text.strip()
            if not inputDir or not os.path.isdir(inputDir):
                slicer.util.errorDisplay("Please choose a valid target directory.")
                return
            if not outputDir or not os.path.isdir(outputDir):
                slicer.util.errorDisplay("Please choose a valid output directory.")
                return

            targetN = int(self.decTargetPointsSpin.value)

            makeSub = bool(self.makeSubdirCheck.checked)
            subName = (self.subdirNameEdit.text.strip() or f"Decimated_{targetN}")
            finalOut = os.path.join(outputDir, subName) if makeSub else outputDir
            if not os.path.isdir(finalOut):
                os.makedirs(finalOut, exist_ok=True)

            count, saved = self.logic.batchDecimateFolder(inputDir, finalOut, targetN)
            slicer.util.infoDisplay(f"Decimate batch done. Processed: {count}  Saved: {saved}\nOutput: {finalOut}")
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def onBuildDecimationTemplate(self):
        try:
            modelNode = self.modelSelector.currentNode()
            if not modelNode:
                slicer.util.errorDisplay("Please select a reference model.")
                return

            targetN = int(self.decTargetPointsSpin.value)

            template = self.logic.build_decimation_template_from_reference(modelNode, targetN)
            self.logic.set_loaded_decimation_template(template)  # lo añadimos abajo en Logic

            slicer.util.infoDisplay(
                f"Decimation template built from: {modelNode.GetName()}\n"
                f"- Output points: {int(template['n_points'])}\n"
                f"- keep_ids length: {len(template['keep_ids'])}"
            )
        except Exception as e:
            slicer.util.errorDisplay(str(e))

    def onBatchApplyDecimationTemplate(self):
        try:
            template = self.logic.get_loaded_decimation_template()
            if template is None:
                slicer.util.errorDisplay("No template available. Click 'Build template' first (or load one if you add that UI).")
                return

            inputDir = self.inputDirEdit.text.strip()
            outputDir = self.outputDirEdit.text.strip()
            if not inputDir or not os.path.isdir(inputDir):
                slicer.util.errorDisplay("Please choose a valid input directory.")
                return
            if not outputDir or not os.path.isdir(outputDir):
                slicer.util.errorDisplay("Please choose a valid output directory.")
                return

            makeSub = bool(self.makeSubdirCheck.checked)
            subName = (self.subdirNameEdit.text.strip() or f"DecimatedTemplate_{int(template['n_points'])}")
            finalOut = os.path.join(outputDir, subName) if makeSub else outputDir
            if not os.path.isdir(finalOut):
                os.makedirs(finalOut, exist_ok=True)

            count, saved = self.logic.batchApplyDecimationTemplateFolder(inputDir, finalOut, template)
            slicer.util.infoDisplay(f"Batch apply template done. Processed: {count}  Saved: {saved}\nOutput: {finalOut}")
        except Exception as e:
            slicer.util.errorDisplay(str(e))



# ------------------------------------------------------------
# Logic
# ------------------------------------------------------------
class ETSE_UV__MeshOrdererLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        try:
            super(ETSE_UV__MeshOrdererLogic, self).__init__()
        except Exception:
            pass
        self._last_order = None
        self._loaded_topology = None
        

    # --- ordering storage ---
    def set_last_order(self, order):
        self._last_order = list(map(int, list(order))) if order is not None else None

    def get_last_order(self):
        return self._last_order

    # --- core helpers ---
    def _polydata_to_numpy(self, poly):
        n = poly.GetNumberOfPoints()
        pts = []
        p = [0.0, 0.0, 0.0]
        for i in range(n):
            poly.GetPoint(i, p)
            pts.append((p[0], p[1], p[2]))
        import numpy as np
        return np.asarray(pts, dtype=float)

    def computeOrdering(self, modelNode, method="t-SNE", tsne_params=None, tsp_start=0):
        poly = modelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Input model has no points.")

        X = self._polydata_to_numpy(poly)
        import numpy as np
        n = X.shape[0]

        if method == "t-SNE":
            ensure_packages(
                [("sklearn", "scikit-learn")],
                interactive=True,
                module_name="ETSE-UV Mesh Orderer",
            )

            from sklearn.manifold import TSNE
            params = tsne_params or {}
            tsne = TSNE(
                n_components=1,
                perplexity=float(params.get('perplexity', 30.0)),
                max_iter=int(params.get('max_iter', 1000)),
                learning_rate=params.get('learning_rate', 'auto'),
                init=str(params.get('init', 'pca')),
                random_state=int(params.get('random_state', 42)),
                verbose=0
            )
            y = tsne.fit_transform(X)
            order = np.argsort(y.squeeze()).astype(int)

        elif method.startswith("PCA"):
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            y = Xc @ Vt.T[:, 0]
            order = np.argsort(y).astype(int)

        elif method.startswith("Trimesh TSP"):
            ensure_packages(
                [("trimesh", "trimesh")],
                interactive=True,
                module_name="ETSE-UV Mesh Orderer",
            )

            import trimesh
            if n == 0:
                raise RuntimeError("No points for TSP.")
            s = max(0, min(int(tsp_start), n - 1))
            traversal, distances = trimesh.points.tsp(X, start=s)
            order = np.asarray(traversal, dtype=int)

        else:
            raise RuntimeError(f"Unknown method: {method}")

        return order

    # --- create path model ---
    def createOrderPath(self, modelNode, order, colorGradient=True):
        poly = modelNode.GetPolyData()
        n = poly.GetNumberOfPoints()
        if order is None or len(order) != n:
            raise RuntimeError("Order length must equal number of points.")

        pathPoints = vtk.vtkPoints()
        pathPoints.SetNumberOfPoints(n)
        p = [0.0, 0.0, 0.0]
        for i, pid in enumerate(order):
            poly.GetPoint(int(pid), p)
            pathPoints.SetPoint(i, p)

        outPD = vtk.vtkPolyData()
        outPD.SetPoints(pathPoints)

        cells = vtk.vtkCellArray()
        if colorGradient:
            scalars = vtk.vtkIntArray()
            scalars.SetName('SegmentId')
            for i in range(n - 1):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, i + 1)
                cells.InsertNextCell(line)
                scalars.InsertNextValue(i)
            outPD.SetLines(cells)
            outPD.GetCellData().SetScalars(scalars)
        else:
            pl = vtk.vtkPolyLine()
            pl.GetPointIds().SetNumberOfIds(n)
            for i in range(n):
                pl.GetPointIds().SetId(i, i)
            cells.InsertNextCell(pl)
            outPD.SetLines(cells)

        outNode = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLModelNode',
            f"{modelNode.GetName()}_OrderPath"
        )
        outNode.SetAndObservePolyData(outPD)
        outNode.CreateDefaultDisplayNodes()
        dn = outNode.GetDisplayNode()
        if dn:
            dn.SetLineWidth(3)
            dn.SetRepresentation(slicer.vtkMRMLDisplayNode.WireframeRepresentation)
            if colorGradient:
                try:
                    colorNode = slicer.util.getNode('vtkMRMLColorTableNodeRainbow')
                except Exception:
                    colorNode = None
                if colorNode:
                    dn.SetAndObserveColorNodeID(colorNode.GetID())
                outPD.GetCellData().SetActiveScalars('SegmentId')
                dn.SetScalarVisibility(True)
                dn.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
                dn.SetScalarRange(0, max(1, n - 2))
        return outNode

    # --- reindex model ---
    def reindexModel(self, modelNode, order):
        poly = modelNode.GetPolyData()
        n = poly.GetNumberOfPoints()
        if order is None or len(order) != n:
            raise RuntimeError("Order length must equal number of points.")

        inv = {int(old): int(new) for new, old in enumerate(order)}

        newPoints = vtk.vtkPoints()
        newPoints.SetNumberOfPoints(n)
        p = [0.0, 0.0, 0.0]
        for new_id, old_id in enumerate(order):
            poly.GetPoint(int(old_id), p)
            newPoints.SetPoint(new_id, p)

        newPolys = vtk.vtkCellArray()
        newLines = vtk.vtkCellArray()
        newStrips = vtk.vtkCellArray()
        newVerts = vtk.vtkCellArray()

        nCells = poly.GetNumberOfCells()
        for cid in range(nCells):
            cell = poly.GetCell(cid)
            idList = vtk.vtkIdList()
            for k in range(cell.GetNumberOfPoints()):
                old_pid = int(cell.GetPointId(k))
                idList.InsertNextId(inv.get(old_pid, 0))
            ctype = cell.GetCellType()
            if ctype in (vtk.VTK_TRIANGLE, vtk.VTK_QUAD, vtk.VTK_POLYGON, vtk.VTK_TETRA, vtk.VTK_HEXAHEDRON):
                newPolys.InsertNextCell(idList)
            elif ctype in (vtk.VTK_LINE, vtk.VTK_POLY_LINE):
                newLines.InsertNextCell(idList)
            elif ctype == vtk.VTK_TRIANGLE_STRIP:
                newStrips.InsertNextCell(idList)
            else:
                newVerts.InsertNextCell(idList)

        out = vtk.vtkPolyData()
        out.SetPoints(newPoints)
        if newPolys.GetNumberOfCells() > 0:
            out.SetPolys(newPolys)
        if newLines.GetNumberOfCells() > 0:
            out.SetLines(newLines)
        if newStrips.GetNumberOfCells() > 0:
            out.SetStrips(newStrips)
        if newVerts.GetNumberOfCells() > 0:
            out.SetVerts(newVerts)
        out.Modified()

        outNode = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLModelNode',
            f"{modelNode.GetName()}_Reindexed"
        )
        outNode.SetAndObservePolyData(out)
        outNode.CreateDefaultDisplayNodes()
        return outNode


    def _savePolyDataToFile(self, polyData, filePath):
        """
        Save using Slicer's ModelStorageNode to control coordinate system (RAS),
        avoiding unwanted LPS<->RAS flips on reload.
        """
        if polyData is None or polyData.GetNumberOfPoints() == 0:
            raise RuntimeError("No polydata to save.")

        # Temporary model node for storage write
        tmpNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "__tmpSaveModel")
        tmpNode.SetAndObservePolyData(polyData)

        storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelStorageNode", "__tmpSaveStorage")
        storage.SetFileName(filePath)

        # IMPORTANT: Force RAS so Slicer does NOT apply an extra LPS->RAS conversion when reloading
        if hasattr(storage, "SetCoordinateSystemToRAS"):
            storage.SetCoordinateSystemToRAS()
        else:
            # fallback for older builds (enum name can vary slightly)
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


    # --- batch ---
    def batchReindexFolder(self, inputDir, outputDir, order):
        if not os.path.isdir(inputDir) or not os.path.isdir(outputDir):
            raise RuntimeError("Invalid directories.")

        files = [
            f for f in os.listdir(inputDir)
            if os.path.isfile(os.path.join(inputDir, f))
            and os.path.splitext(f)[1].lower() in ('.vtk', '.ply')
        ]

        processed = 0
        saved = 0
        for fname in files:
            inPath = os.path.join(inputDir, fname)
            ok, node = slicer.util.loadModel(inPath, returnNode=True)
            if not ok or node is None:
                continue
            try:
                topo = self.get_loaded_topology()
                if topo is not None:
                    outNode = self.apply_loaded_topology_to_model(node, output_name=node.GetName()+"_TopoApplied")
                else:
                    outNode = self.reindexModel(node, order)

                outPath = os.path.join(outputDir, fname)
                self._savePolyDataToFile(outNode.GetPolyData(), outPath)
                saved += 1

            except Exception:
                pass
            finally:
                processed += 1
                slicer.mrmlScene.RemoveNode(node)
                if 'outNode' in locals() and outNode:
                    slicer.mrmlScene.RemoveNode(outNode)
        return processed, saved


    def save_ordering_topology(self, filePath, sourceModelNode):
        """
        Save:
          - order (old->new via list of old ids in new order)
          - remapped faces/lines/strips/verts as flat VTK-style arrays
        """
        order = self.get_last_order()
        if not order:
            raise RuntimeError("No ordering stored to save.")

        poly = sourceModelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Source model has no polydata/points.")

        # Build reindexed polydata (this remaps connectivity)
        rePDNode = self.reindexModel(sourceModelNode, order)
        try:
            rePD = rePDNode.GetPolyData()
            import numpy as np

            def cellArrayToNumpy(cellArray):
                if cellArray is None:
                    return np.asarray([], dtype=np.int64)
                arr = cellArray.GetData()
                if arr is None:
                    return np.asarray([], dtype=np.int64)
                out = np.zeros(arr.GetNumberOfTuples(), dtype=np.int64)
                for i in range(arr.GetNumberOfTuples()):
                    out[i] = int(arr.GetTuple1(i))
                return out

            # Save VTK "connectivity arrays" (n, id0, id1, ..., n, id0, ...)
            polys_np  = cellArrayToNumpy(rePD.GetPolys())
            lines_np  = cellArrayToNumpy(rePD.GetLines())
            strips_np = cellArrayToNumpy(rePD.GetStrips())
            verts_np  = cellArrayToNumpy(rePD.GetVerts())

            np.savez_compressed(
                filePath,
                order=np.asarray(order, dtype=np.int64),
                n_points=np.int64(rePD.GetNumberOfPoints()),
                polys=polys_np,
                lines=lines_np,
                strips=strips_np,
                verts=verts_np,
            )
        finally:
            # cleanup temporary node created by reindexModel
            if rePDNode:
                slicer.mrmlScene.RemoveNode(rePDNode)

    def load_ordering_topology(self, filePath):
        import numpy as np
        data = np.load(filePath, allow_pickle=False)

        order = data.get("order", None)
        if order is None:
            raise RuntimeError("NPZ missing 'order' array.")

        self.set_last_order(order.tolist())

        # store loaded connectivity arrays for optional later use
        self._loaded_topology = {
            "n_points": int(data.get("n_points", 0)),
            "polys":  data.get("polys",  np.asarray([], dtype=np.int64)),
            "lines":  data.get("lines",  np.asarray([], dtype=np.int64)),
            "strips": data.get("strips", np.asarray([], dtype=np.int64)),
            "verts":  data.get("verts",  np.asarray([], dtype=np.int64)),
        }

    def get_loaded_topology(self):
        return getattr(self, "_loaded_topology", None)


    def _numpyToCellArray(self, flat):
        # flat is VTK style: [n, id0, ..., id(n-1), n, ...]
        ca = vtk.vtkCellArray()
        if flat is None or len(flat) == 0:
            return ca
        i = 0
        L = int(len(flat))
        while i < L:
            n = int(flat[i]); i += 1
            ids = vtk.vtkIdList()
            for _ in range(n):
                ids.InsertNextId(int(flat[i])); i += 1
            ca.InsertNextCell(ids)
        return ca

    def apply_loaded_topology_to_model(self, targetModelNode, output_name=None):
        topo = self.get_loaded_topology()
        if topo is None:
            raise RuntimeError("No loaded topology. Load an .npz first.")

        poly = targetModelNode.GetPolyData()
        n = poly.GetNumberOfPoints()
        if n != topo["n_points"]:
            raise RuntimeError("Target model point count does not match saved topology.")

        # IMPORTANT: use the loaded ordering to reorder the target points
        order = self.get_last_order()
        if not order or len(order) != n:
            raise RuntimeError("Loaded ordering missing or size mismatch.")

        newPoints = vtk.vtkPoints()
        newPoints.SetNumberOfPoints(n)
        p = [0.0, 0.0, 0.0]
        for new_id, old_id in enumerate(order):
            poly.GetPoint(int(old_id), p)
            newPoints.SetPoint(new_id, p)

        out = vtk.vtkPolyData()
        out.SetPoints(newPoints)

        polys  = self._numpyToCellArray(topo["polys"])
        lines  = self._numpyToCellArray(topo["lines"])
        strips = self._numpyToCellArray(topo["strips"])
        verts  = self._numpyToCellArray(topo["verts"])
        if polys.GetNumberOfCells()  > 0: out.SetPolys(polys)
        if lines.GetNumberOfCells()  > 0: out.SetLines(lines)
        if strips.GetNumberOfCells() > 0: out.SetStrips(strips)
        if verts.GetNumberOfCells()  > 0: out.SetVerts(verts)
        out.Modified()

        name = output_name or (targetModelNode.GetName() + "_TopoApplied")
        outNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', name)
        outNode.SetAndObservePolyData(out)
        outNode.CreateDefaultDisplayNodes()
        return outNode


    def computeRandomOrdering(self, modelNode, seed=None):
        poly = modelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Input model has no points.")
        n = poly.GetNumberOfPoints()

        import numpy as np
        rng = np.random.default_rng(None if seed is None else int(seed))
        order = rng.permutation(n).astype(int)
        return order

    def disorderModel(self, modelNode, seed=None):
        order = self.computeRandomOrdering(modelNode, seed=seed)
        # optional: store it (useful if you want to repeat apply to others, but not required)
        self.set_last_order(order)
        outNode = self.reindexModel(modelNode, order)
        return outNode

    def batchDisorderFolder(self, inputDir, outputDir, base_seed=None, outputOptions=None):
        if not os.path.isdir(inputDir) or not os.path.isdir(outputDir):
            raise RuntimeError("Invalid directories.")

        files = [
            f for f in os.listdir(inputDir)
            if os.path.isfile(os.path.join(inputDir, f))
            and os.path.splitext(f)[1].lower() in ('.vtk', '.ply')
        ]

        import numpy as np
        rng = np.random.default_rng(None if base_seed is None else int(base_seed))

        opts = outputOptions or {}
        makePolyline = bool(opts.get("makePolyline", False))
        colorGradient = bool(opts.get("colorGradient", True))
        remapModel = bool(opts.get("remapModel", True))

        processed = 0
        saved = 0
        for fname in files:
            inPath = os.path.join(inputDir, fname)
            ok, node = slicer.util.loadModel(inPath, returnNode=True)
            if not ok or node is None:
                continue

            outNode = None
            pathNode = None
            try:
                seed_i = int(rng.integers(0, 2**31 - 1))
                order = self.computeRandomOrdering(node, seed=seed_i)

                if makePolyline:
                    pathNode = self.createOrderPath(node, order, colorGradient=colorGradient)

                if remapModel:
                    outNode = self.reindexModel(node, order)
                    outPath = os.path.join(outputDir, fname)
                    self._savePolyDataToFile(outNode.GetPolyData(), outPath)
                    saved += 1


            except Exception:
                pass
            finally:
                processed += 1
                slicer.mrmlScene.RemoveNode(node)
                if outNode:
                    slicer.mrmlScene.RemoveNode(outNode)
                if pathNode:
                    slicer.mrmlScene.RemoveNode(pathNode)

        return processed, saved

    def _decimate_polydata_to_npoints(self, poly, targetN, maxIters=16):
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Input polydata has no points.")
        targetN = int(targetN)
        if targetN < 4:
            raise RuntimeError("Target N too small.")

        n0 = int(poly.GetNumberOfPoints())
        if targetN >= n0:
            # nothing to do
            return poly

        # Ensure triangles for decimation stability
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(poly)
        tri.Update()
        triPD = tri.GetOutput()

        # Use quadric decimation and tune TargetReduction with binary search
        # TargetReduction = fraction of triangles to remove (approx)
        low = 0.0
        high = 0.9999

        bestPD = None
        bestErr = 10**18

        for _ in range(int(maxIters)):
            mid = 0.5 * (low + high)

            dec = vtk.vtkQuadricDecimation()
            dec.SetInputData(triPD)
            dec.SetTargetReduction(mid)
            dec.AttributeErrorMetricOn()
            dec.Update()
            outPD = dec.GetOutput()

            nOut = int(outPD.GetNumberOfPoints())
            err = abs(nOut - targetN)

            if err < bestErr:
                bestErr = err
                bestPD = outPD

            # If we got too many points, we need MORE reduction
            if nOut > targetN:
                low = mid
            else:
                high = mid

            # Early stop if exact
            if err == 0:
                break

        # Clean to remove unused points, etc.
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(bestPD)
        clean.Update()
        return clean.GetOutput()

    def decimateModelToNPoints(self, modelNode, targetN):
        poly = modelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Input model has no points.")

        outPD = self._decimate_polydata_to_npoints(poly, targetN)

        outNode = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLModelNode',
            f"{modelNode.GetName()}_Decimated_{int(targetN)}"
        )
        outNode.SetAndObservePolyData(outPD)
        outNode.CreateDefaultDisplayNodes()
        return outNode

    def batchDecimateFolder(self, inputDir, outputDir, targetN):
        if not os.path.isdir(inputDir) or not os.path.isdir(outputDir):
            raise RuntimeError("Invalid directories.")

        files = [
            f for f in os.listdir(inputDir)
            if os.path.isfile(os.path.join(inputDir, f))
            and os.path.splitext(f)[1].lower() in ('.vtk', '.ply')
        ]

        processed = 0
        saved = 0
        for fname in files:
            inPath = os.path.join(inputDir, fname)
            ok, node = slicer.util.loadModel(inPath, returnNode=True)
            if not ok or node is None:
                continue

            outNode = None
            try:
                outNode = self.decimateModelToNPoints(node, targetN)

                outPath = os.path.join(outputDir, fname)

                pd = outNode.GetPolyData() if outNode else None
                if pd is None or pd.GetNumberOfPoints() == 0:
                    raise RuntimeError("Decimation produced empty polydata")

                self._savePolyDataToFile(pd, outPath)
                saved += 1

            except Exception as e:
                import traceback
                print(f"[batchDecimateFolder] FAILED on {fname}: {e}")
                traceback.print_exc()

            finally:
                processed += 1
                slicer.mrmlScene.RemoveNode(node)
                if outNode:
                    slicer.mrmlScene.RemoveNode(outNode)

        return processed, saved

    def build_decimation_template_from_reference(self, refModelNode, targetN):
        """
        1) Decimate reference with VTK (geometry-based)
        2) Map output points back to unique nearest original point IDs -> keep_ids
        3) Save connectivity of the decimated output but expressed in 0..N-1 indices
        """
        import numpy as np
        import vtk

        refPoly = refModelNode.GetPolyData()
        if refPoly is None or refPoly.GetNumberOfPoints() == 0:
            raise RuntimeError("Reference model has no points.")

        # decimate reference (your existing function)
        decPD = self._decimate_polydata_to_npoints(refPoly, targetN)
        n_out = decPD.GetNumberOfPoints()
        if n_out <= 0:
            raise RuntimeError("Decimation produced empty output.")

        # locator on original points
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(refPoly)
        locator.BuildLocator()

        used = set()
        keep_ids = np.full(n_out, -1, dtype=np.int64)

        # first pass: nearest unique ids
        p = [0.0, 0.0, 0.0]
        for i in range(n_out):
            decPD.GetPoint(i, p)
            pid = int(locator.FindClosestPoint(p))
            if pid not in used:
                keep_ids[i] = pid
                used.add(pid)

        # second pass: fill duplicates with nearest unused (brutal but effective)
        if np.any(keep_ids < 0):
            # build list of all ids sorted by distance for each missing point
            for i in range(n_out):
                if keep_ids[i] >= 0:
                    continue
                decPD.GetPoint(i, p)

                # get a small candidate neighborhood using FindClosestNPoints
                idList = vtk.vtkIdList()
                locator.FindClosestNPoints(64, p, idList)

                chosen = None
                for k in range(idList.GetNumberOfIds()):
                    cand = int(idList.GetId(k))
                    if cand not in used:
                        chosen = cand
                        break

                # fallback: linear search (worst case)
                if chosen is None:
                    for cand in range(refPoly.GetNumberOfPoints()):
                        if cand not in used:
                            chosen = cand
                            break

                keep_ids[i] = int(chosen)
                used.add(int(chosen))

        # Now build a mapping from original point id -> new index (0..N-1)
        inv = {int(old): int(new) for new, old in enumerate(keep_ids.tolist())}

        # Convert decPD cell arrays to flat numpy but remapped through keep_ids->inv
        def cellArrayToFlatRemapped(cellArray):
            if cellArray is None:
                return np.asarray([], dtype=np.int64)
            arr = cellArray.GetData()
            if arr is None or arr.GetNumberOfTuples() == 0:
                return np.asarray([], dtype=np.int64)

            flat = np.zeros(arr.GetNumberOfTuples(), dtype=np.int64)
            # arr already contains VTK-style flat: n, id0, id1, ...
            for i in range(arr.GetNumberOfTuples()):
                flat[i] = int(arr.GetTuple1(i))

            # remap ids (skip the "n" entries)
            out = flat.copy()
            idx = 0
            L = len(out)
            while idx < L:
                n = int(out[idx]); idx += 1
                for _ in range(n):
                    old_out_id = int(out[idx])           # point id in decPD indexing
                    old_ref_id = int(keep_ids[old_out_id])  # mapped to original id
                    out[idx] = int(inv[old_ref_id])      # new id 0..N-1
                    idx += 1
            return out

        template = {
            "n_points": int(n_out),
            "keep_ids": keep_ids.astype(np.int64),
            "polys":  cellArrayToFlatRemapped(decPD.GetPolys()),
            "lines":  cellArrayToFlatRemapped(decPD.GetLines()),
            "strips": cellArrayToFlatRemapped(decPD.GetStrips()),
            "verts":  cellArrayToFlatRemapped(decPD.GetVerts()),
        }
        return template

    def save_decimation_template(self, filePath, template):
        import numpy as np
        np.savez_compressed(
            filePath,
            n_points=np.int64(template["n_points"]),
            keep_ids=np.asarray(template["keep_ids"], dtype=np.int64),
            polys=np.asarray(template["polys"], dtype=np.int64),
            lines=np.asarray(template["lines"], dtype=np.int64),
            strips=np.asarray(template["strips"], dtype=np.int64),
            verts=np.asarray(template["verts"], dtype=np.int64),
        )

    def load_decimation_template(self, filePath):
        import numpy as np
        data = np.load(filePath, allow_pickle=False)
        self._loaded_decimation_template = {
            "n_points": int(data["n_points"]),
            "keep_ids": data["keep_ids"].astype(np.int64),
            "polys": data["polys"].astype(np.int64),
            "lines": data["lines"].astype(np.int64),
            "strips": data["strips"].astype(np.int64),
            "verts": data["verts"].astype(np.int64),
        }

    def get_loaded_decimation_template(self):
        return getattr(self, "_loaded_decimation_template", None)

    def apply_decimation_template_to_model(self, targetModelNode, template, output_name=None):
        import vtk
        import numpy as np
        poly = targetModelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Target model has no points.")

        keep_ids = template["keep_ids"]
        n_out = int(template["n_points"])

        if poly.GetNumberOfPoints() <= int(np.max(keep_ids)):
            raise RuntimeError("Target model does not have enough points for this template.")

        newPoints = vtk.vtkPoints()
        newPoints.SetNumberOfPoints(n_out)

        p = [0.0, 0.0, 0.0]
        for new_id in range(n_out):
            old_id = int(keep_ids[new_id])
            poly.GetPoint(old_id, p)
            newPoints.SetPoint(new_id, p)

        out = vtk.vtkPolyData()
        out.SetPoints(newPoints)

        polys  = self._numpyToCellArray(template["polys"])
        lines  = self._numpyToCellArray(template["lines"])
        strips = self._numpyToCellArray(template["strips"])
        verts  = self._numpyToCellArray(template["verts"])

        if polys.GetNumberOfCells()  > 0: out.SetPolys(polys)
        if lines.GetNumberOfCells()  > 0: out.SetLines(lines)
        if strips.GetNumberOfCells() > 0: out.SetStrips(strips)
        if verts.GetNumberOfCells()  > 0: out.SetVerts(verts)
        out.Modified()

        name = output_name or (targetModelNode.GetName() + f"_DecimatedTemplate_{n_out}")
        outNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', name)
        outNode.SetAndObservePolyData(out)
        outNode.CreateDefaultDisplayNodes()
        return outNode

    def set_loaded_decimation_template(self, template):
        self._loaded_decimation_template = template

    def batchApplyDecimationTemplateFolder(self, inputDir, outputDir, template):
        if not os.path.isdir(inputDir) or not os.path.isdir(outputDir):
            raise RuntimeError("Invalid directories.")

        files = [
            f for f in os.listdir(inputDir)
            if os.path.isfile(os.path.join(inputDir, f))
            and os.path.splitext(f)[1].lower() in ('.vtk', '.ply')
        ]

        processed = 0
        saved = 0

        for fname in files:
            inPath = os.path.join(inputDir, fname)
            ok, node = slicer.util.loadModel(inPath, returnNode=True)
            if not ok or node is None:
                continue

            outNode = None
            try:
                outNode = self.apply_decimation_template_to_model(
                    node, template, output_name=node.GetName() + f"_DecimatedTemplate_{int(template['n_points'])}"
                )

                outPath = os.path.join(outputDir, fname)
                self._savePolyDataToFile(outNode.GetPolyData(), outPath)
                saved += 1

            except Exception as e:
                import traceback
                print(f"[batchApplyDecimationTemplateFolder] FAILED on {fname}: {e}")
                traceback.print_exc()

            finally:
                processed += 1
                slicer.mrmlScene.RemoveNode(node)
                if outNode:
                    slicer.mrmlScene.RemoveNode(outNode)

        return processed, saved
