# ETSE_UV__SofaHrtfPlotter.py
# 3D Slicer scripted module
#
# Live SOFA / HRTF plotter with OSC input, manual angle override, native
# Slicer plots, and pyfar/matplotlib-style diagnostic plots.

import os
import io
import time
import threading
import traceback
import contextlib
import tempfile

import numpy as np
import slicer
import qt
import ctk
from slicer.ScriptedLoadableModule import *

from Resources.ETSE_UV__Dependencies import ensure_packages

ensure_packages(
    [
        ("pythonosc", "python-osc"),
        ("pyfar", "pyfar"),
        ("sofar", "sofar"),
        ("matplotlib", "matplotlib"),
    ],
    interactive=False,
    module_name="ETSE-UV SOFA HRTF Plotter",
)


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------
class ETSE_UV__SofaHrtfPlotter(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV SOFA HRTF Plotter"
        parent.categories = ["ETSE_UV"]
        parent.dependencies = []
        parent.contributors = ["ETSE-UV"]
        parent.helpText = """
                        <p>Live HRTF plotting module for 3D Slicer.</p>

                        <p><b>Features:</b></p>
                        <ul>
                          <li>Start and stop an OSC listener for source position and HRTF path messages.</li>
                          <li>Load SOFA files manually or from OSC.</li>
                          <li>Optionally override the shown sample using azimuth/elevation sliders.</li>
                          <li>Plot with Slicer's native line plotter or pyfar/matplotlib-style figures.</li>
                          <li>Single sample, phase, spectrogram, elevation/lateral cuts, stacked cuts, and coordinate views.</li>
                        </ul>

                        <p>The original standalone OSC sniffer workflow was adapted to Slicer using a widget-based UI,
                        a background OSC server thread, and a Qt timer on the main GUI thread.</p>
                        """
        parent.acknowledgementText = (
            "Developed by Juan Antonio De Rus Arance at the Escola Tècnica Superior "
            "d'Enginyeria (ETSE-UV), Universitat de València, in the context of the "
            "Signal Processing & Acoustic Technology (SPAT) research group. "
            "Thanks to the 3D Slicer, pyfar, sofar, python-osc, matplotlib, NumPy, and related "
            "open-source communities."
        )


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class ETSE_UV__SofaHrtfPlotterWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = ETSE_UV__SofaHrtfPlotterLogic()

        # ------------------------------------------------------------------
        # Dependencies
        # ------------------------------------------------------------------
        depBox = ctk.ctkCollapsibleButton()
        depBox.text = "Python dependencies"
        self.layout.addWidget(depBox)
        depLayout = qt.QFormLayout(depBox)

        self.installButton = qt.QPushButton("Check / install Python dependencies")
        self.installButton.toolTip = (
            "Install python-osc, pyfar, sofar, and matplotlib into Slicer's Python environment if missing."
        )
        self.installButton.connect("clicked(bool)", self.onInstallDependencies)
        depLayout.addRow(self.installButton)

        # ------------------------------------------------------------------
        # SOFA loading
        # ------------------------------------------------------------------
        sofaBox = ctk.ctkCollapsibleButton()
        sofaBox.text = "SOFA input"
        self.layout.addWidget(sofaBox)
        sofaLayout = qt.QFormLayout(sofaBox)

        sofaPathRow = qt.QHBoxLayout()
        self.sofaPathEdit = qt.QLineEdit()
        self.sofaPathEdit.setPlaceholderText("Select a .sofa file or wait for OSC HRTFPath")
        sofaPathRow.addWidget(self.sofaPathEdit)

        self.browseSofaButton = qt.QPushButton("...")
        self.browseSofaButton.setMaximumWidth(40)
        self.browseSofaButton.toolTip = "Browse for a SOFA file"
        self.browseSofaButton.connect("clicked(bool)", self.onBrowseSofa)
        sofaPathRow.addWidget(self.browseSofaButton)
        sofaLayout.addRow("SOFA file:", sofaPathRow)

        self.loadSofaButton = qt.QPushButton("Load SOFA now")
        self.loadSofaButton.toolTip = "Load the SOFA file from the path above."
        self.loadSofaButton.connect("clicked(bool)", self.onLoadSofa)
        sofaLayout.addRow(self.loadSofaButton)

        # ------------------------------------------------------------------
        # OSC controls
        # ------------------------------------------------------------------
        oscBox = ctk.ctkCollapsibleButton()
        oscBox.text = "OSC listener"
        self.layout.addWidget(oscBox)
        oscLayout = qt.QFormLayout(oscBox)

        self.ipEdit = qt.QLineEdit("127.0.0.1")
        self.ipEdit.setToolTip("OSC listening IP")
        oscLayout.addRow("IP:", self.ipEdit)

        self.portSpin = qt.QSpinBox()
        self.portSpin.minimum = 1
        self.portSpin.maximum = 65535
        self.portSpin.value = 12346
        self.portSpin.setToolTip("OSC listening UDP port")
        oscLayout.addRow("Port:", self.portSpin)

        self.updateDelaySpin = ctk.ctkDoubleSpinBox()
        self.updateDelaySpin.minimum = 0.0
        self.updateDelaySpin.maximum = 30.0
        self.updateDelaySpin.singleStep = 0.1
        self.updateDelaySpin.decimals = 2
        self.updateDelaySpin.value = 0.1
        self.updateDelaySpin.setToolTip(
            "Plot is updated only after this many seconds have elapsed since the last OSC message."
        )
        oscLayout.addRow("Update delay (s):", self.updateDelaySpin)

        oscButtonRow = qt.QHBoxLayout()
        self.startOscButton = qt.QPushButton("Start OSC listener")
        self.startOscButton.connect("clicked(bool)", self.onStartOsc)
        oscButtonRow.addWidget(self.startOscButton)

        self.stopOscButton = qt.QPushButton("Stop OSC listener")
        self.stopOscButton.connect("clicked(bool)", self.onStopOsc)
        oscButtonRow.addWidget(self.stopOscButton)
        oscLayout.addRow(oscButtonRow)

        # ------------------------------------------------------------------
        # Sample override controls
        # ------------------------------------------------------------------
        overrideBox = ctk.ctkCollapsibleButton()
        overrideBox.text = "Sample selection / override"
        overrideBox.collapsed = False
        self.layout.addWidget(overrideBox)
        overrideLayout = qt.QFormLayout(overrideBox)

        self.overrideAnglesCheck = qt.QCheckBox("Override OSC/source sample with azimuth/elevation")
        self.overrideAnglesCheck.checked = False
        self.overrideAnglesCheck.setToolTip(
            "When enabled, plots use the nearest SOFA sample to the Az/El sliders, "
            "independently of the OSC position. This also applies after loading a SOFA file from disk or OSC."
        )
        self.overrideAnglesCheck.connect("toggled(bool)", self.onOverrideToggled)
        overrideLayout.addRow(self.overrideAnglesCheck)

        self.azSlider = ctk.ctkSliderWidget()
        self.azSlider.minimum = 0.0
        self.azSlider.maximum = 360.0
        self.azSlider.singleStep = 1.0
        self.azSlider.value = 0.0
        try:
            self.azSlider.decimals = 1
        except Exception:
            pass
        self.azSlider.setToolTip("Override azimuth in degrees.")
        self.azSlider.connect("valueChanged(double)", self.onOverrideAngleChanged)
        overrideLayout.addRow("Azimuth override (deg):", self.azSlider)

        self.elSlider = ctk.ctkSliderWidget()
        self.elSlider.minimum = -90.0
        self.elSlider.maximum = 90.0
        self.elSlider.singleStep = 1.0
        self.elSlider.value = 0.0
        try:
            self.elSlider.decimals = 1
        except Exception:
            pass
        self.elSlider.setToolTip("Override elevation in degrees.")
        self.elSlider.connect("valueChanged(double)", self.onOverrideAngleChanged)
        overrideLayout.addRow("Elevation override (deg):", self.elSlider)

        # ------------------------------------------------------------------
        # Plot controls
        # ------------------------------------------------------------------
        plotBox = ctk.ctkCollapsibleButton()
        plotBox.text = "Plot controls"
        plotBox.collapsed = False
        self.layout.addWidget(plotBox)
        plotLayout = qt.QFormLayout(plotBox)

        self.plotStyleCombo = qt.QComboBox()
        self.plotStyleCombo.addItems(["Slicer native line plot", "pyfar / matplotlib figure"])
        self.plotStyleCombo.setToolTip(
            "Slicer native is compact and robust for single line plots. "
            "pyfar/matplotlib gives the OSC-sniffer-like views, 2D cuts, phase, and spectrograms."
        )
        plotLayout.addRow("Plot style:", self.plotStyleCombo)

        self.plotModeCombo = qt.QComboBox()
        self.plotModeCombo.addItems([
            "single sample",
            "single sample overview",
            "elevation cut",
            "lateral cut",
            "all views one ear",
            "all views both ears",
            "stacked cut",
            "coordinates",
        ])
        self.plotModeCombo.setToolTip("Choose the diagnostic view to generate.")
        self.plotModeCombo.connect("currentIndexChanged(int)", self.onPlotModeChanged)
        plotLayout.addRow("Mode:", self.plotModeCombo)

        self.representationCombo = qt.QComboBox()
        self.representationCombo.addItems(["freq", "time", "phase", "spectrogram"])
        self.representationCombo.setToolTip("Representation for single sample / stacked sample plots.")
        plotLayout.addRow("Single representation:", self.representationCombo)

        self.earCombo = qt.QComboBox()
        self.earCombo.addItems(["left", "right"])
        self.earCombo.setToolTip("Choose ear channel for single-ear plots.")
        plotLayout.addRow("Ear:", self.earCombo)

        self.cutTypeCombo = qt.QComboBox()
        self.cutTypeCombo.addItems(["elevation", "lateral"])
        self.cutTypeCombo.setToolTip("Cut type used by stacked cut mode.")
        plotLayout.addRow("Stacked cut type:", self.cutTypeCombo)

        self.cutValueSpin = ctk.ctkDoubleSpinBox()
        self.cutValueSpin.minimum = -360.0
        self.cutValueSpin.maximum = 360.0
        self.cutValueSpin.singleStep = 1.0
        self.cutValueSpin.decimals = 1
        self.cutValueSpin.value = 0.0
        self.cutValueSpin.setToolTip("Fixed elevation/lateral value in degrees for stacked cut mode.")
        plotLayout.addRow("Stacked cut value (deg):", self.cutValueSpin)

        self.angleStepSpin = ctk.ctkDoubleSpinBox()
        self.angleStepSpin.minimum = 1.0
        self.angleStepSpin.maximum = 180.0
        self.angleStepSpin.singleStep = 1.0
        self.angleStepSpin.decimals = 1
        self.angleStepSpin.value = 30.0
        self.angleStepSpin.setToolTip("Angle spacing used when selecting samples for stacked cut mode.")
        plotLayout.addRow("Stacked angle step (deg):", self.angleStepSpin)

        self.earModeCombo = qt.QComboBox()
        self.earModeCombo.addItems(["left", "right", "both"])
        self.earModeCombo.setToolTip("Ear selection used by stacked cut mode.")
        plotLayout.addRow("Stacked ear mode:", self.earModeCombo)

        ylimRow = qt.QHBoxLayout()
        self.ylimCheck = qt.QCheckBox("Use")
        self.ylimMinSpin = ctk.ctkDoubleSpinBox()
        self.ylimMinSpin.minimum = -200.0
        self.ylimMinSpin.maximum = 200.0
        self.ylimMinSpin.value = -80.0
        self.ylimMinSpin.singleStep = 1.0
        self.ylimMaxSpin = ctk.ctkDoubleSpinBox()
        self.ylimMaxSpin.minimum = -200.0
        self.ylimMaxSpin.maximum = 200.0
        self.ylimMaxSpin.value = 20.0
        self.ylimMaxSpin.singleStep = 1.0
        ylimRow.addWidget(self.ylimCheck)
        ylimRow.addWidget(qt.QLabel("min"))
        ylimRow.addWidget(self.ylimMinSpin)
        ylimRow.addWidget(qt.QLabel("max"))
        ylimRow.addWidget(self.ylimMaxSpin)
        plotLayout.addRow("Y limits for stacked freq (dB):", ylimRow)

        angleLimitsRow = qt.QHBoxLayout()
        self.angleLimitsCheck = qt.QCheckBox("Use")
        self.angleMinSpin = ctk.ctkDoubleSpinBox()
        self.angleMinSpin.minimum = -360.0
        self.angleMinSpin.maximum = 360.0
        self.angleMinSpin.value = -180.0
        self.angleMinSpin.singleStep = 1.0
        self.angleMaxSpin = ctk.ctkDoubleSpinBox()
        self.angleMaxSpin.minimum = -360.0
        self.angleMaxSpin.maximum = 360.0
        self.angleMaxSpin.value = 180.0
        self.angleMaxSpin.singleStep = 1.0
        angleLimitsRow.addWidget(self.angleLimitsCheck)
        angleLimitsRow.addWidget(qt.QLabel("min"))
        angleLimitsRow.addWidget(self.angleMinSpin)
        angleLimitsRow.addWidget(qt.QLabel("max"))
        angleLimitsRow.addWidget(self.angleMaxSpin)
        plotLayout.addRow("Angle limits for stacked cut:", angleLimitsRow)

        self.plotButton = qt.QPushButton("Plot now")
        self.plotButton.toolTip = "Plot the selected mode immediately."
        self.plotButton.connect("clicked(bool)", self.onPlotNow)
        plotLayout.addRow(self.plotButton)

        # ------------------------------------------------------------------
        # Status / info
        # ------------------------------------------------------------------
        infoBox = ctk.ctkCollapsibleButton()
        infoBox.text = "Status"
        infoBox.collapsed = False
        self.layout.addWidget(infoBox)
        infoLayout = qt.QFormLayout(infoBox)

        self.listenerStatusLabel = qt.QLabel("Stopped")
        infoLayout.addRow("OSC status:", self.listenerStatusLabel)

        self.loadedPathLabel = qt.QLabel("(none)")
        self.loadedPathLabel.wordWrap = True
        infoLayout.addRow("Loaded HRTF:", self.loadedPathLabel)

        self.positionLabel = qt.QLabel("(0.0, 0.0, 0.0)")
        self.positionLabel.wordWrap = True
        infoLayout.addRow("Last OSC position:", self.positionLabel)

        self.snappedLabel = qt.QLabel("(none)")
        self.snappedLabel.wordWrap = True
        infoLayout.addRow("Last plotted sample:", self.snappedLabel)

        self.messageLabel = qt.QLabel("Ready")
        self.messageLabel.wordWrap = True
        infoLayout.addRow("Message:", self.messageLabel)

        inspectRow = qt.QHBoxLayout()
        self.inspectSummaryLabel = qt.QLabel("No SOFA loaded")
        self.inspectSummaryLabel.wordWrap = True
        self.inspectButton = qt.QPushButton("Open SOFA inspect()")
        self.inspectButton.toolTip = "Open the SOFA inspect() output in a separate scrollable popup."
        self.inspectButton.connect("clicked(bool)", self.onShowInspect)
        inspectRow.addWidget(self.inspectSummaryLabel, 1)
        inspectRow.addWidget(self.inspectButton)
        infoLayout.addRow("SOFA inspect():", inspectRow)

        self._inspectDialog = None
        self._inspectTextEdit = None

        self.layout.addStretch(1)

        self.onOverrideToggled(False)
        self.onPlotModeChanged(0)

        # Polling timer for GUI-safe updates/plotting
        self.timer = qt.QTimer()
        self.timer.setInterval(200)
        self.timer.connect("timeout()", self.onTimer)
        self.timer.start()

        self.refreshGUI()

    def cleanup(self):
        try:
            if hasattr(self, "timer") and self.timer:
                self.timer.stop()
        except Exception:
            pass
        if hasattr(self, "logic") and self.logic:
            self.logic.cleanup()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _plotArgs(self):
        ylim = None
        if bool(self.ylimCheck.checked):
            ymin = float(self.ylimMinSpin.value)
            ymax = float(self.ylimMaxSpin.value)
            if ymin < ymax:
                ylim = (ymin, ymax)

        angleLimits = None
        if bool(self.angleLimitsCheck.checked):
            amin = float(self.angleMinSpin.value)
            amax = float(self.angleMaxSpin.value)
            if amin < amax:
                angleLimits = (amin, amax)

        return dict(
            plotMode=str(self.plotModeCombo.currentText),
            plotStyle=str(self.plotStyleCombo.currentText),
            earIndex=int(self.earCombo.currentIndex),
            representation=str(self.representationCombo.currentText),
            overrideAngles=bool(self.overrideAnglesCheck.checked),
            overrideAzDeg=float(self.azSlider.value),
            overrideElDeg=float(self.elSlider.value),
            cutType=str(self.cutTypeCombo.currentText),
            cutValue=float(self.cutValueSpin.value),
            angleStep=float(self.angleStepSpin.value),
            earMode=str(self.earModeCombo.currentText),
            ylimDb=ylim,
            angleLimits=angleLimits,
        )

    def _markPlotPending(self):
        # Used when the override sliders change: the next timer tick can refresh the plot.
        try:
            self.logic.markPendingUpdate()
        except Exception:
            pass

    def onOverrideToggled(self, checked):
        self.azSlider.enabled = bool(checked)
        self.elSlider.enabled = bool(checked)
        self._markPlotPending()
        self.refreshGUI()

    def onOverrideAngleChanged(self, value):
        if bool(self.overrideAnglesCheck.checked):
            self._markPlotPending()

    def onPlotModeChanged(self, index):
        mode = str(self.plotModeCombo.currentText)
        isStacked = (mode == "stacked cut")
        self.cutTypeCombo.enabled = isStacked
        self.cutValueSpin.enabled = isStacked
        self.angleStepSpin.enabled = isStacked
        self.earModeCombo.enabled = isStacked
        self.ylimCheck.enabled = isStacked
        self.ylimMinSpin.enabled = isStacked
        self.ylimMaxSpin.enabled = isStacked
        self.angleLimitsCheck.enabled = isStacked
        self.angleMinSpin.enabled = isStacked
        self.angleMaxSpin.enabled = isStacked

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def onInstallDependencies(self):
        try:
            self.logic.ensureDependencies(interactive=True)
            slicer.util.infoDisplay(
                "Dependencies are available in Slicer's Python environment.",
                windowTitle="Dependencies",
            )
            self.refreshGUI()
        except Exception as e:
            slicer.util.errorDisplay(f"Dependency installation failed:\n{e}")

    def onBrowseSofa(self):
        filePath = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Select SOFA file",
            "",
            "SOFA files (*.sofa);;All files (*)",
        )
        if filePath:
            self.sofaPathEdit.text = filePath

    def onLoadSofa(self):
        path = self.sofaPathEdit.text.strip()
        if not path:
            slicer.util.errorDisplay("Please choose a SOFA file.")
            return
        try:
            self.logic.loadSofaFile(path)
            self.refreshGUI()
        except Exception as e:
            slicer.util.errorDisplay(f"Error loading SOFA file:\n{e}")

    def onStartOsc(self):
        try:
            self.logic.ensureDependencies(interactive=True)
            self.logic.updateDelaySec = float(self.updateDelaySpin.value)
            self.logic.startServer(self.ipEdit.text.strip(), int(self.portSpin.value))
            self.refreshGUI()
        except Exception as e:
            slicer.util.errorDisplay(f"Could not start OSC listener:\n{e}")

    def onStopOsc(self):
        try:
            self.logic.stopServer()
            self.refreshGUI()
        except Exception as e:
            slicer.util.errorDisplay(f"Could not stop OSC listener:\n{e}")

    def onPlotNow(self):
        try:
            self.logic.ensureDependencies(interactive=True)
            self.logic.plotCurrent(**self._plotArgs())
            self.refreshGUI()
        except Exception as e:
            slicer.util.errorDisplay(f"Plot error:\n{e}")
            traceback.print_exc()

    def onTimer(self):
        try:
            self.logic.updateDelaySec = float(self.updateDelaySpin.value)
            if self.logic.isAutoPlotDue():
                self.logic.consumePendingUpdate()
                self.logic.plotCurrent(**self._plotArgs())
            self.refreshGUI()
        except Exception:
            # Avoid spamming the user with dialogs from timer callbacks.
            traceback.print_exc()

    def onShowInspect(self):
        state = self.logic.getStateSnapshot()
        text = state.get("sofaInspectText", "") or "No SOFA inspect() text available."

        if self._inspectDialog is None:
            self._inspectDialog = qt.QDialog(slicer.util.mainWindow())
            self._inspectDialog.setWindowTitle("SOFA inspect()")
            self._inspectDialog.resize(900, 650)
            layout = qt.QVBoxLayout(self._inspectDialog)

            self._inspectTextEdit = qt.QPlainTextEdit()
            self._inspectTextEdit.readOnly = True
            self._inspectTextEdit.setLineWrapMode(qt.QPlainTextEdit.NoWrap)
            font = qt.QFont("Courier New")
            font.setStyleHint(qt.QFont.Monospace)
            self._inspectTextEdit.setFont(font)
            layout.addWidget(self._inspectTextEdit)

            buttonRow = qt.QHBoxLayout()
            copyButton = qt.QPushButton("Copy text")
            closeButton = qt.QPushButton("Close")
            copyButton.connect("clicked(bool)", lambda checked=False: qt.QApplication.clipboard().setText(self._inspectTextEdit.toPlainText()))
            closeButton.connect("clicked(bool)", self._inspectDialog.close)
            buttonRow.addStretch(1)
            buttonRow.addWidget(copyButton)
            buttonRow.addWidget(closeButton)
            layout.addLayout(buttonRow)

        self._inspectTextEdit.setPlainText(text)
        try:
            self._inspectTextEdit.verticalScrollBar().setValue(0)
        except Exception:
            pass
        self._inspectDialog.show()
        try:
            self._inspectDialog.raise_()
            self._inspectDialog.activateWindow()
        except Exception:
            pass

    def refreshGUI(self):
        state = self.logic.getStateSnapshot()
        self.listenerStatusLabel.text = state["serverStatus"]
        self.loadedPathLabel.text = state["lastHrtfPath"] or "(none)"
        self.positionLabel.text = "({:.4f}, {:.4f}, {:.4f})".format(*state["lastPosition"])

        snapped = state["lastSnappedInfo"]
        if snapped:
            self.snappedLabel.text = (
                "idx={idx} | source={source} | cart=({x:.4f}, {y:.4f}, {z:.4f}) | "
                "az={az:.2f}°, el={el:.2f}° | lat={lat:.2f}°, pol={pol:.2f}°"
            ).format(**snapped)
        else:
            self.snappedLabel.text = "(none)"

        self.messageLabel.text = state["statusMessage"]
        inspectText = state.get("sofaInspectText", "") or ""
        if inspectText:
            firstLine = inspectText.splitlines()[0] if inspectText.splitlines() else "SOFA inspect() available"
            self.inspectSummaryLabel.text = f"Available ({len(inspectText.splitlines())} lines): {firstLine}"
            self.inspectButton.enabled = True
        else:
            self.inspectSummaryLabel.text = "No SOFA loaded"
            self.inspectButton.enabled = False


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class ETSE_UV__SofaHrtfPlotterLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

        self.lock = threading.Lock()

        self.dependenciesReady = False
        self.Dispatcher = None
        self.osc_server = None
        self.pf = None
        self.sf = None
        self.mpl = None
        self.plt = None

        self.lastHrtfPath = None
        self.lastPosition = (0.0, 0.0, 0.0)
        self.lastUpdateTime = 0.0
        self.updateDelaySec = 0.1
        self.updateNeeded = False

        self.dataIr = None
        self.srcCoords = None
        self.recCoords = None

        self.server = None
        self.serverThread = None
        self.serverIp = "127.0.0.1"
        self.serverPort = 12346

        self.statusMessage = "Ready"
        self.sofaInspectText = ""
        self.lastSnappedInfo = None

        # Plot node/window references kept so repeated plots do not clutter the scene.
        self.plotNodes = {}
        self._nativePlotNodes = []
        self._mplDialog = None
        self._mplImageLabel = None
        self._mplScrollArea = None
        self._mplLastImagePath = None
        self._mplRawPixmap = None
        self._mplFitToWindow = True

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------
    def ensureDependencies(self, interactive=False):
        if self.dependenciesReady:
            return

        ensure_packages(
            [
                ("pythonosc", "python-osc"),
                ("pyfar", "pyfar"),
                ("sofar", "sofar"),
                ("matplotlib", "matplotlib"),
            ],
            interactive=interactive,
            module_name="ETSE-UV SOFA HRTF Plotter",
        )

        try:
            from pythonosc.dispatcher import Dispatcher
            from pythonosc import osc_server
            import pyfar as pf
            import sofar as sf
        except Exception as e:
            raise RuntimeError(
                "Python packages were not imported successfully. "
                "You may need to restart Slicer once after installation.\n\n"
                f"Original error: {e}"
            ) from e

        self.Dispatcher = Dispatcher
        self.osc_server = osc_server
        self.pf = pf
        self.sf = sf
        self.dependenciesReady = True

    def _ensureMatplotlib(self):
        if self.plt is not None:
            return self.plt

        try:
            import matplotlib as mpl
            try:
                # Slicer documentation recommends avoiding the default Tk backend.
                # Use Agg and render figures into a Qt image widget managed by Slicer.
                mpl.use("Agg", force=True)
            except Exception:
                pass
            import matplotlib.pyplot as plt
            plt.ioff()
        except Exception as e:
            raise RuntimeError(
                "Could not import/use matplotlib for pyfar-style plotting. "
                "Use 'Slicer native line plot' or reinstall matplotlib.\n\n"
                f"Original error: {e}"
            ) from e

        self.mpl = mpl
        self.plt = plt
        return plt
    def startServer(self, ip, port):
        self.ensureDependencies(interactive=False)
        self.stopServer()

        self.serverIp = ip
        self.serverPort = int(port)

        dispatcher = self.Dispatcher()
        dispatcher.map("/3DTI-OSC/source1/pos", self._handlePosition)
        dispatcher.map("/3DTI-OSC/listener/HRTFPath", self._handleHrtfPath)

        self.server = self.osc_server.ThreadingOSCUDPServer((self.serverIp, self.serverPort), dispatcher)

        def _serve():
            print(f"[OSC] Listening on {self.serverIp}:{self.serverPort}")
            try:
                self.server.serve_forever()
            except Exception:
                traceback.print_exc()

        self.serverThread = threading.Thread(target=_serve, daemon=True)
        self.serverThread.start()
        self.statusMessage = f"OSC listener running on {self.serverIp}:{self.serverPort}"

    def stopServer(self):
        if self.server is not None:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                traceback.print_exc()

        if self.serverThread is not None:
            try:
                self.serverThread.join(timeout=1.0)
            except Exception:
                traceback.print_exc()

        self.server = None
        self.serverThread = None
        self.statusMessage = "OSC listener stopped"

    def cleanup(self):
        self.stopServer()

    # ------------------------------------------------------------------
    # OSC handlers (background thread)
    # ------------------------------------------------------------------
    def _handlePosition(self, address, *args):
        if len(args) < 3:
            self.statusMessage = f"Invalid OSC position payload on {address}: {args}"
            return

        try:
            x = float(args[0])
            y = float(args[1])
            z = float(args[2])
        except Exception:
            self.statusMessage = f"Could not parse OSC position payload: {args}"
            return

        with self.lock:
            self.lastPosition = (x, y, z)
            self.lastUpdateTime = time.time()
            self.updateNeeded = True
        self.statusMessage = "Received OSC position update"

    def _handleHrtfPath(self, address, path):
        try:
            path = str(path)
            if path == self.lastHrtfPath:
                return
            self.loadSofaFile(path)
        except Exception as e:
            self.statusMessage = f"Error loading SOFA from OSC path: {e}"
            traceback.print_exc()

    # ------------------------------------------------------------------
    # SOFA loading
    # ------------------------------------------------------------------
    def loadSofaFile(self, path):
        self.ensureDependencies(interactive=False)

        if not path:
            raise RuntimeError("Empty SOFA path.")
        if not os.path.exists(path):
            raise RuntimeError(f"SOFA file does not exist:\n{path}")

        self.statusMessage = f"Loading SOFA: {path}"

        sofaObj = self.sf.read_sofa(path)
        inspectText = self._inspectSofaObject(sofaObj)
        dataIr, srcCoords, recCoords = self.pf.io.read_sofa(path)

        if not inspectText:
            inspectText = self._fallbackSofaSummary(path, dataIr, srcCoords, recCoords)

        with self.lock:
            self.lastHrtfPath = path
            self.sofaInspectText = inspectText
            self.dataIr = dataIr
            self.srcCoords = srcCoords
            self.recCoords = recCoords
            self.lastUpdateTime = time.time()
            self.updateNeeded = True

        self.statusMessage = f"Loaded SOFA: {os.path.basename(path)}"

    def _inspectSofaObject(self, sofaObj):
        """Return text from sofa.inspect(), including functions that print and return None."""
        if sofaObj is None or not hasattr(sofaObj, "inspect"):
            return ""

        buffer = io.StringIO()
        result = None
        try:
            with contextlib.redirect_stdout(buffer):
                result = sofaObj.inspect()
        except Exception as e:
            return f"SOFA inspect() failed: {e}"

        printed = buffer.getvalue().strip()
        if printed:
            return printed
        if result is not None:
            return str(result)
        return ""

    def _fallbackSofaSummary(self, path, dataIr, srcCoords, recCoords):
        lines = [
            f"SOFA file: {path}",
            "inspect() returned no text, so this fallback summary was generated.",
        ]
        try:
            lines.append(f"Signal sampling rate: {float(dataIr.sampling_rate):.1f} Hz")
        except Exception:
            pass
        try:
            lines.append(f"Signal time shape: {np.asarray(dataIr.time).shape}")
        except Exception:
            pass
        try:
            lines.append(f"Source coordinates: {self._coord_count(srcCoords)}")
        except Exception:
            pass
        try:
            lines.append(f"Receiver coordinates: {self._coord_count(recCoords)}")
        except Exception:
            pass
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plot scheduling
    # ------------------------------------------------------------------
    def isAutoPlotDue(self):
        with self.lock:
            return bool(
                self.updateNeeded
                and self.dataIr is not None
                and self.srcCoords is not None
                and (time.time() - self.lastUpdateTime) >= self.updateDelaySec
            )

    def consumePendingUpdate(self):
        with self.lock:
            self.updateNeeded = False

    def markPendingUpdate(self):
        with self.lock:
            if self.dataIr is not None and self.srcCoords is not None:
                self.lastUpdateTime = time.time()
                self.updateNeeded = True

    # ------------------------------------------------------------------
    # Public plotting entry
    # ------------------------------------------------------------------
    def plotCurrent(
        self,
        plotMode="single sample",
        plotStyle="Slicer native line plot",
        earIndex=0,
        representation="freq",
        overrideAngles=False,
        overrideAzDeg=0.0,
        overrideElDeg=0.0,
        cutType="elevation",
        cutValue=0.0,
        angleStep=30.0,
        earMode="left",
        ylimDb=None,
        angleLimits=None,
    ):
        self.ensureDependencies(interactive=False)

        with self.lock:
            if self.dataIr is None or self.srcCoords is None:
                raise RuntimeError("No SOFA/HRTF data loaded.")
            dataIr = self.dataIr
            srcCoords = self.srcCoords
            lastPosition = self.lastPosition

        sample = self._selectedSample(srcCoords, lastPosition, overrideAngles, overrideAzDeg, overrideElDeg)

        mode = (plotMode or "single sample").lower().strip()
        style = (plotStyle or "Slicer native line plot").lower().strip()
        representation = (representation or "freq").lower().strip()

        if mode == "single sample" and style.startswith("slicer") and representation in ("freq", "time", "phase"):
            self._plotNativeSingleSample(dataIr, sample, int(earIndex), representation)
        else:
            if mode == "single sample":
                self._plotPyfarSingleSample(dataIr, sample, int(earIndex), representation)
            elif mode == "single sample overview":
                self._plotPyfarSingleOverview(dataIr, sample, int(earIndex))
            elif mode == "elevation cut":
                self._plotPyfarElevationCut(dataIr, srcCoords, sample)
            elif mode == "lateral cut":
                self._plotPyfarLateralCut(dataIr, srcCoords, sample)
            elif mode == "all views one ear":
                self._plotPyfarAllViewsOneEar(dataIr, srcCoords, sample, int(earIndex))
            elif mode == "all views both ears":
                self._plotPyfarAllViewsBothEars(dataIr, srcCoords, sample)
            elif mode == "stacked cut":
                self._plotPyfarStackedCut(
                    dataIr,
                    srcCoords,
                    cutType=cutType,
                    cutValue=float(cutValue),
                    angleStep=float(angleStep),
                    earMode=earMode,
                    representation=representation,
                    ylimDb=ylimDb,
                    angleLimits=angleLimits,
                )
            elif mode == "coordinates":
                self._plotCoordinates(srcCoords, sample)
            else:
                raise RuntimeError(f"Unknown plot mode: {plotMode}")

        self._updateLastSnappedInfo(sample)
        sourceText = "override Az/El" if bool(overrideAngles) else "OSC/cartesian"
        self.statusMessage = f"Updated {mode} plot using {sourceText} sample"

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _coord_count(self, coords):
        try:
            return len(coords)
        except Exception:
            try:
                return int(self._cart_array(coords).shape[0])
            except Exception:
                return 0

    def _normalise_index(self, idx):
        if isinstance(idx, tuple):
            idx = idx[0]
        return int(np.asarray(idx).ravel()[0])

    def _as_nx3(self, arr):
        arr = np.asarray(arr, dtype=float)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            if arr.size < 3:
                raise RuntimeError(f"Coordinate array has too few values: {arr.shape}")
            return arr[:3].reshape(1, 3)
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] >= 3:
            return arr[:, :3]
        if arr.shape[0] >= 3:
            return arr[:3, :].T
        raise RuntimeError(f"Cannot interpret coordinate array shape as Nx3: {arr.shape}")

    def _cart_array(self, coords):
        if hasattr(coords, "cartesian"):
            return self._as_nx3(coords.cartesian)
        if hasattr(coords, "get_cart"):
            return self._as_nx3(coords.get_cart())
        return self._as_nx3(np.asarray(coords))

    def _spherical_elevation_array(self, coords):
        if hasattr(coords, "get_sph"):
            try:
                return self._as_nx3(coords.get_sph(convention="top_elev", unit="deg"))
            except Exception:
                pass
        if hasattr(coords, "spherical_elevation"):
            arr = self._as_nx3(coords.spherical_elevation)
            # pyfar properties are radians for angular components.
            arr[:, 0] = np.rad2deg(arr[:, 0])
            arr[:, 1] = np.rad2deg(arr[:, 1])
            return arr

        cart = self._cart_array(coords)
        x, y, z = cart[:, 0], cart[:, 1], cart[:, 2]
        r = np.linalg.norm(cart, axis=1)
        r_safe = np.maximum(r, np.finfo(float).eps)
        az = np.rad2deg(np.arctan2(y, x)) % 360.0
        el = np.rad2deg(np.arcsin(np.clip(z / r_safe, -1.0, 1.0)))
        return np.column_stack((az, el, r))

    def _spherical_side_array(self, coords):
        if hasattr(coords, "get_sph"):
            try:
                return self._as_nx3(coords.get_sph(convention="side", unit="deg"))
            except Exception:
                pass
        for attr in ("spherical_side", "spherical_lateral", "spherical_front"):
            if hasattr(coords, attr):
                try:
                    arr = self._as_nx3(getattr(coords, attr))
                    arr[:, 0] = np.rad2deg(arr[:, 0])
                    arr[:, 1] = np.rad2deg(arr[:, 1])
                    return arr
                except Exception:
                    pass

        # Fallback approximation of pyfar side coordinates.
        cart = self._cart_array(coords)
        x, y, z = cart[:, 0], cart[:, 1], cart[:, 2]
        r = np.linalg.norm(cart, axis=1)
        r_safe = np.maximum(r, np.finfo(float).eps)
        lateral = np.rad2deg(np.arcsin(np.clip(y / r_safe, -1.0, 1.0)))
        polar = np.rad2deg(np.arctan2(z, x)) % 360.0
        return np.column_stack((lateral, polar, r))

    def _single_coord_from_index(self, srcCoords, idx):
        try:
            return srcCoords[idx]
        except Exception:
            cart = self._cart_array(srcCoords)[idx]
            try:
                return self.pf.Coordinates(float(cart[0]), float(cart[1]), float(cart[2]))
            except Exception:
                return self.pf.Coordinates(float(cart[0]), float(cart[1]), float(cart[2]), "cart")

    def _make_cart_coord(self, x, y, z):
        try:
            return self.pf.Coordinates(float(x), float(y), float(z))
        except Exception:
            return self.pf.Coordinates(float(x), float(y), float(z), "cart")

    def _angle_diff_deg(self, a, b):
        return (np.asarray(a, dtype=float) - float(b) + 180.0) % 360.0 - 180.0

    def _nearest_index_from_az_el(self, srcCoords, azDeg, elDeg):
        sph = self._spherical_elevation_array(srcCoords)
        da = self._angle_diff_deg(sph[:, 0], azDeg)
        de = sph[:, 1] - float(elDeg)
        score = da * da + de * de
        return int(np.nanargmin(score))

    def _nearest_index_from_cart(self, srcCoords, x, y, z):
        # Prefer pyfar's own nearest-search when available.
        try:
            queryPoint = self._make_cart_coord(x, y, z)
            idx, _ = srcCoords.find_nearest(queryPoint, k=1)
            return self._normalise_index(idx)
        except Exception:
            pass
        try:
            idx, _mask = srcCoords.find_nearest_k(float(x), float(y), float(z), k=1, domain="cart")
            return self._normalise_index(idx)
        except Exception:
            pass

        cart = self._cart_array(srcCoords)
        query = np.array([float(x), float(y), float(z)], dtype=float)
        return int(np.argmin(np.linalg.norm(cart - query[None, :], axis=1)))

    def _mask_single(self, srcCoords, idx):
        n = self._coord_count(srcCoords)
        mask = np.zeros(n, dtype=bool)
        if 0 <= int(idx) < n:
            mask[int(idx)] = True
        return mask

    def _selectedSample(self, srcCoords, lastPosition, overrideAngles, overrideAzDeg, overrideElDeg):
        if bool(overrideAngles):
            idx = self._nearest_index_from_az_el(srcCoords, overrideAzDeg, overrideElDeg)
            source = "override Az/El"
        else:
            x, y, z = lastPosition
            idx = self._nearest_index_from_cart(srcCoords, x, y, z)
            source = "OSC/cartesian"

        singleCoord = self._single_coord_from_index(srcCoords, idx)
        cart = self._cart_array(singleCoord)[0]
        sph = self._spherical_elevation_array(singleCoord)[0]
        side = self._spherical_side_array(singleCoord)[0]
        return {
            "idx": int(idx),
            "mask": self._mask_single(srcCoords, idx),
            "coord": singleCoord,
            "cart": cart,
            "az": float(sph[0]),
            "el": float(sph[1]),
            "lat": float(side[0]),
            "pol": float(side[1]),
            "source": source,
        }

    def _slice_mask(self, srcCoords, kind, value, tol_values=(1.0, 2.0, 5.0, 10.0)):
        kind = (kind or "elevation").lower()
        # Use pyfar's find_slice if present.
        for tol in tol_values:
            try:
                _idx, mask = srcCoords.find_slice(kind, unit="deg", value=float(value), tol=float(tol))
                mask = np.asarray(mask, dtype=bool).ravel()
                if np.sum(mask) > 0:
                    return mask
            except Exception:
                pass

        # Fallback: choose coordinates within tolerance, and if none, closest angle band.
        if kind == "lateral":
            arr = self._spherical_side_array(srcCoords)
            values = arr[:, 0]
        else:
            arr = self._spherical_elevation_array(srcCoords)
            values = arr[:, 1]

        for tol in tol_values:
            mask = np.abs(values - float(value)) <= float(tol)
            if np.sum(mask) > 0:
                return mask

        idx = int(np.nanargmin(np.abs(values - float(value))))
        mask = np.zeros_like(values, dtype=bool)
        mask[idx] = True
        return mask

    def _sorted_cut(self, dataIr, srcCoords, mask, coordinateConvention, varyingColumn, earIndex=None):
        mask = np.asarray(mask, dtype=bool).ravel()
        if coordinateConvention == "side":
            allCoords = self._spherical_side_array(srcCoords)
        else:
            allCoords = self._spherical_elevation_array(srcCoords)

        if mask.size != allCoords.shape[0]:
            raise RuntimeError(
                f"Cut mask length ({mask.size}) does not match number of source coordinates "
                f"({allCoords.shape[0]})."
            )

        sampleIndices = np.flatnonzero(mask)
        if sampleIndices.size == 0:
            raise RuntimeError("Cut mask did not select any SOFA samples.")

        coords = allCoords[sampleIndices]
        angles = coords[:, int(varyingColumn)]
        sortIdx = np.argsort(angles)
        sortedSampleIndices = sampleIndices[sortIdx]
        anglesSorted = angles[sortIdx]

        signalsSorted = self._select_signal(dataIr, sortedSampleIndices, earIndex)
        return anglesSorted, signalsSorted
    def _select_signal(self, dataIr, sampleIndices, earIndex=None):
        """Return a pyfar Signal selected over the SOFA position dimension.

        Avoid chained indexing such as dataIr[mask][sortIdx], because pyfar can
        interpret it as indexing too many channel dimensions in Slicer's Python.
        """
        sampleIndices = np.asarray(sampleIndices, dtype=int)
        scalar = (sampleIndices.ndim == 0)
        sampleIndices1d = sampleIndices.reshape(-1)

        # First try native pyfar indexing.
        try:
            if earIndex is None:
                out = dataIr[int(sampleIndices1d[0])] if scalar else dataIr[sampleIndices1d]
            else:
                out = (
                    dataIr[int(sampleIndices1d[0]), int(earIndex)]
                    if scalar else dataIr[sampleIndices1d, int(earIndex)]
                )
            try:
                return out.copy()
            except Exception:
                return out
        except Exception:
            pass

        # Fallback: build a new Signal from the raw time array.
        arr = np.asarray(dataIr.time)
        if arr.ndim < 2:
            raise RuntimeError(f"Unexpected pyfar time array shape: {arr.shape}")

        try:
            if earIndex is None:
                raw = arr[sampleIndices1d, ...]
            else:
                raw = arr[sampleIndices1d, int(earIndex), ...]
        except Exception as e:
            raise RuntimeError(
                f"Could not select HRTF samples from raw time array. "
                f"time shape={arr.shape}, indices shape={sampleIndices1d.shape}, "
                f"earIndex={earIndex}. Original error: {e}"
            ) from e

        if scalar:
            raw = np.squeeze(raw, axis=0)
        try:
            return self.pf.Signal(raw, float(dataIr.sampling_rate), domain="time")
        except TypeError:
            return self.pf.Signal(raw, float(dataIr.sampling_rate))

    def _select_from_signal(self, signal, indices, earIndex=None):
        """Select indices from an already-created pyfar Signal without chained boolean indexing."""
        indices = np.asarray(indices, dtype=int).reshape(-1)
        scalar = indices.size == 1
        try:
            if earIndex is None:
                out = signal[int(indices[0])] if scalar else signal[indices]
            else:
                out = signal[int(indices[0]), int(earIndex)] if scalar else signal[indices, int(earIndex)]
            try:
                return out.copy()
            except Exception:
                return out
        except Exception:
            arr = np.asarray(signal.time)
            if earIndex is None:
                raw = arr[indices, ...]
            else:
                raw = arr[indices, int(earIndex), ...]
            if scalar:
                raw = np.squeeze(raw, axis=0)
            try:
                return self.pf.Signal(raw, float(signal.sampling_rate), domain="time")
            except TypeError:
                return self.pf.Signal(raw, float(signal.sampling_rate))
    def _signal_at(self, dataIr, idx, earIndex=None):
        return self._select_signal(dataIr, np.asarray(int(idx), dtype=int), earIndex)
    def _raw_ir(self, dataIr, idx, earIndex):
        arr = np.asarray(dataIr.time)
        try:
            ir = arr[int(idx), int(earIndex), ...]
        except Exception:
            try:
                ir = arr[int(idx)][int(earIndex)]
            except Exception as e:
                raise RuntimeError(
                    f"Could not extract raw IR. time shape={arr.shape}, "
                    f"idx={idx}, earIndex={earIndex}. Original error: {e}"
                ) from e
        ir = np.asarray(ir, dtype=float).squeeze()
        if ir.ndim != 1:
            raise RuntimeError(f"Unexpected IR shape for single sample: {ir.shape}")
        if ir.size == 0:
            raise RuntimeError("Selected IR is empty.")
        return ir
    def _plotNativeSingleSample(self, dataIr, sample, earIndex, representation):
        idx = int(sample["idx"])
        earLabel = "Left" if int(earIndex) == 0 else "Right"
        ir = self._raw_ir(dataIr, idx, earIndex)
        fs = float(dataIr.sampling_rate)

        if representation == "time":
            xData = np.arange(ir.size, dtype=float) / fs
            yData = ir
            xName = "Time_s"
            yName = "Amplitude"
            xTitle = "Time (s)"
            yTitle = "Amplitude"
            title = (
                f"Single HRTF sample - {earLabel} ear - Time - "
                f"Az {sample['az']:.1f}°, El {sample['el']:.1f}°"
            )
        elif representation == "phase":
            fftVals = np.fft.rfft(ir)
            xData = np.fft.rfftfreq(ir.size, d=1.0 / fs)
            yData = np.angle(fftVals)
            xName = "Frequency_Hz"
            yName = "Phase_rad"
            xTitle = "Frequency (Hz)"
            yTitle = "Phase (rad)"
            title = (
                f"Single HRTF sample - {earLabel} ear - Phase - "
                f"Az {sample['az']:.1f}°, El {sample['el']:.1f}°"
            )
        else:
            fftVals = np.fft.rfft(ir)
            xData = np.fft.rfftfreq(ir.size, d=1.0 / fs)
            yData = 20.0 * np.log10(np.maximum(np.abs(fftVals), np.finfo(float).eps))
            xName = "Frequency_Hz"
            yName = "Magnitude_dB"
            xTitle = "Frequency (Hz)"
            yTitle = "Magnitude (dB)"
            title = (
                f"Single HRTF sample - {earLabel} ear - Freq - "
                f"Az {sample['az']:.1f}°, El {sample['el']:.1f}°"
            )

        self._showSlicerPlot(xData, yData, xName, yName, xTitle, yTitle, title)

    def _showSlicerPlot(self, xData, yData, xName, yName, xTitle, yTitle, title):
        arrayData = np.column_stack((np.asarray(xData, dtype=float), np.asarray(yData, dtype=float)))

        # Remove the previous native plot nodes before creating a new plot. Reusing
        # slicer.util.plot(nodes=...) across different column names can leave a
        # stale plot series without a valid X column and triggers:
        #   [VTK] No X column is set (index 0).
        for node in list(getattr(self, "_nativePlotNodes", [])):
            try:
                if node is not None and slicer.mrmlScene.GetNodeByID(node.GetID()):
                    slicer.mrmlScene.RemoveNode(node)
            except Exception:
                pass
        self._nativePlotNodes = []

        tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "ETSE_UV_SOFA_HRTF_PlotTable")
        slicer.util.updateTableFromArray(tableNode, arrayData)
        table = tableNode.GetTable()
        table.GetColumn(0).SetName(str(xName))
        table.GetColumn(1).SetName(str(yName))

        plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "ETSE_UV_SOFA_HRTF_PlotSeries")
        plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
        plotSeriesNode.SetXColumnName(str(xName))
        plotSeriesNode.SetYColumnName(str(yName))
        try:
            plotSeriesNode.SetPlotType(plotSeriesNode.PlotTypeLine)
        except Exception:
            pass
        try:
            plotSeriesNode.SetMarkerStyle(plotSeriesNode.MarkerStyleNone)
        except Exception:
            pass

        chartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", "ETSE_UV_SOFA_HRTF_PlotChart")
        chartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())
        chartNode.SetTitle(str(title))
        chartNode.SetXAxisTitle(str(xTitle))
        chartNode.SetYAxisTitle(str(yTitle))

        self._nativePlotNodes = [chartNode, plotSeriesNode, tableNode]
        slicer.modules.plots.logic().ShowChartInLayout(chartNode)
    def _clean_xticks(self, ax, max_ticks=15, rotation=45, fontsize=8):
        ticks = list(ax.get_xticks())
        if not ticks:
            return
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        if len(labels) != len(ticks) or all(label == "" for label in labels):
            labels = [f"{t:.1f}" for t in ticks]
        keepIdx = np.linspace(0, len(ticks) - 1, min(max_ticks, len(ticks)), dtype=int)
        ticksNew = [ticks[i] for i in keepIdx]
        labelsNew = [labels[i] for i in keepIdx]
        ax.set_xticks(ticksNew)
        ax.set_xticklabels(labelsNew, rotation=rotation, fontsize=fontsize)

    def _screen_available_size(self):
        """Best-effort available screen size for sizing plot popups."""
        try:
            geom = slicer.util.mainWindow().screen().availableGeometry()
            return int(geom.width()), int(geom.height())
        except Exception:
            pass
        try:
            geom = qt.QApplication.desktop().availableGeometry(slicer.util.mainWindow())
            return int(geom.width()), int(geom.height())
        except Exception:
            return 1600, 900

    def _update_mpl_label_pixmap(self, fitToWindow=None):
        if fitToWindow is not None:
            self._mplFitToWindow = bool(fitToWindow)
        if self._mplRawPixmap is None or self._mplImageLabel is None:
            return

        pixmap = self._mplRawPixmap
        if self._mplFitToWindow and self._mplScrollArea is not None:
            try:
                viewport = self._mplScrollArea.viewport()
                maxW = max(200, int(viewport.width()) - 8)
                maxH = max(200, int(viewport.height()) - 8)
            except Exception:
                screenW, screenH = self._screen_available_size()
                maxW, maxH = int(screenW * 0.92), int(screenH * 0.82)
            shown = pixmap.scaled(maxW, maxH, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)
        else:
            shown = pixmap

        self._mplImageLabel.setPixmap(shown)
        self._mplImageLabel.resize(shown.size())

    def _show_mpl(self, fig, preferredDpi=110, maximize=True):
        self._ensureMatplotlib()

        try:
            fig.tight_layout()
        except Exception:
            pass

        try:
            fig.canvas.draw()
        except Exception:
            pass

        try:
            baseDir = getattr(slicer.app, "temporaryPath", "") or tempfile.gettempdir()
        except Exception:
            baseDir = tempfile.gettempdir()
        os.makedirs(baseDir, exist_ok=True)
        imagePath = os.path.join(baseDir, "ETSE_UV_SOFA_HRTF_matplotlib.png")

        # Do not use plt.show()/plt.pause() in Slicer. Render with Agg and put
        # the rasterized figure in a Qt dialog. Lower DPI keeps huge multi-panel
        # figures manageable; the image can still be viewed at 100%.
        fig.savefig(imagePath, dpi=int(preferredDpi), bbox_inches="tight")
        self._mplLastImagePath = imagePath

        pixmap = qt.QPixmap(imagePath)
        if pixmap.isNull():
            raise RuntimeError(f"Could not render matplotlib figure to image: {imagePath}")
        self._mplRawPixmap = pixmap

        if self._mplDialog is None:
            self._mplDialog = qt.QDialog(slicer.util.mainWindow())
            self._mplDialog.setWindowTitle("ETSE-UV SOFA HRTF pyfar/matplotlib figure")
            self._mplDialog.setWindowFlags(self._mplDialog.windowFlags() | qt.Qt.WindowMaximizeButtonHint)
            layout = qt.QVBoxLayout(self._mplDialog)

            self._mplScrollArea = qt.QScrollArea()
            self._mplScrollArea.setWidgetResizable(False)
            self._mplImageLabel = qt.QLabel()
            self._mplImageLabel.setAlignment(qt.Qt.AlignCenter)
            self._mplImageLabel.setBackgroundRole(qt.QPalette.Base)
            self._mplImageLabel.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)
            self._mplImageLabel.setScaledContents(False)
            self._mplScrollArea.setWidget(self._mplImageLabel)
            layout.addWidget(self._mplScrollArea, 1)

            buttonRow = qt.QHBoxLayout()
            fitButton = qt.QPushButton("Fit to window")
            oneToOneButton = qt.QPushButton("100%")
            maximizeButton = qt.QPushButton("Maximize")
            saveButton = qt.QPushButton("Save PNG as...")
            closeButton = qt.QPushButton("Close")
            fitButton.connect("clicked(bool)", lambda checked=False: self._update_mpl_label_pixmap(True))
            oneToOneButton.connect("clicked(bool)", lambda checked=False: self._update_mpl_label_pixmap(False))
            maximizeButton.connect("clicked(bool)", self._mplDialog.showMaximized)
            saveButton.connect("clicked(bool)", self._saveLastMplPngAs)
            closeButton.connect("clicked(bool)", self._mplDialog.close)
            buttonRow.addWidget(fitButton)
            buttonRow.addWidget(oneToOneButton)
            buttonRow.addWidget(maximizeButton)
            buttonRow.addStretch(1)
            buttonRow.addWidget(saveButton)
            buttonRow.addWidget(closeButton)
            layout.addLayout(buttonRow)

        screenW, screenH = self._screen_available_size()
        self._mplDialog.resize(int(screenW * 0.92), int(screenH * 0.86))
        self._mplDialog.show()
        if maximize:
            try:
                self._mplDialog.showMaximized()
            except Exception:
                pass
        self._update_mpl_label_pixmap(True)
        try:
            self._mplDialog.raise_()
            self._mplDialog.activateWindow()
        except Exception:
            pass

    def _saveLastMplPngAs(self):
        if not self._mplLastImagePath or not os.path.exists(self._mplLastImagePath):
            slicer.util.errorDisplay("No rendered matplotlib image is available yet.")
            return
        outPath = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Save rendered HRTF figure",
            "ETSE_UV_SOFA_HRTF_plot.png",
            "PNG image (*.png);;All files (*)",
        )
        if not outPath:
            return
        if not str(outPath).lower().endswith(".png"):
            outPath += ".png"
        try:
            import shutil
            shutil.copyfile(self._mplLastImagePath, outPath)
            slicer.util.infoDisplay(f"Saved figure to:\n{outPath}")
        except Exception as e:
            slicer.util.errorDisplay(f"Could not save figure:\n{e}")

    def _plot_signal_representation(self, signal, ax, representation):
        rep = (representation or "freq").lower()
        try:
            if rep == "time":
                self.pf.plot.time(signal, ax=ax)
            elif rep == "phase":
                self.pf.plot.phase(signal, ax=ax)
            elif rep == "spectrogram":
                try:
                    self.pf.plot.spectrogram(signal, unit="ms", window_length=64, ax=ax)
                except TypeError:
                    self.pf.plot.spectrogram(signal, ax=ax)
            else:
                self.pf.plot.freq(signal, ax=ax)
            return
        except Exception as pyfar_error:
            # Fallback for Slicer/pyfar combinations where a plot helper fails.
            try:
                timeData = np.asarray(signal.time, dtype=float).squeeze()
                fs = float(signal.sampling_rate)
                if timeData.ndim > 1:
                    timeData = timeData.reshape(-1, timeData.shape[-1])[0]
                if rep == "time":
                    x = np.arange(timeData.size, dtype=float) / fs
                    ax.plot(x, timeData)
                    ax.set_xlabel("Time in s")
                    ax.set_ylabel("Amplitude")
                elif rep == "phase":
                    fftVals = np.fft.rfft(timeData)
                    f = np.fft.rfftfreq(timeData.size, d=1.0 / fs)
                    ax.semilogx(f[1:], np.angle(fftVals)[1:])
                    ax.set_xlabel("Frequency in Hz")
                    ax.set_ylabel("Phase in radians")
                elif rep == "spectrogram":
                    nfft = min(64, max(8, timeData.size // 4))
                    ax.specgram(timeData, Fs=fs, NFFT=nfft, noverlap=0)
                    ax.set_xlabel("Time in s")
                    ax.set_ylabel("Frequency in Hz")
                else:
                    fftVals = np.fft.rfft(timeData)
                    f = np.fft.rfftfreq(timeData.size, d=1.0 / fs)
                    magDb = 20.0 * np.log10(np.maximum(np.abs(fftVals), np.finfo(float).eps))
                    ax.semilogx(f[1:], magDb[1:])
                    ax.set_xlabel("Frequency in Hz")
                    ax.set_ylabel("Magnitude in dB")
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Could not plot {rep} representation with pyfar or fallback matplotlib. "
                    f"pyfar error: {pyfar_error}; fallback error: {fallback_error}"
                ) from fallback_error
    def _plot_2d_representation(self, signals, ax, representation):
        rep = (representation or "freq").lower()
        try:
            if rep == "time":
                self.pf.plot.time_2d(signals, ax=ax)
            else:
                # pyfar has no phase_2d/spectrogram_2d equivalent in this workflow; use freq_2d for cuts.
                self.pf.plot.freq_2d(signals, ax=ax)
            return
        except Exception as pyfar_error:
            # Fallback for 2D cuts: build a basic image from raw time/frequency data.
            try:
                timeData = np.asarray(signals.time, dtype=float)
                fs = float(signals.sampling_rate)
                timeData = np.squeeze(timeData)
                if timeData.ndim == 1:
                    timeData = timeData.reshape(1, -1)
                if timeData.ndim > 2:
                    timeData = timeData.reshape(-1, timeData.shape[-1])

                if rep == "time":
                    image = timeData
                    ylabel = "Time sample"
                else:
                    spec = np.fft.rfft(timeData, axis=-1)
                    image = 20.0 * np.log10(np.maximum(np.abs(spec), np.finfo(float).eps)).T
                    freqs = np.fft.rfftfreq(timeData.shape[-1], d=1.0 / fs)
                    # Keep the displayed frequency axis simple and robust.
                    extent = [0, max(0, timeData.shape[0] - 1), freqs[0], freqs[-1]]
                    ax.imshow(image, aspect="auto", origin="lower", extent=extent)
                    ax.set_ylabel("Frequency in Hz")
                    return

                ax.imshow(image.T, aspect="auto", origin="lower")
                ax.set_ylabel(ylabel)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Could not plot 2D {rep} representation with pyfar or fallback matplotlib. "
                    f"pyfar error: {pyfar_error}; fallback error: {fallback_error}"
                ) from fallback_error
    def _plotPyfarSingleSample(self, dataIr, sample, earIndex, representation):
        plt = self._ensureMatplotlib()
        earLabel = "Left" if int(earIndex) == 0 else "Right"
        sig = self._signal_at(dataIr, sample["idx"], earIndex)

        fig = plt.figure("Single Sample", figsize=(10, 5))
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        self._plot_signal_representation(sig, ax, representation)
        ax.set_title(
            f"{representation.title()} - {earLabel} ear - "
            f"Az {sample['az']:.1f}°, El {sample['el']:.1f}°"
        )
        self._show_mpl(fig)

    def _plotPyfarSingleOverview(self, dataIr, sample, earIndex):
        plt = self._ensureMatplotlib()
        earLabel = "Left" if int(earIndex) == 0 else "Right"
        sig = self._signal_at(dataIr, sample["idx"], earIndex)

        fig = plt.figure("Single Sample Overview", figsize=(14, 10))
        fig.clf()
        axs = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
        self._plot_signal_representation(sig, axs[0], "time")
        axs[0].set_title("Time")
        self._plot_signal_representation(sig, axs[1], "freq")
        axs[1].set_title("Frequency")
        self._plot_signal_representation(sig, axs[2], "phase")
        axs[2].set_title("Phase")
        self._plot_signal_representation(sig, axs[3], "spectrogram")
        axs[3].set_title("Spectrogram")
        fig.suptitle(
            f"Single Sample Overview - {earLabel} ear - "
            f"Az {sample['az']:.1f}°, El {sample['el']:.1f}°"
        )
        self._show_mpl(fig)

    def _plotPyfarElevationCut(self, dataIr, srcCoords, sample):
        plt = self._ensureMatplotlib()
        fig = plt.figure("Elevation View", figsize=(20, 12))
        fig.clf()

        ax3d0 = fig.add_subplot(2, 3, 1, projection="3d")
        ax3d1 = fig.add_subplot(2, 3, 2, projection="3d")
        ax3d2 = fig.add_subplot(2, 3, 3, projection="3d")
        ax0 = fig.add_subplot(2, 3, 4)
        ax1 = fig.add_subplot(2, 3, 5)
        ax2 = fig.add_subplot(2, 3, 6)

        elevMask = self._slice_mask(srcCoords, "elevation", sample["el"])
        anglesSorted, cutBoth = self._sorted_cut(dataIr, srcCoords, elevMask, "top_elev", 0, earIndex=None)
        idxAz = int(np.argmin(np.abs(self._angle_diff_deg(anglesSorted, sample["az"])))) if len(anglesSorted) else 0

        try:
            srcCoords.show(mask=sample["mask"], ax=ax3d0)
            ax3d0.set_title("Single Sample 3D")
        except Exception:
            ax3d0.axis("off")
        try:
            srcCoords.show(mask=elevMask, ax=ax3d1)
            ax3d1.set_title(f"Elevation 3D @{sample['el']:.1f}°")
            srcCoords.show(mask=elevMask, ax=ax3d2)
            ax3d2.set_title(f"Elevation 3D @{sample['el']:.1f}°; sample Az {sample['az']:.1f}°")
        except Exception:
            ax3d1.axis("off")
            ax3d2.axis("off")

        self.pf.plot.freq(self._signal_at(dataIr, sample["idx"], None), ax=ax0)
        ax0.set_title("Freq Single Sample")

        for ear, ax, title in [(0, ax1, "Freq Elevation Cut Left"), (1, ax2, "Freq Elevation Cut Right")]:
            try:
                sig = cutBoth[:, ear]
                self.pf.plot.freq_2d(sig, ax=ax)
                ax.axvline(idxAz, color="red", linestyle="--")
                ax.set_title(title)
                ax.set_xticks(np.arange(len(anglesSorted)))
                ax.set_xticklabels([f"{a:.1f}" for a in anglesSorted], rotation=45, fontsize=8)
                ax.set_xlabel("Azimuth (deg)")
                self._clean_xticks(ax)
            except Exception:
                ax.axis("off")

        self._show_mpl(fig)

    def _plotPyfarLateralCut(self, dataIr, srcCoords, sample):
        plt = self._ensureMatplotlib()
        fig = plt.figure("Lateral View", figsize=(20, 12))
        fig.clf()

        ax3d0 = fig.add_subplot(2, 3, 1, projection="3d")
        ax3d1 = fig.add_subplot(2, 3, 2, projection="3d")
        ax3d2 = fig.add_subplot(2, 3, 3, projection="3d")
        ax0 = fig.add_subplot(2, 3, 4)
        ax1 = fig.add_subplot(2, 3, 5)
        ax2 = fig.add_subplot(2, 3, 6)

        latMask = self._slice_mask(srcCoords, "lateral", sample["lat"])
        anglesSorted, cutBoth = self._sorted_cut(dataIr, srcCoords, latMask, "side", 1, earIndex=None)
        idxPol = int(np.argmin(np.abs(anglesSorted - sample["pol"]))) if len(anglesSorted) else 0

        try:
            srcCoords.show(mask=sample["mask"], ax=ax3d0)
            ax3d0.set_title("Single Sample 3D")
        except Exception:
            ax3d0.axis("off")
        try:
            srcCoords.show(mask=latMask, ax=ax3d1)
            ax3d1.set_title(f"Lateral 3D @{sample['lat']:.1f}°")
            srcCoords.show(mask=latMask, ax=ax3d2)
            ax3d2.set_title(f"Lateral 3D @{sample['lat']:.1f}°; sample Pol {sample['pol']:.1f}°")
        except Exception:
            ax3d1.axis("off")
            ax3d2.axis("off")

        self.pf.plot.freq(self._signal_at(dataIr, sample["idx"], None), ax=ax0)
        ax0.set_title("Freq Single Sample")

        for ear, ax, title in [(0, ax1, "Freq Lateral Cut Left"), (1, ax2, "Freq Lateral Cut Right")]:
            try:
                sig = cutBoth[:, ear]
                self.pf.plot.freq_2d(sig, ax=ax)
                ax.axvline(idxPol, color="red", linestyle="--")
                ax.set_title(title)
                ax.set_xticks(np.arange(len(anglesSorted)))
                ax.set_xticklabels([f"{p:.1f}" for p in anglesSorted], rotation=45, fontsize=8)
                ax.set_xlabel("Polar (deg)")
                self._clean_xticks(ax)
            except Exception:
                ax.axis("off")

        self._show_mpl(fig)

    def _plotPyfarAllViewsOneEar(self, dataIr, srcCoords, sample, earIndex):
        plt = self._ensureMatplotlib()
        earLabel = "Left" if int(earIndex) == 0 else "Right"
        fig = plt.figure("All Views (One Ear)", figsize=(20, 14))
        fig.clf()

        axs = [fig.add_subplot(3, 3, i + 1, projection="3d" if i < 3 else None) for i in range(9)]
        elevMask = self._slice_mask(srcCoords, "elevation", sample["el"])
        latMask = self._slice_mask(srcCoords, "lateral", sample["lat"])

        azAngles, elevCut = self._sorted_cut(dataIr, srcCoords, elevMask, "top_elev", 0, earIndex=earIndex)
        polAngles, latCut = self._sorted_cut(dataIr, srcCoords, latMask, "side", 1, earIndex=earIndex)
        idxAz = int(np.argmin(np.abs(self._angle_diff_deg(azAngles, sample["az"])))) if len(azAngles) else 0
        idxPol = int(np.argmin(np.abs(polAngles - sample["pol"]))) if len(polAngles) else 0

        try:
            srcCoords.show(mask=sample["mask"], ax=axs[0])
            axs[0].set_title("Single Sample 3D")
            srcCoords.show(mask=elevMask, ax=axs[1])
            axs[1].set_title(f"Elevation Cut @{sample['el']:.1f}°")
            srcCoords.show(mask=latMask, ax=axs[2])
            axs[2].set_title(f"Lateral Cut @{sample['lat']:.1f}°")
        except Exception:
            for ax in axs[:3]:
                ax.axis("off")

        sig = self._signal_at(dataIr, sample["idx"], earIndex)
        self.pf.plot.time(sig, ax=axs[3]); axs[3].set_title("Time Single Sample")
        self.pf.plot.freq(sig, ax=axs[6]); axs[6].set_title("Freq Single Sample")

        cutPlots = [
            (elevCut, azAngles, idxAz, axs[4], "Time Elevation Cut", "Azimuth (deg)", "time"),
            (latCut, polAngles, idxPol, axs[5], "Time Lateral Cut", "Polar (deg)", "time"),
            (elevCut, azAngles, idxAz, axs[7], "Freq Elevation Cut", "Azimuth (deg)", "freq"),
            (latCut, polAngles, idxPol, axs[8], "Freq Lateral Cut", "Polar (deg)", "freq"),
        ]
        for sigs, angles, idxLine, ax, title, xlabel, rep in cutPlots:
            try:
                self._plot_2d_representation(sigs, ax, rep)
                ax.axvline(idxLine, color="red", linestyle="--")
                ax.set_title(title)
                ax.set_xticks(np.arange(len(angles)))
                ax.set_xticklabels([f"{a:.1f}" for a in angles], rotation=45, fontsize=8)
                ax.set_xlabel(xlabel)
                self._clean_xticks(ax)
            except Exception:
                ax.axis("off")

        fig.suptitle(f"All Views - {earLabel} ear - Az {sample['az']:.1f}°, El {sample['el']:.1f}°")
        self._show_mpl(fig)

    def _plotPyfarAllViewsBothEars(self, dataIr, srcCoords, sample):
        plt = self._ensureMatplotlib()
        fig = plt.figure("All Views (Both Ears)", figsize=(20, 20))
        fig.clf()
        axs = [fig.add_subplot(5, 3, i + 1, projection="3d" if i < 3 else None) for i in range(15)]

        elevMask = self._slice_mask(srcCoords, "elevation", sample["el"])
        latMask = self._slice_mask(srcCoords, "lateral", sample["lat"])

        azAngles, elevBoth = self._sorted_cut(dataIr, srcCoords, elevMask, "top_elev", 0, earIndex=None)
        polAngles, latBoth = self._sorted_cut(dataIr, srcCoords, latMask, "side", 1, earIndex=None)
        idxAz = int(np.argmin(np.abs(self._angle_diff_deg(azAngles, sample["az"])))) if len(azAngles) else 0
        idxPol = int(np.argmin(np.abs(polAngles - sample["pol"]))) if len(polAngles) else 0

        try:
            srcCoords.show(mask=sample["mask"], ax=axs[0]); axs[0].set_title("Single Sample 3D")
            srcCoords.show(mask=elevMask, ax=axs[1]); axs[1].set_title(f"Elevation 3D @{sample['el']:.1f}°")
            srcCoords.show(mask=latMask, ax=axs[2]); axs[2].set_title(f"Lateral 3D @{sample['lat']:.1f}°")
        except Exception:
            for ax in axs[:3]:
                ax.axis("off")

        bothSig = self._signal_at(dataIr, sample["idx"], None)
        self.pf.plot.time(bothSig, ax=axs[3]); axs[3].set_title("Time Single Sample")
        self.pf.plot.freq(bothSig, ax=axs[6]); axs[6].set_title("Freq Single Sample")
        self.pf.plot.phase(bothSig, ax=axs[9]); axs[9].set_title("Phase Single Sample")
        self._plot_signal_representation(bothSig, axs[12], "spectrogram")
        axs[12].set_title("Spectrogram Single Sample")

        cutPlots = [
            (elevBoth[:, 0], azAngles, idxAz, axs[4], "Time Elevation Cut (Left)", "Azimuth (deg)", "time"),
            (latBoth[:, 0], polAngles, idxPol, axs[5], "Time Lateral Cut (Left)", "Polar (deg)", "time"),
            (elevBoth[:, 0], azAngles, idxAz, axs[7], "Freq Elevation Cut (Left)", "Azimuth (deg)", "freq"),
            (latBoth[:, 0], polAngles, idxPol, axs[8], "Freq Lateral Cut (Left)", "Polar (deg)", "freq"),
            (elevBoth[:, 1], azAngles, idxAz, axs[10], "Time Elevation Cut (Right)", "Azimuth (deg)", "time"),
            (latBoth[:, 1], polAngles, idxPol, axs[11], "Time Lateral Cut (Right)", "Polar (deg)", "time"),
            (elevBoth[:, 1], azAngles, idxAz, axs[13], "Freq Elevation Cut (Right)", "Azimuth (deg)", "freq"),
            (latBoth[:, 1], polAngles, idxPol, axs[14], "Freq Lateral Cut (Right)", "Polar (deg)", "freq"),
        ]

        for sigs, angles, idxLine, ax, title, xlabel, rep in cutPlots:
            try:
                self._plot_2d_representation(sigs, ax, rep)
                ax.axvline(idxLine, color="red", linestyle="--")
                ax.set_title(title)
                ax.set_xticks(np.arange(len(angles)))
                ax.set_xticklabels([f"{a:.1f}" for a in angles], rotation=45, fontsize=8)
                ax.set_xlabel(xlabel)
                self._clean_xticks(ax)
            except Exception:
                ax.axis("off")

        fig.suptitle(f"All Views - Both Ears - Az {sample['az']:.1f}°, El {sample['el']:.1f}°")
        self._show_mpl(fig)

    def _wrap_to_signed_deg(self, angles):
        return (np.asarray(angles, dtype=float) + 180.0) % 360.0 - 180.0

    def _circ_dist_signed_deg(self, a, b):
        return np.abs((np.asarray(a, dtype=float) - float(b) + 180.0) % 360.0 - 180.0)

    def _target_angles_for_paper_stack(self, angleStep, angleLimits):
        step = max(float(angleStep), 1.0)
        if angleLimits is None:
            return np.arange(0.0, 360.0, step)

        amin, amax = float(angleLimits[0]), float(angleLimits[1])
        if amax <= amin:
            return np.arange(0.0, 360.0, step)

        # Interpret the UI values as a circular interval, but order labels from 0°
        # when the interval wraps through zero. Example: -60..240 -> 0,30,...,240,300,330.
        vals = np.arange(np.floor(amin / step) * step, amax + 0.5 * step, step)
        vals = np.mod(vals, 360.0)
        vals = np.unique(np.round(vals, 8))
        vals = sorted(vals, key=lambda v: (v < 0.0, v))
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0:
            return np.arange(0.0, 360.0, step)

        # Put 0.. before wrapped angles to match common HRTF paper layout.
        vals = vals[vals < 360.0]
        zeroFirst = vals[vals < 300.0]
        wrappedTail = vals[vals >= 300.0]
        return np.concatenate((zeroFirst, wrappedTail))

    def _mag_db_from_raw_ir(self, ir, fs, logReference=1.0):
        ir = np.asarray(ir, dtype=float).squeeze()
        if ir.ndim != 1:
            ir = ir.reshape(-1)
        freqs = np.fft.rfftfreq(ir.size, d=1.0 / float(fs))
        H = np.fft.rfft(ir)
        magDb = 20.0 * np.log10(np.maximum(np.abs(H) / float(logReference), 1e-12))
        return freqs, magDb

    def _plotPyfarStackedCut(self, dataIr, srcCoords, cutType, cutValue, angleStep, earMode, representation, ylimDb, angleLimits):
        # New default: paper-style vertically shifted HRTF curves, similar to the
        # Fig.31-like notebook function. For non-frequency representations, keep
        # the older per-row behaviour because the paper layout is frequency-based.
        representation = (representation or "freq").lower()
        if representation != "freq":
            return self._plotPyfarStackedCutRows(dataIr, srcCoords, cutType, cutValue, angleStep, earMode, representation, ylimDb, angleLimits)

        plt = self._ensureMatplotlib()
        cutType = (cutType or "elevation").lower()
        earMode = (earMode or "both").lower()
        cutValue = float(cutValue)

        if cutType == "lateral":
            cutMask = self._slice_mask(srcCoords, "lateral", cutValue)
            allAngles = self._spherical_side_array(srcCoords)[:, 1]  # varying polar
            cutLabel = "LATERAL"
            rightAxisLabel = "Polar angle"
            defaultTitle = f"Lateral cut @ {cutValue:.1f}°"
            if abs(cutValue) < 1e-9:
                defaultTitle = "Median Plane (lateral cut @ 0°)"
        else:
            cutMask = self._slice_mask(srcCoords, "elevation", cutValue)
            allAngles = self._spherical_elevation_array(srcCoords)[:, 0]  # varying azimuth
            cutLabel = "ELEVATION"
            rightAxisLabel = "Azimuth angle"
            defaultTitle = f"Elevation cut @ {cutValue:.1f}°"

        sampleIdx = np.flatnonzero(np.asarray(cutMask, dtype=bool).ravel())
        if sampleIdx.size == 0:
            raise RuntimeError(f"No samples found for {cutType} cut at {cutValue}°.")

        varyingSigned = self._wrap_to_signed_deg(allAngles[sampleIdx])
        sortIdx = np.argsort(varyingSigned)
        sampleIdxSorted = sampleIdx[sortIdx]
        varyingSignedSorted = varyingSigned[sortIdx]

        targetAngles = self._target_angles_for_paper_stack(angleStep, angleLimits)
        selectedSampleIdx = []
        requestedAngles = []
        snappedAngles = []
        snappedErrors = []

        for requested in targetAngles:
            requestedSigned = float(self._wrap_to_signed_deg([requested])[0])
            distances = self._circ_dist_signed_deg(varyingSignedSorted, requestedSigned)
            pos = int(np.nanargmin(distances))
            sampleI = int(sampleIdxSorted[pos])
            if sampleI in selectedSampleIdx:
                continue
            selectedSampleIdx.append(sampleI)
            requestedAngles.append(float(requested % 360.0))
            snappedAngles.append(float(varyingSignedSorted[pos]))
            snappedErrors.append(float(distances[pos]))

        if not selectedSampleIdx:
            raise RuntimeError("No samples selected for stacked cut.")

        print(f"\n{cutLabel} CUT @ {cutValue:.1f}°")
        print("requested -> snapped (error)")
        for req, snap, err in zip(requestedAngles, snappedAngles, snappedErrors):
            print(f"{req:7.1f}° -> {snap:7.1f}°   ({err:.2f}°)")

        fs = float(dataIr.sampling_rate)
        fMaxDefault = min(24000.0, fs / 2.0)
        freqLimits = (0.0, fMaxDefault)
        fmin, fmax = freqLimits
        if fmax <= fmin:
            fmax = fs / 2.0

        shiftDb = 40.0
        topMarginDb = 5.0
        bottomMarginDb = shiftDb
        logReference = 1.0
        n = len(selectedSampleIdx)

        fig, ax = plt.subplots(num="Stacked Cut - Paper Style", figsize=(8.0, max(7.0, 0.8 * n + 2.2)), dpi=160)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)

        try:
            colors = plt.cm.hsv(np.linspace(0.0, 1.0, n, endpoint=False))
        except Exception:
            colors = [None] * n

        baselines = []
        rightLabels = []
        plotMask = None

        for i, (sampleI, reqAngle) in enumerate(zip(selectedSampleIdx, requestedAngles)):
            baseline = -i * shiftDb
            baselines.append(baseline)
            rightLabels.append(f"{reqAngle:.0f}°")
            ax.axhline(baseline, color="0.60", linestyle="--", linewidth=1.0, zorder=0)

            curveColor = colors[i]
            if earMode in ("right", "both"):
                irR = self._raw_ir(dataIr, sampleI, 1)
                freqsHz, magR = self._mag_db_from_raw_ir(irR, fs, logReference)
                if plotMask is None:
                    plotMask = (freqsHz >= fmin) & (freqsHz <= fmax)
                    if not np.any(plotMask):
                        raise RuntimeError("Empty frequency range for stacked cut.")
                ax.plot(freqsHz[plotMask] / 1000.0, magR[plotMask] + baseline,
                        color=curveColor, linestyle="-", linewidth=1.6)

            if earMode in ("left", "both"):
                irL = self._raw_ir(dataIr, sampleI, 0)
                freqsHz, magL = self._mag_db_from_raw_ir(irL, fs, logReference)
                if plotMask is None:
                    plotMask = (freqsHz >= fmin) & (freqsHz <= fmax)
                    if not np.any(plotMask):
                        raise RuntimeError("Empty frequency range for stacked cut.")
                linestyle = (0, (2, 2)) if earMode == "both" else "-"
                ax.plot(freqsHz[plotMask] / 1000.0, magL[plotMask] + baseline,
                        color=curveColor, linestyle=linestyle, linewidth=1.6)

        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Relative amplitude (dB)")
        ax.set_xlim(fmin / 1000.0, fmax / 1000.0)
        ax.set_ylim(baselines[-1] - bottomMarginDb, topMarginDb)
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        ymin, ymax = ax.get_ylim()
        ygrid = np.arange(np.floor(ymin / 10.0) * 10.0, np.ceil(ymax / 10.0) * 10.0 + 10.0, 10.0)
        for y in ygrid:
            if np.any(np.isclose(y, baselines)):
                continue
            ax.axhline(y, color="0.85", linestyle=":", linewidth=0.8, zorder=0)

        axR = ax.twinx()
        axR.set_ylim(ax.get_ylim())
        axR.set_yticks(baselines)
        axR.set_yticklabels(rightLabels, fontsize=11)
        axR.tick_params(axis="y", length=0, pad=10)
        axR.set_ylabel(rightAxisLabel)
        axR.grid(False)

        # 10 dB scale marker.
        try:
            from matplotlib import transforms
            trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
            x0, x1 = -0.035, -0.015
            yTop = baselines[0]
            yBot = baselines[0] - 10.0
            ax.plot([x0, x0], [yBot, yTop], color="k", lw=1.2, transform=trans, clip_on=False)
            ax.plot([x0, x1], [yTop, yTop], color="k", lw=1.2, transform=trans, clip_on=False)
            ax.plot([x0, x1], [yBot, yBot], color="k", lw=1.2, transform=trans, clip_on=False)
            ax.annotate("", xy=(x0, yTop), xytext=(x0, yTop - 3.0), xycoords=trans, textcoords=trans,
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="k"))
            ax.annotate("", xy=(x0, yBot), xytext=(x0, yBot + 3.0), xycoords=trans, textcoords=trans,
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="k"))
            ax.text(-0.075, (yTop + yBot) / 2.0, "10dB", transform=trans,
                    ha="center", va="center", fontsize=12)
        except Exception:
            pass

        for spine in ax.spines.values():
            spine.set_color("red")
        for spine in axR.spines.values():
            spine.set_color("red")

        if earMode == "both":
            ax.plot([], [], color="k", linestyle="-", label="Right ear")
            ax.plot([], [], color="k", linestyle=(0, (2, 2)), label="Left ear")
            ax.legend(loc="upper right", frameon=False)
        elif earMode == "right":
            ax.plot([], [], color="k", linestyle="-", label="Right ear")
            ax.legend(loc="upper right", frameon=False)
        else:
            ax.plot([], [], color="k", linestyle="-", label="Left ear")
            ax.legend(loc="upper right", frameon=False)

        fig.suptitle(defaultTitle)
        try:
            fig.tight_layout(rect=[0, 0, 1, 0.97])
        except Exception:
            pass
        self._show_mpl(fig, preferredDpi=140, maximize=True)

    def _plotPyfarStackedCutRows(self, dataIr, srcCoords, cutType, cutValue, angleStep, earMode, representation, ylimDb, angleLimits):
        plt = self._ensureMatplotlib()
        cutType = (cutType or "elevation").lower()
        earMode = (earMode or "left").lower()
        representation = (representation or "freq").lower()

        if cutType == "lateral":
            cutMask = self._slice_mask(srcCoords, "lateral", cutValue)
            angles, sortedSignals = self._sorted_cut(dataIr, srcCoords, cutMask, "side", 1, earIndex=None)
            xlabel = "Polar angle"
        else:
            cutMask = self._slice_mask(srcCoords, "elevation", cutValue)
            angles, sortedSignals = self._sorted_cut(dataIr, srcCoords, cutMask, "top_elev", 0, earIndex=None)
            xlabel = "Azimuth angle"

        if angleLimits is not None:
            amin, amax = angleLimits
            keep = (angles >= float(amin)) & (angles <= float(amax))
            keepIdx = np.flatnonzero(keep)
            angles = angles[keepIdx]
            sortedSignals = self._select_from_signal(sortedSignals, keepIdx)
            if len(angles) == 0:
                raise RuntimeError(f"No samples found inside angle limits [{amin}, {amax}].")

        if len(angles) == 0:
            raise RuntimeError(f"No samples found for {cutType} cut at {cutValue}°.")

        targetAngles = np.arange(angles.min(), angles.max() + 0.5 * float(angleStep), float(angleStep))
        nearestIdx = [int(np.argmin(np.abs(angles - a))) for a in targetAngles]
        selectedIdx = []
        for i in nearestIdx:
            if i not in selectedIdx:
                selectedIdx.append(i)

        selectedAngles = angles[selectedIdx]
        nRows = max(1, len(selectedIdx))
        fig, axs = plt.subplots(
            nRows,
            1,
            num="Stacked Cut",
            figsize=(10, max(4, 2.2 * nRows)),
            squeeze=False,
        )
        fig.clf()
        axs = fig.subplots(nRows, 1, squeeze=False)

        for row, idxSel in enumerate(selectedIdx):
            ax = axs[row, 0]
            angleVal = selectedAngles[row]

            if earMode == "both":
                sigL = self._select_from_signal(sortedSignals, [idxSel], 0)
                sigR = self._select_from_signal(sortedSignals, [idxSel], 1)
                self._plot_signal_representation(sigL, ax, representation)
                self._plot_signal_representation(sigR, ax, representation)
            elif earMode == "right":
                sig = self._select_from_signal(sortedSignals, [idxSel], 1)
                self._plot_signal_representation(sig, ax, representation)
            else:
                sig = self._select_from_signal(sortedSignals, [idxSel], 0)
                self._plot_signal_representation(sig, ax, representation)

            ax.set_title("")
            ax.set_ylabel(f"{angleVal:.0f}°")
            if ylimDb is not None and representation == "freq":
                ax.set_ylim(float(ylimDb[0]), float(ylimDb[1]))
            if row != nRows - 1:
                ax.set_xlabel("")
                if representation == "freq":
                    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        axs[-1, 0].set_xlabel("Frequency (Hz)" if representation == "freq" else "Time / Frequency")
        fig.suptitle(f"Stacked {cutType} cut @{cutValue:.1f}° ({xlabel}; {earMode})")
        self._show_mpl(fig, preferredDpi=110, maximize=True)

    def _plotCoordinates(self, srcCoords, sample):
        plt = self._ensureMatplotlib()
        cart = self._cart_array(srcCoords)
        sph = self._spherical_elevation_array(srcCoords)
        side = self._spherical_side_array(srcCoords)

        fig = plt.figure("SOFA Source Coordinates", figsize=(16, 10))
        fig.clf()
        ax0 = fig.add_subplot(2, 2, 1, projection="3d")
        ax1 = fig.add_subplot(2, 2, 2)
        ax2 = fig.add_subplot(2, 2, 3)
        ax3 = fig.add_subplot(2, 2, 4)

        ax0.scatter(cart[:, 0], cart[:, 1], cart[:, 2], s=8)
        ax0.scatter([sample["cart"][0]], [sample["cart"][1]], [sample["cart"][2]], s=50)
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax0.set_zlabel("z")
        ax0.set_title("Cartesian source positions")

        ax1.scatter(sph[:, 0], sph[:, 1], s=8)
        ax1.scatter([sample["az"]], [sample["el"]], s=50)
        ax1.set_xlabel("Azimuth (deg)")
        ax1.set_ylabel("Elevation (deg)")
        ax1.set_title("Top elevation coordinates")

        ax2.scatter(side[:, 0], side[:, 1], s=8)
        ax2.scatter([sample["lat"]], [sample["pol"]], s=50)
        ax2.set_xlabel("Lateral (deg)")
        ax2.set_ylabel("Polar (deg)")
        ax2.set_title("Side coordinates")

        ax3.axis("off")
        ax3.text(
            0.0,
            1.0,
            "Selected sample\n"
            f"Index: {sample['idx']}\n"
            f"Source: {sample['source']}\n"
            f"Cartesian: ({sample['cart'][0]:.4f}, {sample['cart'][1]:.4f}, {sample['cart'][2]:.4f})\n"
            f"Az / El: {sample['az']:.2f}°, {sample['el']:.2f}°\n"
            f"Lat / Pol: {sample['lat']:.2f}°, {sample['pol']:.2f}°",
            va="top",
            family="monospace",
        )

        self._show_mpl(fig)

    # ------------------------------------------------------------------
    # State for GUI refresh
    # ------------------------------------------------------------------
    def _updateLastSnappedInfo(self, sample):
        cart = sample["cart"]
        with self.lock:
            self.lastSnappedInfo = {
                "idx": int(sample["idx"]),
                "source": sample["source"],
                "x": float(cart[0]),
                "y": float(cart[1]),
                "z": float(cart[2]),
                "az": float(sample["az"]),
                "el": float(sample["el"]),
                "lat": float(sample["lat"]),
                "pol": float(sample["pol"]),
            }

    def getStateSnapshot(self):
        with self.lock:
            return {
                "serverStatus": (
                    f"Running ({self.serverIp}:{self.serverPort})"
                    if self.server is not None else "Stopped"
                ),
                "lastHrtfPath": self.lastHrtfPath,
                "lastPosition": self.lastPosition,
                "lastSnappedInfo": self.lastSnappedInfo,
                "statusMessage": self.statusMessage,
                "sofaInspectText": self.sofaInspectText,
            }
