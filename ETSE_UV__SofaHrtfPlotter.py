# ETSE_UV__SofaHrtfPlotter.py
# 3D Slicer scripted module
#
# Basic tools included in this first version:
#   1) Start/stop an OSC listener.
#   2) Load an HRTF SOFA file manually or from OSC.
#   3) Plot the nearest single HRTF sample (time or frequency) inside Slicer.
#
# Notes:
#   - This version intentionally uses Slicer's native plotting infrastructure
#     instead of Matplotlib windows, which tends to be more robust inside Slicer.
#   - The blocking argparse + while True notebook pattern was replaced by:
#       * background OSC server thread
#       * QTimer polling on the main GUI thread

import os
import time
import threading
import traceback
import importlib

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
                          <li>Plot the nearest HRTF sample in time or frequency representation.</li>
                          <li>Choose left or right ear channel.</li>
                        </ul>

                        <p>The original standalone workflow was adapted to Slicer using a widget-based UI,
                        a background OSC server thread, and a Qt timer on the main GUI thread.</p>
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
            "Install python-osc, pyfar, and sofar into Slicer's Python environment if missing."
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
        self.updateDelaySpin.value = 2.0
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
        # Plot controls
        # ------------------------------------------------------------------
        plotBox = ctk.ctkCollapsibleButton()
        plotBox.text = "Single sample plot"
        self.layout.addWidget(plotBox)
        plotLayout = qt.QFormLayout(plotBox)

        self.representationCombo = qt.QComboBox()
        self.representationCombo.addItems(["freq", "time"])
        self.representationCombo.setToolTip("Choose time-domain or frequency-domain plot")
        plotLayout.addRow("Representation:", self.representationCombo)

        self.earCombo = qt.QComboBox()
        self.earCombo.addItems(["left", "right"])
        self.earCombo.setToolTip("Choose ear channel to plot")
        plotLayout.addRow("Ear:", self.earCombo)

        self.plotButton = qt.QPushButton("Plot current sample now")
        self.plotButton.toolTip = "Plot the current nearest sample immediately."
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
        infoLayout.addRow("Last snapped sample:", self.snappedLabel)

        self.messageLabel = qt.QLabel("Ready")
        self.messageLabel.wordWrap = True
        infoLayout.addRow("Message:", self.messageLabel)

        self.inspectText = qt.QPlainTextEdit()
        self.inspectText.readOnly = True
        self.inspectText.setMinimumHeight(160)
        infoLayout.addRow("SOFA inspect():", self.inspectText)

        self.layout.addStretch(1)

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
            self.logic.plotSingleSample(
                earIndex=self.earCombo.currentIndex,
                representation=self.representationCombo.currentText,
            )
            self.refreshGUI()
        except Exception as e:
            slicer.util.errorDisplay(f"Plot error:\n{e}")

    def onTimer(self):
        try:
            self.logic.updateDelaySec = float(self.updateDelaySpin.value)
            if self.logic.isAutoPlotDue():
                self.logic.consumePendingUpdate()
                self.logic.plotSingleSample(
                    earIndex=self.earCombo.currentIndex,
                    representation=self.representationCombo.currentText,
                )
            self.refreshGUI()
        except Exception:
            # Avoid spamming the user with dialogs from timer callbacks.
            traceback.print_exc()

    def refreshGUI(self):
        state = self.logic.getStateSnapshot()
        self.listenerStatusLabel.text = state["serverStatus"]
        self.loadedPathLabel.text = state["lastHrtfPath"] or "(none)"
        self.positionLabel.text = "({:.4f}, {:.4f}, {:.4f})".format(*state["lastPosition"])

        snapped = state["lastSnappedInfo"]
        if snapped:
            self.snappedLabel.text = (
                "cart=({x:.4f}, {y:.4f}, {z:.4f}) | az={az:.2f}°, el={el:.2f}°"
            ).format(**snapped)
        else:
            self.snappedLabel.text = "(none)"

        self.messageLabel.text = state["statusMessage"]
        self.inspectText.setPlainText(state["sofaInspectText"])


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

        self.lastHrtfPath = None
        self.lastPosition = (0.0, 0.0, 0.0)
        self.lastUpdateTime = 0.0
        self.updateDelaySec = 2.0
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

        # Reused Slicer plot nodes to avoid cluttering the scene
        self.plotNodes = {}

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
    # ------------------------------------------------------------------
    # OSC server
    # ------------------------------------------------------------------
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
        inspectText = str(sofaObj.inspect())
        dataIr, srcCoords, recCoords = self.pf.io.read_sofa(path)

        with self.lock:
            self.lastHrtfPath = path
            self.sofaInspectText = inspectText
            self.dataIr = dataIr
            self.srcCoords = srcCoords
            self.recCoords = recCoords
            self.lastUpdateTime = time.time()
            self.updateNeeded = True

        self.statusMessage = f"Loaded SOFA: {os.path.basename(path)}"

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

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plotSingleSample(self, earIndex=0, representation="freq"):
        self.ensureDependencies(interactive=False)

        with self.lock:
            if self.dataIr is None or self.srcCoords is None:
                raise RuntimeError("No SOFA/HRTF data loaded.")

            dataIr = self.dataIr
            srcCoords = self.srcCoords
            x, y, z = self.lastPosition

        # Current pyfar API:
        #   - query point must be a pf.Coordinates object
        #   - nearest search is done with find_nearest(...)
        queryPoint = self.pf.Coordinates(x, y, z)
        idx, _ = srcCoords.find_nearest(queryPoint, k=1)

        # Robustly reduce returned index to a plain integer
        if isinstance(idx, tuple):
            idx = tuple(np.asarray(i).ravel()[0] for i in idx)
        else:
            idx = int(np.asarray(idx).ravel()[0])

        # Get the snapped coordinate as a Coordinates object
        snappedCoords = srcCoords[idx]

        # Current pyfar API uses properties, not get_cart()/get_sph()
        snappedCart = np.asarray(snappedCoords.cartesian, dtype=float).squeeze()
        snappedX, snappedY, snappedZ = [float(v) for v in snappedCart.tolist()]

        snappedSph = np.asarray(snappedCoords.spherical_elevation, dtype=float).squeeze()
        azimuthDeg = float(np.rad2deg(snappedSph[0]))
        elevationDeg = float(np.rad2deg(snappedSph[1]))

        earLabel = "Left" if int(earIndex) == 0 else "Right"

        ir = np.asarray(dataIr.time[idx][int(earIndex)], dtype=float).squeeze()
        if ir.ndim != 1:
            raise RuntimeError(f"Unexpected IR shape for single sample: {ir.shape}")
        if ir.size == 0:
            raise RuntimeError("Selected IR is empty.")

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
                f"Az {azimuthDeg:.1f}°, El {elevationDeg:.1f}°"
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
                f"Az {azimuthDeg:.1f}°, El {elevationDeg:.1f}°"
            )

        self._showPlot(xData, yData, xName, yName, xTitle, yTitle, title)

        with self.lock:
            self.lastSnappedInfo = {
                "x": snappedX,
                "y": snappedY,
                "z": snappedZ,
                "az": azimuthDeg,
                "el": elevationDeg,
            }

        self.statusMessage = f"Updated {representation} plot for {earLabel.lower()} ear"

        
    def _showPlot(self, xData, yData, xName, yName, xTitle, yTitle, title):
        arrayData = np.column_stack((np.asarray(xData, dtype=float), np.asarray(yData, dtype=float)))

        chartNode = slicer.util.plot(
            arrayData,
            xColumnIndex=0,
            columnNames=[xName, yName],
            title=title,
            show=False,
            nodes=self.plotNodes,
        )
        chartNode.SetTitle(title)
        chartNode.SetXAxisTitle(xTitle)
        chartNode.SetYAxisTitle(yTitle)

        slicer.modules.plots.logic().ShowChartInLayout(chartNode)

    # ------------------------------------------------------------------
    # State for GUI refresh
    # ------------------------------------------------------------------
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
