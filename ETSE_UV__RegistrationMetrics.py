import os
import glob
import json
import csv

import vtk
import slicer
import vtkmodules.util.numpy_support as vtk_np
import numpy as np
import qt
import ctk
from slicer.ScriptedLoadableModule import *

from Resources.ETSE_UV__Dependencies import ensure_packages

ensure_packages(
    [("trimesh", "trimesh")],
    interactive=False,
    module_name="ETSE-UV Registration Metrics",
)

import trimesh

# ===========================
#  MODULE
# ===========================
class ETSE_UV__RegistrationMetrics(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Registration Metrics"
        parent.categories = ["ETSE_UV"]
        parent.contributors = ["ETSE-UV", "J.A. De Rus"]
        parent.helpText = """
        Calcula métricas de registro entre dos mallas (A=registrada, B=target) usando distancias a superficie:

        - Distancias ABSOLUTE (módulo de la distancia a la superficie).
        - Distancias SIGNED (positivas/negativas según el lado de la superficie).

        Para cada par A,B se calculan:
        - Estadísticos unidireccionales A→B y B→A (mean, median, std, RMS, percentiles, Hausdorff one-sided).
        - Métricas simétricas (Chamfer ABSOLUTE, Hausdorff ABSOLUTE simétrico).
        - Estadísticos SIGNED A→B y B→A.

        Modos:
        - Par único (modelos de la escena).
        - Batch por carpetas (empareja por nombre base).

        Salidas:
        - JSON con métricas por par y resumen global.
        - TXT con resumen global (incluye textos explicativos).
        - CSV por par con todas las métricas agregadas (sin distancias por vértice).
        - CSV global con resumen agregado sobre todos los pares.
        """
        parent.acknowledgementText = """
        Desarrollado en ETSE-UV, basado en ideas de SlicerMorph y módulos personalizados previos.
        """


# ===========================
#  WIDGET
# ===========================
class ETSE_UV__RegistrationMetricsWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        layout = self.layout

        self.logic = ETSE_UV__RegistrationMetricsLogic()

        # ---------- PAR ÚNICO ----------
        singleCollapsible = ctk.ctkCollapsibleButton()
        singleCollapsible.text = "Procesamiento de un Solo Par (A=registrada, B=target)"
        layout.addWidget(singleCollapsible)
        formSingle = qt.QFormLayout(singleCollapsible)

        self.singleASelector = self._modelCombo()
        self.singleBSelector = self._modelCombo()
        formSingle.addRow("Modelo A (registrado):", self.singleASelector)
        formSingle.addRow("Modelo B (target):", self.singleBSelector)

        # Opcional: pintar distancias como escalar en los modelos (single pair)
        paintBox = qt.QGroupBox("Pintar distancias en la escena (opcional)")
        paintLayout = qt.QFormLayout(paintBox)

        self.paintDistancesChk = qt.QCheckBox("Crear escalar de distancias en el/los modelos")
        self.paintDistancesChk.checked = False

        self.paintModeCombo = qt.QComboBox()
        self.paintModeCombo.addItem("ABS A→B (en modelo A)")
        self.paintModeCombo.addItem("ABS B→A (en modelo B)")
        self.paintModeCombo.addItem("SIGNED A→B (en modelo A)")
        self.paintModeCombo.addItem("SIGNED B→A (en modelo B)")

        paintLayout.addRow(self.paintDistancesChk)
        paintLayout.addRow("Modo de pintado:", self.paintModeCombo)

        formSingle.addRow(paintBox)


        self.singleExportCsvChk = qt.QCheckBox("Exportar métricas para este par")
        self.singleExportCsvChk.checked = True
        formSingle.addRow(self.singleExportCsvChk)

        self.singleJsonNameEdit = qt.QLineEdit("ETSE_UV_single_registration_metrics.json")
        self.singleTxtNameEdit = qt.QLineEdit("ETSE_UV_single_registration_metrics_summary.txt")
        formSingle.addRow("Nombre archivo JSON:", self.singleJsonNameEdit)
        formSingle.addRow("Nombre archivo TXT:", self.singleTxtNameEdit)

        self.singleRunBtn = qt.QPushButton("Calcular métricas (Par Único)")
        formSingle.addRow(self.singleRunBtn)
        self.singleRunBtn.connect("clicked()", self.onRunSingle)

        self.singleStatus = qt.QLabel("Listo.")
        formSingle.addRow(self.singleStatus)

        # ---------- BATCH ----------
        batchCollapsible = ctk.ctkCollapsibleButton()
        batchCollapsible.text = "Procesamiento por Lotes (Carpetas A y B)"
        layout.addWidget(batchCollapsible)
        formBatch = qt.QFormLayout(batchCollapsible)

        self.batchAFolderBtn = ctk.ctkDirectoryButton()
        self.batchBFolderBtn = ctk.ctkDirectoryButton()
        self.batchOutFolderBtn = ctk.ctkDirectoryButton()

        self.batchExtEdit = qt.QLineEdit("vtk vtp stl ply obj")

        self.batchExportCsvChk = qt.QCheckBox("Exportar por par")
        self.batchExportCsvChk.checked = True

        self.batchJsonNameEdit = qt.QLineEdit("ETSE_UV_registration_metrics_dataset.json")
        self.batchTxtNameEdit = qt.QLineEdit("ETSE_UV_registration_metrics_dataset_summary.txt")

        formBatch.addRow("Carpeta Modelos A (registrados):", self.batchAFolderBtn)
        formBatch.addRow("Carpeta Modelos B (target):", self.batchBFolderBtn)
        formBatch.addRow("Carpeta de salida:", self.batchOutFolderBtn)
        formBatch.addRow("Extensiones (separadas por espacio):", self.batchExtEdit)
        formBatch.addRow(self.batchExportCsvChk)
        formBatch.addRow("Nombre archivo JSON:", self.batchJsonNameEdit)
        formBatch.addRow("Nombre archivo TXT:", self.batchTxtNameEdit)

        # Opcional: modelos pintados en batch (crea modelos nuevos, no toca originales)
        paintBatchBox = qt.QGroupBox("Pintar distancias (batch, crea modelos nuevos)")
        paintBatchLayout = qt.QFormLayout(paintBatchBox)

        self.batchPaintChk = qt.QCheckBox("Crear modelos pintados con distancias")
        self.batchPaintChk.checked = False

        self.batchPaintModeCombo = qt.QComboBox()
        self.batchPaintModeCombo.addItem("ABS A→B (en modelo A)")
        self.batchPaintModeCombo.addItem("ABS B→A (en modelo B)")
        self.batchPaintModeCombo.addItem("SIGNED A→B (en modelo A)")
        self.batchPaintModeCombo.addItem("SIGNED B→A (en modelo B)")

        paintBatchLayout.addRow(self.batchPaintChk)
        paintBatchLayout.addRow("Modo de pintado:", self.batchPaintModeCombo)

        formBatch.addRow(paintBatchBox)


        self.batchRunBtn = qt.QPushButton("Calcular métricas (Batch)")
        formBatch.addRow(self.batchRunBtn)
        self.batchRunBtn.connect("clicked()", self.onRunBatch)

        self.batchStatus = qt.QLabel("Listo.")
        formBatch.addRow(self.batchStatus)

        layout.addStretch(1)

    def _modelCombo(self):
        w = slicer.qMRMLNodeComboBox()
        w.nodeTypes = ["vtkMRMLModelNode"]
        w.selectNodeUponCreation = True
        w.addEnabled = False
        w.removeEnabled = False
        w.noneEnabled = False
        w.setMRMLScene(slicer.mrmlScene)
        return w

    # ---------- PAR ÚNICO ----------
    def onRunSingle(self):
        if trimesh is None:
            slicer.util.errorDisplay("Instala 'trimesh' para usar este módulo.")
            return

        A = self.singleASelector.currentNode()
        B = self.singleBSelector.currentNode()
        if not A or not B:
            slicer.util.errorDisplay("Selecciona ambos modelos: A (registrado) y B (target).")
            return

        # Directorio de salida por defecto: carpeta de escena o HOME
        outDir = slicer.app.defaultScenePath if slicer.app.defaultScenePath else os.path.expanduser("~")

        jsonPath = os.path.join(
            outDir,
            self.singleJsonNameEdit.text.strip() or "ETSE_UV_single_registration_metrics.json"
        )
        txtPath = os.path.join(
            outDir,
            self.singleTxtNameEdit.text.strip() or "ETSE_UV_single_registration_metrics_summary.txt"
        )
        globalCsvName = os.path.splitext(os.path.basename(txtPath))[0] + "_global_summary.csv"

        try:
            # 1) calcular métricas del par
            pairRes = self.logic.compute_metrics_for_pair(
                modelA=A,
                modelB=B,
                nameA=A.GetName(),
                nameB=B.GetName()
            )

            # 2) construir GLOBAL usando SIEMPRE el agregador (aunque sólo haya 1 par)
            globalRes = self.logic.aggregate_global_metrics([pairRes])

            # Consola (par + global)
            self.logic.print_pair_to_console(pairRes)
            self.logic.print_global_to_console(globalRes)

            # 2.5) Opcional: pintar distancias como escalar en los modelos
            if self.paintDistancesChk.checked:
                mode = self.paintModeCombo.currentText
                try:
                    self.logic.paint_distances_on_models(A, B, mode)
                except Exception as e:
                    slicer.util.warningDisplay(
                        f"No se pudieron pintar las distancias en los modelos ({mode}): {e}"
                    )


            # ¿Guardamos algo en disco?
            if self.singleExportCsvChk.checked:
                dataset = {"pairs": [pairRes], "GLOBAL": globalRes}
                with open(jsonPath, "w", encoding="utf-8") as f:
                    json.dump(dataset, f, indent=2, ensure_ascii=False)
                with open(txtPath, "w", encoding="utf-8") as f:
                    f.write(self.logic.format_global_summary_txt(globalRes))

                # CSV por par
                base = self.logic.common_base_name(
                    pairRes["name_of_model_A"],
                    pairRes["name_of_model_B"]
                )
                self.logic.export_pair_metrics_to_csv(outDir, base, pairRes)

                # CSV global (summary)
                globalCsvPath = self.logic.export_global_summary_to_csv(
                    outDir, globalCsvName, globalRes
                )

                msg = (
                    f"Hecho.\n"
                    f"JSON: {jsonPath}\n"
                    f"TXT: {txtPath}\n"
                    f"CSV global: {globalCsvPath}"
                )
            else:
                # NO se guarda NADA (ni CSV, ni JSON, ni TXT)
                msg = (
                    "Hecho (sin guardar archivos en disco).\n"
                    "La casilla 'Exportar CSV de métricas para este par' está desactivada."
                )

            self.singleStatus.setText(msg)
            slicer.util.infoDisplay(msg)

        except Exception as e:
            self.singleStatus.setText(f"Error: {e}")
            slicer.util.errorDisplay(str(e))


    # ---------- BATCH ----------
    def onRunBatch(self):
        if trimesh is None:
            slicer.util.errorDisplay("Instala 'trimesh' para usar este módulo.")
            return

        dirA = self.batchAFolderBtn.directory
        dirB = self.batchBFolderBtn.directory
        outDir = self.batchOutFolderBtn.directory

        if not (os.path.isdir(dirA) and os.path.isdir(dirB) and os.path.isdir(outDir)):
            slicer.util.errorDisplay("Revisa las carpetas de A, B y salida.")
            return

        exts = [e.strip().lstrip(".") for e in self.batchExtEdit.text.split(" ") if e.strip()]
        if not exts:
            slicer.util.errorDisplay("Introduce al menos una extensión de archivo.")
            return

        jsonPath = os.path.join(
            outDir,
            self.batchJsonNameEdit.text.strip() or "ETSE_UV_registration_metrics_dataset.json"
        )
        txtPath = os.path.join(
            outDir,
            self.batchTxtNameEdit.text.strip() or "ETSE_UV_registration_metrics_dataset_summary.txt"
        )
        globalCsvName = os.path.splitext(os.path.basename(txtPath))[0] + "_global_summary.csv"
        exportAll = self.batchExportCsvChk.checked  # AHORA: controla TODO (CSV+JSON+TXT)

        self.batchStatus.setText("Buscando archivos...")
        slicer.app.processEvents()

        # Recolectar archivos de A
        input_files_A = []
        for ext in exts:
            input_files_A.extend(glob.glob(os.path.join(dirA, f"*.{ext}")))
        if not input_files_A:
            self.batchStatus.setText("No se encontraron modelos en carpeta A.")
            return

        pairs_results = []
        processed = 0

        for a_path in sorted(input_files_A):
            base = os.path.splitext(os.path.basename(a_path))[0]

            # Buscar B con la misma base + cualquier extensión permitida
            b_path = None
            for ext in exts:
                candidate = os.path.join(dirB, f"{base}.{ext}")
                if os.path.exists(candidate):
                    b_path = candidate
                    break

            if not b_path:
                self.batchStatus.setText(f"Saltando: no hay modelo B para '{os.path.basename(a_path)}'")
                slicer.app.processEvents()
                continue

            self.batchStatus.setText(
                f"Procesando par: {os.path.basename(a_path)}  ↔  {os.path.basename(b_path)}"
            )
            slicer.app.processEvents()

            A = slicer.util.loadModel(a_path)
            B = slicer.util.loadModel(b_path)
            if not A or not B:
                self.batchStatus.setText(f"Error cargando par basado en '{base}'")
                slicer.app.processEvents()
                if A:
                    slicer.mrmlScene.RemoveNode(A)
                if B:
                    slicer.mrmlScene.RemoveNode(B)
                continue

            try:
                pairRes = self.logic.compute_metrics_for_pair(
                    modelA=A,
                    modelB=B,
                    nameA=os.path.basename(a_path),
                    nameB=os.path.basename(b_path)
                )

                self.logic.print_pair_to_console(pairRes)
                pairs_results.append(pairRes)
                processed += 1

                # CSV por par SOLO si exportAll
                if exportAll:
                    base_name = self.logic.common_base_name(
                        pairRes["name_of_model_A"],
                        pairRes["name_of_model_B"]
                    )
                    self.logic.export_pair_metrics_to_csv(outDir, base_name, pairRes)
                # Modelos pintados opcionales (no modifican A ni B, crean clones)
                if hasattr(self, "batchPaintChk") and self.batchPaintChk.checked:
                    mode = self.batchPaintModeCombo.currentText
                    try:
                        self.logic.paint_distances_on_models(A, B, mode)
                    except Exception as e:
                        slicer.util.warningDisplay(
                            f"No se pudieron pintar las distancias para el par '{base}' ({mode}): {e}"
                        )
            except Exception as e:
                slicer.util.errorDisplay(f"Error procesando par '{base}': {e}")
            finally:
                slicer.mrmlScene.RemoveNode(A)
                slicer.mrmlScene.RemoveNode(B)

        # Resumen global en memoria
        globalRes = self.logic.aggregate_global_metrics(pairs_results)
        self.logic.print_global_to_console(globalRes)

        globalCsvPath = None

        if exportAll and pairs_results:
            # JSON + TXT + CSV global SOLO si exportAll == True
            dataset = {"pairs": pairs_results, "GLOBAL": globalRes}
            with open(jsonPath, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            with open(txtPath, "w", encoding="utf-8") as f:
                f.write(self.logic.format_global_summary_txt(globalRes))

            globalCsvPath = self.logic.export_global_summary_to_csv(
                outDir, globalCsvName, globalRes
            )

            msg = (
                f"Completado. {processed} pares procesados.\n"
                f"JSON: {jsonPath}\nTXT: {txtPath}\nCSV global: {globalCsvPath}"
            )
        else:
            # NO se guarda NADA a disco
            msg = (
                f"Completado. {processed} pares procesados.\n"
                f"No se han guardado archivos (CSV/JSON/TXT) porque "
                f"'Exportar CSV de métricas por par' está desactivado."
            )

        self.batchStatus.setText(msg)
        slicer.util.infoDisplay(msg)



# ===========================
#  LOGIC
# ===========================
class ETSE_UV__RegistrationMetricsLogic(ScriptedLoadableModuleLogic):

    # ---------- Consola ----------
    def print_pair_to_console(self, pairRes):
        Aname = pairRes["name_of_model_A"]
        Bname = pairRes["name_of_model_B"]
        abs_metrics = pairRes["metrics"]["absolute_distance_to_surface"]
        signed_metrics = pairRes["metrics"]["signed_distance_to_surface"]

        abs_AB = abs_metrics["A_to_B"]
        abs_BA = abs_metrics["B_to_A"]
        abs_sym = abs_metrics["symmetric_AB_and_BA"]

        print("\n[ETSE_UV__RegistrationMetrics] PAR")
        print(f"  A (registrado) : {Aname}")
        print(f"  B (target)     : {Bname}")
        print(f"  #puntos A = {pairRes['number_of_points_in_A']}, #puntos B = {pairRes['number_of_points_in_B']}")

        # ABSOLUTE A→B
        print("  ABSOLUTE distance A→B (from A vertices to closest point on B surface):")
        print(f"    mean = {abs_AB['mean_absolute_distance_from_A_vertices_to_closest_point_on_B_surface']:.6f}")
        print(f"    median = {abs_AB['median_absolute_distance_from_A_vertices_to_closest_point_on_B_surface']:.6f}")
        print(f"    std = {abs_AB['standard_deviation_absolute_distance_from_A_vertices_to_closest_point_on_B_surface']:.6f}")
        print(f"    RMS = {abs_AB['root_mean_square_absolute_distance_from_A_vertices_to_closest_point_on_B_surface']:.6f}")
        print(f"    Hausdorff one-sided max = {abs_AB['hausdorff_absolute_distance_one_sided_max_from_A_to_B']:.6f}")
        print(f"    p90 = {abs_AB['percentile_90_absolute_distance_from_A_vertices_to_closest_point_on_B_surface']:.6f}")
        print(f"    p95 = {abs_AB['percentile_95_absolute_distance_from_A_vertices_to_closest_point_on_B_surface']:.6f}")
        print(f"    p99 = {abs_AB['percentile_99_absolute_distance_from_A_vertices_to_closest_point_on_B_surface']:.6f}")
        print(f"    sum = {abs_AB['sum_of_absolute_distances_from_A_vertices_to_closest_point_on_B_surface']:.6f} "
              f"(depende del nº de puntos de A)")
        print(f"    number_of_points_in_A = {abs_AB['number_of_points_in_A']}")

        # ABSOLUTE B→A
        print("  ABSOLUTE distance B→A (from B vertices to closest point on A surface):")
        print(f"    mean = {abs_BA['mean_absolute_distance_from_B_vertices_to_closest_point_on_A_surface']:.6f}")
        print(f"    median = {abs_BA['median_absolute_distance_from_B_vertices_to_closest_point_on_A_surface']:.6f}")
        print(f"    std = {abs_BA['standard_deviation_absolute_distance_from_B_vertices_to_closest_point_on_A_surface']:.6f}")
        print(f"    RMS = {abs_BA['root_mean_square_absolute_distance_from_B_vertices_to_closest_point_on_A_surface']:.6f}")
        print(f"    Hausdorff one-sided max = {abs_BA['hausdorff_absolute_distance_one_sided_max_from_B_to_A']:.6f}")
        print(f"    p90 = {abs_BA['percentile_90_absolute_distance_from_B_vertices_to_closest_point_on_A_surface']:.6f}")
        print(f"    p95 = {abs_BA['percentile_95_absolute_distance_from_B_vertices_to_closest_point_on_A_surface']:.6f}")
        print(f"    p99 = {abs_BA['percentile_99_absolute_distance_from_B_vertices_to_closest_point_on_A_surface']:.6f}")
        print(f"    sum = {abs_BA['sum_of_absolute_distances_from_B_vertices_to_closest_point_on_A_surface']:.6f} "
              f"(depende del nº de puntos de B)")
        print(f"    number_of_points_in_B = {abs_BA['number_of_points_in_B']}")

        # ABSOLUTE symmetric (Chamfer/Hausdorff)
        print("  ABSOLUTE symmetric metrics (A<->B):")
        print(f"    CHAMFER symmetric mean of means (A→B & B→A) = "
              f"{abs_sym['chamfer_absolute_distance_symmetric_mean_of_means_between_A_and_B']:.6f}")
        print(f"    CHAMFER symmetric weighted mean (by #points in A and B) = "
              f"{abs_sym['chamfer_absolute_distance_symmetric_weighted_mean_between_A_and_B']:.6f}")
        print(f"    CHAMFER symmetric RMS of RMS (A→B & B→A) = "
              f"{abs_sym['chamfer_absolute_distance_symmetric_rms_of_rms_between_A_and_B']:.6f}")
        print(f"    HAUSDORFF symmetric max (two-sided) = "
              f"{abs_sym['hausdorff_absolute_distance_symmetric_maximum_between_A_and_B']:.6f}")
        print(f"    HAUSDORFF symmetric p95 (robust) = "
              f"{abs_sym['hausdorff_absolute_distance_symmetric_percentile_95_between_A_and_B']:.6f}")
        print(f"    HAUSDORFF symmetric p99 (robust) = "
              f"{abs_sym['hausdorff_absolute_distance_symmetric_percentile_99_between_A_and_B']:.6f}")

        # SIGNED
        signed_AB = signed_metrics["A_to_B"]
        signed_BA = signed_metrics["B_to_A"]

        print("  SIGNED distance A→B (from A vertices to signed surface of B):")
        print(f"    mean_signed = {signed_AB['mean_signed_distance_from_A_vertices_to_signed_surface_of_B']:.6f}")
        print(f"    median_signed = {signed_AB['median_signed_distance_from_A_vertices_to_signed_surface_of_B']:.6f}")
        print(f"    std_signed = {signed_AB['standard_deviation_signed_distance_from_A_vertices_to_signed_surface_of_B']:.6f}")
        print(f"    min_signed = {signed_AB['minimum_signed_distance_from_A_vertices_to_signed_surface_of_B']:.6f}")
        print(f"    max_signed = {signed_AB['maximum_signed_distance_from_A_vertices_to_signed_surface_of_B']:.6f}")
        print(f"    p90_signed = {signed_AB['percentile_90_signed_distance_from_A_vertices_to_signed_surface_of_B']:.6f}")
        print(f"    p95_signed = {signed_AB['percentile_95_signed_distance_from_A_vertices_to_signed_surface_of_B']:.6f}")
        print(f"    p99_signed = {signed_AB['percentile_99_signed_distance_from_A_vertices_to_signed_surface_of_B']:.6f}")
        print(f"    number_of_points_in_A = {signed_AB['number_of_points_in_A']}")

        print("  SIGNED distance B→A (from B vertices to signed surface of A):")
        print(f"    mean_signed = {signed_BA['mean_signed_distance_from_B_vertices_to_signed_surface_of_A']:.6f}")
        print(f"    median_signed = {signed_BA['median_signed_distance_from_B_vertices_to_signed_surface_of_A']:.6f}")
        print(f"    std_signed = {signed_BA['standard_deviation_signed_distance_from_B_vertices_to_signed_surface_of_A']:.6f}")
        print(f"    min_signed = {signed_BA['minimum_signed_distance_from_B_vertices_to_signed_surface_of_A']:.6f}")
        print(f"    max_signed = {signed_BA['maximum_signed_distance_from_B_vertices_to_signed_surface_of_A']:.6f}")
        print(f"    p90_signed = {signed_BA['percentile_90_signed_distance_from_B_vertices_to_signed_surface_of_A']:.6f}")
        print(f"    p95_signed = {signed_BA['percentile_95_signed_distance_from_B_vertices_to_signed_surface_of_A']:.6f}")
        print(f"    p99_signed = {signed_BA['percentile_99_signed_distance_from_B_vertices_to_signed_surface_of_A']:.6f}")
        print(f"    number_of_points_in_B = {signed_BA['number_of_points_in_B']}")

    def _nice_label_for_global_metric_key(self, key):
        """
        Etiquetas más legibles para el resumen global.
        Para las que no estén aquí, se usa el nombre de clave tal cual.
        """
        mapping = {
            # Symmetric ABSOLUTE (Chamfer / Hausdorff)
            "chamfer_absolute_distance_symmetric_mean_of_means_between_A_and_B":
                "Chamfer ABSOLUTE (symmetric, mean of means between A and B)",
            "chamfer_absolute_distance_symmetric_weighted_mean_between_A_and_B":
                "Chamfer ABSOLUTE (symmetric, weighted mean by #points in A and B)",
            "chamfer_absolute_distance_symmetric_rms_of_rms_between_A_and_B":
                "Chamfer ABSOLUTE (symmetric, RMS of RMS between A and B)",
            "hausdorff_absolute_distance_symmetric_maximum_between_A_and_B":
                "Hausdorff ABSOLUTE (symmetric, maximum two-sided between A and B)",
            "hausdorff_absolute_distance_symmetric_percentile_95_between_A_and_B":
                "Hausdorff ABSOLUTE (symmetric, percentile 95 two-sided between A and B)",
            "hausdorff_absolute_distance_symmetric_percentile_99_between_A_and_B":
                "Hausdorff ABSOLUTE (symmetric, percentile 99 two-sided between A and B)",
        }
        return mapping.get(key, key)

    def print_global_to_console(self, globalRes):
        print("\n[ETSE_UV__RegistrationMetrics] RESUMEN GLOBAL DEL DATASET")
        if not globalRes or globalRes.get("number_of_pairs", 0) == 0:
            print("  Sin pares procesados.")
            return
        print(f"  Número de pares: {globalRes['number_of_pairs']}")

        for key, stats in globalRes.get("metrics_aggregate_over_pairs", {}).items():
            label = self._nice_label_for_global_metric_key(key)
            print(
                f"  - {label}: "
                f"mean_over_pairs={stats['mean']:.6f} | "
                f"median_over_pairs={stats['median']:.6f} | "
                f"std_over_pairs={stats['std']:.6f} | "
                f"min_value={stats['min']:.6f} | "
                f"max_value={stats['max']:.6f} | "
                f"n_pairs_used={stats['count']}"
            )

    # ---------- Helpers generales ----------
    def common_base_name(self, nameA, nameB):
        a = os.path.splitext(os.path.basename(nameA))[0]
        b = os.path.splitext(os.path.basename(nameB))[0]
        if a == b:
            return a
        return f"{a}__vs__{b}"

    def _base_name(self, name_or_path):
        """Nombre base sin ruta ni extensión."""
        return os.path.splitext(os.path.basename(name_or_path))[0]

    def create_painted_model_with_distances(self, sourceModelNode, distances, arrayName, newNodeName):
        """
        Crea un NUEVO vtkMRMLModelNode que es un clon del modelo original
        pero con un array escalar por punto llamado `arrayName` con las distancias.
        El modelo original NO se modifica.
        """
        polyData = sourceModelNode.GetPolyData()
        if polyData is None:
            raise ValueError(
                f"El modelo '{sourceModelNode.GetName()}' no tiene PolyData para añadir escalares."
            )

        n_points = polyData.GetNumberOfPoints()
        if n_points != len(distances):
            raise ValueError(
                f"El número de puntos del modelo ({n_points}) no coincide con el tamaño "
                f"del vector de distancias ({len(distances)})."
            )

        # Copiar geometría
        polyCopy = vtk.vtkPolyData()
        polyCopy.DeepCopy(polyData)

        distances = np.asarray(distances, dtype=np.float64)
        distanceScalars = vtk_np.numpy_to_vtk(
            distances, deep=True, array_type=vtk.VTK_DOUBLE
        )
        distanceScalars.SetName(arrayName)

        pd = polyCopy.GetPointData()
        if pd.HasArray(arrayName):
            pd.RemoveArray(arrayName)
        pd.AddArray(distanceScalars)
        pd.SetActiveScalars(arrayName)
        pd.Modified()

        # Crear nuevo nodo
        newNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", newNodeName)
        newNode.SetAndObservePolyData(polyCopy)

        srcDisplay = sourceModelNode.GetDisplayNode()
        if srcDisplay:
            newDisplay = slicer.mrmlScene.CreateNodeByClass("vtkMRMLModelDisplayNode")
            slicer.mrmlScene.AddNode(newDisplay)
            newDisplay.Copy(srcDisplay)
            newNode.SetAndObserveDisplayNodeID(newDisplay.GetID())
        else:
            newNode.CreateDefaultDisplayNodes()

        disp = newNode.GetDisplayNode()
        if disp:
            disp.SetActiveScalarName(arrayName)
            disp.SetScalarVisibility(True)
            disp.Modified()

        newNode.Modified()
        slicer.app.processEvents()
        return newNode

    def _vtk_model_to_trimesh(self, modelNode):
        poly = modelNode.GetPolyData()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError(f"El modelo '{modelNode.GetName()}' no tiene puntos.")

        V = vtk_np.vtk_to_numpy(poly.GetPoints().GetData())

        if poly.GetPolys() and poly.GetPolys().GetNumberOfCells() > 0:
            raw = vtk_np.vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)
            F = raw[:, 1:]
        else:
            if V.shape[0] < 3:
                raise ValueError(
                    f"El modelo '{modelNode.GetName()}' tiene <3 puntos y sin caras, "
                    f"no se puede crear una malla válida."
                )
            F = np.array([[0, 1, 2]])

        return trimesh.Trimesh(vertices=V, faces=F, process=False)

    def _closest_absolute_distances(self, from_vertices, to_mesh, chunk_size=100000):
        """
        Distancias absolutas vértice-superficie (A→B o B→A) calculadas en bloques
        para reducir el uso máximo de memoria. El resultado es el mismo que llamar
        a trimesh.proximity.closest_point de una sola vez.
        """
        verts = np.asarray(from_vertices)
        n = verts.shape[0]

        # Preasignar array de salida
        dists = np.empty(n, dtype=np.float64)

        # Procesar en bloques
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            _, d_chunk, _ = trimesh.proximity.closest_point(to_mesh, verts[start:end])
            dists[start:end] = d_chunk
            start = end

        return dists


    def _signed_distances(self, from_vertices, to_mesh, chunk_size=100000):
        """
        Distancias con signo vértice-superficie calculadas en bloques.
        Misma lógica que signed_distance completo, pero con menor pico de memoria.
        """
        verts = np.asarray(from_vertices)
        n = verts.shape[0]

        dists = np.empty(n, dtype=np.float64)

        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            d_chunk = trimesh.proximity.signed_distance(to_mesh, verts[start:end])
            dists[start:end] = d_chunk
            start = end

        return dists


    def addDistancesAsScalarsToModel(self, modelNode, distances, arrayName="Distance"):
        """
        Añade un vector de distancias como array escalar de punto en el modelo dado
        y lo activa para visualización.
        """
        polyData = modelNode.GetPolyData()
        if polyData is None:
            raise ValueError(
                f"El modelo '{modelNode.GetName()}' no tiene PolyData para añadir escalares."
            )

        n_points = polyData.GetNumberOfPoints()
        if n_points != len(distances):
            raise ValueError(
                f"El número de puntos del modelo ({n_points}) no coincide con el tamaño "
                f"del vector de distancias ({len(distances)})."
            )

        distances = np.asarray(distances, dtype=np.float64)
        distanceScalars = vtk_np.numpy_to_vtk(
            distances, deep=True, array_type=vtk.VTK_DOUBLE
        )
        distanceScalars.SetName(arrayName)

        pd = polyData.GetPointData()
        if pd.HasArray(arrayName):
            pd.RemoveArray(arrayName)
        pd.AddArray(distanceScalars)
        pd.Modified()

        displayNode = modelNode.GetDisplayNode()
        if displayNode:
            displayNode.SetActiveScalarName(arrayName)
            displayNode.SetScalarVisibility(True)
            displayNode.Modified()

        modelNode.Modified()
        slicer.app.processEvents()

    def paint_distances_on_models(self, modelA, modelB, mode_text):
        """
        Calcula distancias vértice-superficie y crea NUEVOS modelos pintados,
        sin modificar los modelos originales.
        Devuelve el/los nuevos nodos creados (lista).

        Modos (mode_text):
          - 'ABS A→B (en modelo A)'    -> |dist| de vértices A a superficie B, pintado en clon de A
          - 'ABS B→A (en modelo B)'    -> |dist| de vértices B a superficie A, pintado en clon de B
          - 'SIGNED A→B (en modelo A)' -> dist. con signo A→B, pintado en clon de A
          - 'SIGNED B→A (en modelo B)' -> dist. con signo B→A, pintado en clon de B
        """
        if trimesh is None:
            raise RuntimeError("El módulo 'trimesh' no está disponible.")

        tmA = self._vtk_model_to_trimesh(modelA)
        tmB = self._vtk_model_to_trimesh(modelB)

        mode = mode_text.upper().strip()

        new_nodes = []
        baseA = self._base_name(modelA.GetName())
        baseB = self._base_name(modelB.GetName())

        if "A→B" in mode and "ABS" in mode:
            d = self._closest_absolute_distances(tmA.vertices, tmB)
            arrayName = "ETSE_UV_AbsDist_A_to_B"
            newName = f"Abs_{baseA}_to_{baseB}"
            new_nodes.append(
                self.create_painted_model_with_distances(modelA, d, arrayName, newName)
            )

        elif "B→A" in mode and "ABS" in mode:
            d = self._closest_absolute_distances(tmB.vertices, tmA)
            arrayName = "ETSE_UV_AbsDist_B_to_A"
            newName = f"Abs_{baseB}_to_{baseA}"
            new_nodes.append(
                self.create_painted_model_with_distances(modelB, d, arrayName, newName)
            )

        elif "A→B" in mode and "SIGNED" in mode:
            d = self._signed_distances(tmA.vertices, tmB)
            arrayName = "ETSE_UV_SignedDist_A_to_B"
            newName = f"Signed_{baseA}_to_{baseB}"
            new_nodes.append(
                self.create_painted_model_with_distances(modelA, d, arrayName, newName)
            )

        elif "B→A" in mode and "SIGNED" in mode:
            d = self._signed_distances(tmB.vertices, tmA)
            arrayName = "ETSE_UV_SignedDist_B_to_A"
            newName = f"Signed_{baseB}_to_{baseA}"
            new_nodes.append(
                self.create_painted_model_with_distances(modelB, d, arrayName, newName)
            )

        else:
            raise ValueError(f"Modo de pintado no reconocido: '{mode_text}'")

        return new_nodes


    def _stats_absolute_A_to_B(self, dists, n_points, who="A", to="B"):
        d = np.asarray(dists)
        if d.size == 0:
            raise ValueError("Vector de distancias (absolute) vacío.")

        mean = float(np.mean(d))
        median = float(np.median(d))
        std = float(np.std(d))
        rms = float(np.sqrt(np.mean(d ** 2)))
        maxv = float(np.max(d))
        p90 = float(np.percentile(d, 90.0))
        p95 = float(np.percentile(d, 95.0))
        p99 = float(np.percentile(d, 99.0))
        total = float(np.sum(d))

        out = {}
        if who == "A" and to == "B":
            out["number_of_points_in_A"] = int(n_points)
            out["mean_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"] = mean
            out["median_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"] = median
            out["standard_deviation_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"] = std
            out["root_mean_square_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"] = rms
            out["hausdorff_absolute_distance_one_sided_max_from_A_to_B"] = maxv
            out["percentile_90_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"] = p90
            out["percentile_95_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"] = p95
            out["percentile_99_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"] = p99
            out["sum_of_absolute_distances_from_A_vertices_to_closest_point_on_B_surface"] = total

        elif who == "B" and to == "A":
            out["number_of_points_in_B"] = int(n_points)
            out["mean_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"] = mean
            out["median_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"] = median
            out["standard_deviation_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"] = std
            out["root_mean_square_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"] = rms
            out["hausdorff_absolute_distance_one_sided_max_from_B_to_A"] = maxv
            out["percentile_90_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"] = p90
            out["percentile_95_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"] = p95
            out["percentile_99_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"] = p99
            out["sum_of_absolute_distances_from_B_vertices_to_closest_point_on_A_surface"] = total

        else:
            raise ValueError("Par who/to no soportado en _stats_absolute_A_to_B.")

        return out


    def _stats_signed_A_to_B(self, dists, n_points, who="A", to="B"):
        d = np.asarray(dists)
        if d.size == 0:
            raise ValueError("Vector de distancias (signed) vacío.")

        mean = float(np.mean(d))
        median = float(np.median(d))
        std = float(np.std(d))
        minv = float(np.min(d))
        maxv = float(np.max(d))
        p90 = float(np.percentile(d, 90.0))
        p95 = float(np.percentile(d, 95.0))
        p99 = float(np.percentile(d, 99.0))

        out = {}
        if who == "A" and to == "B":
            out["number_of_points_in_A"] = int(n_points)
            out["mean_signed_distance_from_A_vertices_to_signed_surface_of_B"] = mean
            out["median_signed_distance_from_A_vertices_to_signed_surface_of_B"] = median
            out["standard_deviation_signed_distance_from_A_vertices_to_signed_surface_of_B"] = std
            out["minimum_signed_distance_from_A_vertices_to_signed_surface_of_B"] = minv
            out["maximum_signed_distance_from_A_vertices_to_signed_surface_of_B"] = maxv
            out["percentile_90_signed_distance_from_A_vertices_to_signed_surface_of_B"] = p90
            out["percentile_95_signed_distance_from_A_vertices_to_signed_surface_of_B"] = p95
            out["percentile_99_signed_distance_from_A_vertices_to_signed_surface_of_B"] = p99
        elif who == "B" and to == "A":
            out["number_of_points_in_B"] = int(n_points)
            out["mean_signed_distance_from_B_vertices_to_signed_surface_of_A"] = mean
            out["median_signed_distance_from_B_vertices_to_signed_surface_of_A"] = median
            out["standard_deviation_signed_distance_from_B_vertices_to_signed_surface_of_A"] = std
            out["minimum_signed_distance_from_B_vertices_to_signed_surface_of_A"] = minv
            out["maximum_signed_distance_from_B_vertices_to_signed_surface_of_A"] = maxv
            out["percentile_90_signed_distance_from_B_vertices_to_signed_surface_of_A"] = p90
            out["percentile_95_signed_distance_from_B_vertices_to_signed_surface_of_A"] = p95
            out["percentile_99_signed_distance_from_B_vertices_to_signed_surface_of_A"] = p99
        else:
            raise ValueError("Par who/to no soportado en _stats_signed_A_to_B.")

        return out

    # ---------- Métricas por par ----------
    def compute_metrics_for_pair(self, modelA, modelB, nameA=None, nameB=None):
        """
        A: modelo registrado, B: modelo target.
        Calcula y devuelve TODAS las métricas para UN PAR:

          - Distancias ABSOLUTE (closest) A→B y B→A.
          - Métricas ABSOLUTE simétricas (Chamfer / Hausdorff).
          - Distancias SIGNED A→B y B→A.

        Chamfer y Hausdorff se calculan sobre distancias ABSOLUTE.

        Devuelve:
            pair (dict con todas las métricas para ese par).
        """
        if trimesh is None:
            raise RuntimeError("El módulo 'trimesh' no está disponible.")

        tmA = self._vtk_model_to_trimesh(modelA)
        tmB = self._vtk_model_to_trimesh(modelB)

        A = tmA.vertices
        B = tmB.vertices

        nA = A.shape[0]
        nB = B.shape[0]

        # ---------- ABSOLUTE ----------
        d_AB_abs = self._closest_absolute_distances(A, tmB)  # A→B
        d_BA_abs = self._closest_absolute_distances(B, tmA)  # B→A

        abs_A_to_B = self._stats_absolute_A_to_B(d_AB_abs, nA, who="A", to="B")
        abs_B_to_A = self._stats_absolute_A_to_B(d_BA_abs, nB, who="B", to="A")

        # Chamfer y Hausdorff ABSOLUTE
        mean_AB = abs_A_to_B["mean_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"]
        mean_BA = abs_B_to_A["mean_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"]

        rms_AB = abs_A_to_B["root_mean_square_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"]
        rms_BA = abs_B_to_A["root_mean_square_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"]

        max_AB = abs_A_to_B["hausdorff_absolute_distance_one_sided_max_from_A_to_B"]
        max_BA = abs_B_to_A["hausdorff_absolute_distance_one_sided_max_from_B_to_A"]

        p95_AB = abs_A_to_B["percentile_95_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"]
        p95_BA = abs_B_to_A["percentile_95_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"]

        p99_AB = abs_A_to_B["percentile_99_absolute_distance_from_A_vertices_to_closest_point_on_B_surface"]
        p99_BA = abs_B_to_A["percentile_99_absolute_distance_from_B_vertices_to_closest_point_on_A_surface"]

        sum_AB = abs_A_to_B["sum_of_absolute_distances_from_A_vertices_to_closest_point_on_B_surface"]
        sum_BA = abs_B_to_A["sum_of_absolute_distances_from_B_vertices_to_closest_point_on_A_surface"]

        chamfer_mean_of_means = 0.5 * (mean_AB + mean_BA)
        chamfer_weighted_mean = (sum_AB + sum_BA) / float(nA + nB) if (nA + nB) > 0 else 0.0
        chamfer_rms_of_rms = 0.5 * (rms_AB + rms_BA)

        hausdorff_max_symmetric = max(max_AB, max_BA)
        hausdorff_p95_symmetric = max(p95_AB, p95_BA)
        hausdorff_p99_symmetric = max(p99_AB, p99_BA)

        abs_symmetric = {
            "chamfer_absolute_distance_symmetric_mean_of_means_between_A_and_B": float(chamfer_mean_of_means),
            "chamfer_absolute_distance_symmetric_weighted_mean_between_A_and_B": float(chamfer_weighted_mean),
            "chamfer_absolute_distance_symmetric_rms_of_rms_between_A_and_B": float(chamfer_rms_of_rms),
            "hausdorff_absolute_distance_symmetric_maximum_between_A_and_B": float(hausdorff_max_symmetric),
            "hausdorff_absolute_distance_symmetric_percentile_95_between_A_and_B": float(hausdorff_p95_symmetric),
            "hausdorff_absolute_distance_symmetric_percentile_99_between_A_and_B": float(hausdorff_p99_symmetric),
        }

        # ---------- SIGNED ----------
        signed_A_to_B = {}
        signed_B_to_A = {}
        try:
            d_AB_signed = self._signed_distances(A, tmB)
            d_BA_signed = self._signed_distances(B, tmA)
            signed_A_to_B = self._stats_signed_A_to_B(d_AB_signed, nA, who="A", to="B")
            signed_B_to_A = self._stats_signed_A_to_B(d_BA_signed, nB, who="B", to="A")
        except Exception as e:
            slicer.util.warningDisplay(
                f"Distancia SIGNED no disponible para este par "
                f"({nameA or modelA.GetName()} vs {nameB or modelB.GetName()}): {e}"
            )

        pair = {
            "name_of_model_A": nameA or modelA.GetName(),
            "name_of_model_B": nameB or modelB.GetName(),
            "number_of_points_in_A": int(nA),
            "number_of_points_in_B": int(nB),
            "metrics": {
                "absolute_distance_to_surface": {
                    "A_to_B": abs_A_to_B,
                    "B_to_A": abs_B_to_A,
                    "symmetric_AB_and_BA": abs_symmetric,
                },
                "signed_distance_to_surface": {
                    "A_to_B": signed_A_to_B,
                    "B_to_A": signed_B_to_A,
                },
            },
        }

        return pair

    # ---------- Agregación global ----------
    def aggregate_global_metrics(self, pairs_list):
        """
        Aggrega sobre TODOS los pares:
          - Métricas simétricas ABSOLUTE (Chamfer/Hausdorff).
          - Métricas unidireccionales ABSOLUTE A→B y B→A
            (todas las que están en los dicts A_to_B y B_to_A).
        """
        if not pairs_list:
            return {"number_of_pairs": 0, "metrics_aggregate_over_pairs": {}}

        agg = {}

        # ---- 1) Métricas symmétricas (symmetric_AB_and_BA) ----
        symmetric_keys = [
            "chamfer_absolute_distance_symmetric_mean_of_means_between_A_and_B",
            "chamfer_absolute_distance_symmetric_weighted_mean_between_A_and_B",
            "chamfer_absolute_distance_symmetric_rms_of_rms_between_A_and_B",
            "hausdorff_absolute_distance_symmetric_maximum_between_A_and_B",
            "hausdorff_absolute_distance_symmetric_percentile_95_between_A_and_B",
            "hausdorff_absolute_distance_symmetric_percentile_99_between_A_and_B",
        ]

        for k in symmetric_keys:
            vals = []
            for p in pairs_list:
                try:
                    v = p["metrics"]["absolute_distance_to_surface"]["symmetric_AB_and_BA"][k]
                    vals.append(float(v))
                except Exception:
                    pass
            if vals:
                agg[k] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "count": int(len(vals)),
                }

        # ---- 2) Métricas unidireccionales ABSOLUTE A→B y B→A ----
        # Tomamos las claves del primer par como referencia.
        example_A_to_B = pairs_list[0]["metrics"]["absolute_distance_to_surface"]["A_to_B"]
        example_B_to_A = pairs_list[0]["metrics"]["absolute_distance_to_surface"]["B_to_A"]

        # A→B
        for key in example_A_to_B.keys():
            vals = []
            for p in pairs_list:
                try:
                    v = p["metrics"]["absolute_distance_to_surface"]["A_to_B"][key]
                    vals.append(float(v))
                except Exception:
                    pass
            if vals:
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "count": int(len(vals)),
                }

        # B→A
        for key in example_B_to_A.keys():
            vals = []
            for p in pairs_list:
                try:
                    v = p["metrics"]["absolute_distance_to_surface"]["B_to_A"][key]
                    vals.append(float(v))
                except Exception:
                    pass
            if vals:
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "count": int(len(vals)),
                }

        return {"number_of_pairs": len(pairs_list), "metrics_aggregate_over_pairs": agg}

    def format_global_summary_txt(self, globalRes):
        """
        Devuelve un TXT con:
          - nº de pares
          - sección de métricas simétricas
          - sección ABSOLUTE A→B con texto explicativo
          - sección ABSOLUTE B→A con texto explicativo
        """
        if not globalRes or globalRes.get("number_of_pairs", 0) == 0:
            return "Sin pares procesados.\n"

        agg = globalRes["metrics_aggregate_over_pairs"]
        lines = []
        lines.append(f"Número de pares: {globalRes['number_of_pairs']}")
        lines.append("")

        # Clasificamos claves según su tipo
        sym_keys = [k for k in agg.keys() if "symmetric" in k]
        a2b_keys = [k for k in agg.keys() if "from_A_vertices_to_closest_point_on_B_surface" in k]
        b2a_keys = [k for k in agg.keys() if "from_B_vertices_to_closest_point_on_A_surface" in k]
        # otros (por si se meten cosas extra)
        other_keys = [
            k for k in agg.keys()
            if k not in sym_keys and k not in a2b_keys and k not in b2a_keys
        ]

        # ---- Métricas simétricas (Chamfer/Hausdorff) ----
        lines.append("=== ABSOLUTE symmetric metrics between A and B (Chamfer / Hausdorff) ===")
        for key in sym_keys:
            stats = agg[key]
            label = self._nice_label_for_global_metric_key(key)
            lines.append(
                f"- {label}: mean_over_pairs={stats['mean']:.6f} | "
                f"median_over_pairs={stats['median']:.6f} | "
                f"std_over_pairs={stats['std']:.6f} | "
                f"min_value={stats['min']:.6f} | "
                f"max_value={stats['max']:.6f} | "
                f"n_pairs_used={stats['count']}"
            )

        lines.append("")
        # ---- A→B ----
        if a2b_keys:
            lines.append("ABSOLUTE distance A→B (from A vertices to closest point on B surface):")
            for key in a2b_keys:
                stats = agg[key]
                label = key
                lines.append(
                    f"- {label}: mean_over_pairs={stats['mean']:.6f} | "
                    f"median_over_pairs={stats['median']:.6f} | "
                    f"std_over_pairs={stats['std']:.6f} | "
                    f"min_value={stats['min']:.6f} | "
                    f"max_value={stats['max']:.6f} | "
                    f"n_pairs_used={stats['count']}"
                )

        lines.append("")
        # ---- B→A ----
        if b2a_keys:
            lines.append("ABSOLUTE distance B→A (from B vertices to closest point on A surface):")
            for key in b2a_keys:
                stats = agg[key]
                label = key
                lines.append(
                    f"- {label}: mean_over_pairs={stats['mean']:.6f} | "
                    f"median_over_pairs={stats['median']:.6f} | "
                    f"std_over_pairs={stats['std']:.6f} | "
                    f"min_value={stats['min']:.6f} | "
                    f"max_value={stats['max']:.6f} | "
                    f"n_pairs_used={stats['count']}"
                )

        if other_keys:
            lines.append("")
            lines.append("Otros agregados globales:")
            for key in other_keys:
                stats = agg[key]
                label = key
                lines.append(
                    f"- {label}: mean_over_pairs={stats['mean']:.6f} | "
                    f"median_over_pairs={stats['median']:.6f} | "
                    f"std_over_pairs={stats['std']:.6f} | "
                    f"min_value={stats['min']:.6f} | "
                    f"max_value={stats['max']:.6f} | "
                    f"n_pairs_used={stats['count']}"
                )

        lines.append("")
        return "\n".join(lines)

    # ---------- Export CSV (por par) ----------
    def export_pair_metrics_to_csv(self, outDir, base, pairRes):
        """
        Crea un CSV por par con SOLO métricas agregadas (nada de distancias por vértice):
          - {base}_ETSE_UV_registration_metrics.csv
        """
        os.makedirs(outDir, exist_ok=True)
        path_metrics = os.path.join(outDir, f"{base}_ETSE_UV_registration_metrics.csv")

        with open(path_metrics, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pair_name", "metric_group", "metric_name", "value"])

            pair_name = f"{pairRes['name_of_model_A']} vs {pairRes['name_of_model_B']}"

            abs_metrics = pairRes["metrics"]["absolute_distance_to_surface"]
            signed_metrics = pairRes["metrics"]["signed_distance_to_surface"]

            # ABSOLUTE A→B
            for key, val in abs_metrics["A_to_B"].items():
                w.writerow([pair_name, "absolute_distance_A_to_B", key, val])

            # ABSOLUTE B→A
            for key, val in abs_metrics["B_to_A"].items():
                w.writerow([pair_name, "absolute_distance_B_to_A", key, val])

            # ABSOLUTE symmetric
            for key, val in abs_metrics["symmetric_AB_and_BA"].items():
                w.writerow([pair_name, "absolute_distance_symmetric_AB_and_BA", key, val])

            # SIGNED A→B
            for key, val in signed_metrics["A_to_B"].items():
                w.writerow([pair_name, "signed_distance_A_to_B", key, val])

            # SIGNED B→A
            for key, val in signed_metrics["B_to_A"].items():
                w.writerow([pair_name, "signed_distance_B_to_A", key, val])

        return path_metrics

    # ---------- Export CSV (global summary) ----------
    def export_global_summary_to_csv(self, outDir, fileName, globalRes):
        """
        CSV global de resumen:
          - columnas: metric_key, metric_label, mean_over_pairs, median_over_pairs,
                      std_over_pairs, min_value, max_value, n_pairs_used
        """
        if not globalRes or globalRes.get("number_of_pairs", 0) == 0:
            return None

        os.makedirs(outDir, exist_ok=True)
        path = os.path.join(outDir, fileName)

        agg = globalRes.get("metrics_aggregate_over_pairs", {})

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "metric_key",
                "metric_label",
                "mean_over_pairs",
                "median_over_pairs",
                "std_over_pairs",
                "min_value",
                "max_value",
                "n_pairs_used",
            ])
            for key, stats in agg.items():
                label = self._nice_label_for_global_metric_key(key)
                w.writerow([
                    key,
                    label,
                    stats["mean"],
                    stats["median"],
                    stats["std"],
                    stats["min"],
                    stats["max"],
                    stats["count"],
                ])

        return path
