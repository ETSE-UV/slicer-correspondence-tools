# ETSE_UV__MarkupNameTools.py
# 3D Slicer scripted module
#
# Rename / transcribe Markups Fiducial point labels after MALPACA prediction.
# Typical use:
#   MALPACA output labels: "Median Predicted Landmarks-3"
#   PGD/original template: label "3", description "1-3-Canal_Entrance_Down"
#
# Supports:
#   - currently loaded vtkMRMLMarkupsFiducialNode
#   - direct single .mrk.json processing
#   - batch folder processing
#   - clearing labels/descriptions

import os
import re
import json
import slicer
import qt
import ctk
from slicer.ScriptedLoadableModule import *


# ---------------------------------------------------------------------------
# Embedded fallback template copied from P_05_Left.mrk.json
# Each tuple is: (template_label, template_description)
# The first point is index 1.
# ---------------------------------------------------------------------------
DEFAULT_TEMPLATE_RECORDS = [('1', '1-1-Canal_Entrance-Up'),
 ('2', '1-2-Canal_Entrance_Backwards'),
 ('3', '1-3-Canal_Entrance_Down'),
 ('4', '1-4-Canal_Entrance_Forward'),
 ('5', '2-1-1-Concha-Cavum-UP'),
 ('6', '2-1-2-Concha-Cavum-CenterUp'),
 ('7', '2-1-3-Concha-Cavum-Center'),
 ('8', '2-1-4-Concha-Cavum-CenterBottom'),
 ('9', '2-1-5-Concha-Cavum-Bottom'),
 ('10', '2-2-1-Concha-Posterior-Bottom'),
 ('11', '2-2-2-Concha-Posterior-CenterBottom'),
 ('12', '2-2-3-Concha-Posterior-Center'),
 ('13', '2-2-4-Concha-Posterior-CenterUp'),
 ('14', '2-2-5-Concha-Posterior-UP'),
 ('15', '2-3-1-Concha-Cymba-DiagonalBottom'),
 ('16', '2-3-2-Concha-Cymba-Center'),
 ('17', '2-3-3-Concha-Cymba-DiagonalTop'),
 ('18', '2-3-4-Concha-Cymba-NearCruxBottom'),
 ('19', '2-3-5-Concha-Cymba-NearAntiHelixTop'),
 ('20', '3-1-TriangularFossa-Center'),
 ('21', '3-2-TriangularFossa-Top'),
 ('22', '3-3-TriangularFossa-Back'),
 ('23', '3-4-TriangularFossa-Bottom'),
 ('24', '3-5-TriangularFossa-Forward'),
 ('25', '4-1-ScaphoidFossa-Bottom'),
 ('26', '4-2-ScaphoidFossa-MiddleBottom'),
 ('27', '4-3-ScaphoidFossa-Center'),
 ('28', '4-4-ScaphoidFossa-MiddleTop'),
 ('29', '4-5-ScaphoidFossa-Top'),
 ('30', '4-6-ScaphoidFossa-TopForward'),
 ('31', '5-1-Tragus-CenterTip'),
 ('32', '5-2-Tragus-CenterInner'),
 ('33', '5-3-Tragus-CenterOuter'),
 ('34', '5-4-Tragus-CenterBottomTip'),
 ('35', '5-5-Tragus-CenterBottomInner'),
 ('36', '5-6-Tragus-CenterBottomOuter'),
 ('37', '5-7-Tragus-CenterTopTip'),
 ('38', '5-8-Tragus-CenterTopInner'),
 ('39', '5-9-Tragus-CenterTopOuter'),
 ('40', '5-10-Tragus-AnteriorNotchCenter'),
 ('41', '5-11-Tragus-AnteriorNotchInner'),
 ('42', '5-12-Tragus-AnteriorNotchOuter'),
 ('43', '6-1-1-HelixCrux-Outer-TopEdge'),
 ('44', '6-1-2-HelixCrux-Outer-TopInner'),
 ('45', '6-1-3-HelixCrux-Outer-TopOuter'),
 ('46', '6-1-4-HelixCrux-Outer-MiddleTopEdge'),
 ('47', '6-1-5-HelixCrux-Outer-MiddleTopInner'),
 ('48', '6-1-6-HelixCrux-Outer-MiddleTopOuter'),
 ('49', '6-1-7-HelixCrux-Outer-MiddleEdge'),
 ('50', '6-1-8-HelixCrux-Outer-MiddleInner'),
 ('51', '6-1-9-HelixCrux-Outer-MiddleOuter'),
 ('52', '6-1-10-HelixCrux-Outer-BottomMiddleEdge'),
 ('53', '6-1-11-HelixCrux-Outer-BottomMiddleInner'),
 ('54', '6-1-12-HelixCrux-Outer-BottomMiddleOuter'),
 ('55', '6-1-13-HelixCrux-Outer-BottomEdge(zone anterior notch)'),
 ('56', '6-1-14-HelixCrux-Outer-BottomInner(zone anterior notch)'),
 ('57', '6-1-15-HelixCrux-Outer-BottomOuter(zone anterior notch)'),
 ('58', '6-2-1-HelixCrux-Inner-ForwardEdge'),
 ('59', '6-2-2-HelixCrux-Inner-ForwardTop'),
 ('60', '6-2-3-HelixCrux-Inner-ForwardBottom'),
 ('61', '6-2-4-HelixCrux-Inner-MiddleForwardEdge'),
 ('62', '6-2-5-HelixCrux-Inner-MiddleForwardTop'),
 ('63', '6-2-6-HelixCrux-Inner-MiddleForwardBottom'),
 ('64', '6-2-7-HelixCrux-Inner-MiddleEdge'),
 ('65', '6-2-8-HelixCrux-Inner-MiddleTop'),
 ('66', '6-2-9-HelixCrux-Inner-MiddleBottom'),
 ('67', '6-2-10-HelixCrux-Inner-MiddleBackwardEdge'),
 ('68', '6-2-11-HelixCrux-Inner-MiddleBackwardTop'),
 ('69', '6-2-12-HelixCrux-Inner-MiddleBackwardBottom'),
 ('70', '6-2-13-HelixCrux-Inner-BackwardEdge'),
 ('71', '6-2-14-HelixCrux-Inner-BackwardTop'),
 ('72', '6-2-15-HelixCrux-Inner-BackwardBottom'),
 ('73', '7-1-1-Antihelix-SuperiorCrux-TopEdge'),
 ('74', '7-1-2-Antihelix-SuperiorCrux-TopBackward'),
 ('75', '7-1-3-Antihelix-SuperiorCrux-TopForward'),
 ('76', '7-1-4-Antihelix-SuperiorCrux-MiddleEdge'),
 ('77', '7-1-5-Antihelix-SuperiorCrux-MiddleBackward'),
 ('78', '7-1-6-Antihelix-SuperiorCrux-MiddleForward'),
 ('79', '7-1-7-Antihelix-SuperiorCrux-BottomEdge'),
 ('80', '7-1-8-Antihelix-SuperiorCrux-BottomBackward'),
 ('81', '7-1-6-Antihelix-SuperiorCrux-BottomForward'),
 ('82', '7-2-1-Antihelix-InferiorCrux-ForwardEdge'),
 ('83', '7-2-2-Antihelix-InferiorCrux-ForwardUp'),
 ('84', '7-2-3-Antihelix-InferiorCrux-ForwardDown'),
 ('85', '7-2-4-Antihelix-InferiorCrux-MiddleEdge'),
 ('86', '7-2-5-Antihelix-InferiorCrux-MiddleUp'),
 ('87', '7-2-6-Antihelix-InferiorCrux-MiddleDown'),
 ('88', '7-2-7-Antihelix-InferiorCrux-BackwardEdge'),
 ('89', '7-2-8-Antihelix-InferiorCrux-BackwardUp'),
 ('90', '7-2-9-Antihelix-InferiorCrux-BackwardDown'),
 ('91', '7-3-1-Antihelix-Antihelix-TopForward (cut limit with triangular fossa)'),
 ('92', '7-3-2-Antihelix-Antihelix-Top_MiddleForward (cut limit with triangular fossa)'),
 ('93', '7-3-3-Antihelix-Antihelix-Top_Middle'),
 ('94', '7-3-4-Antihelix-Antihelix-Top_MiddleBackward'),
 ('95', '7-3-5-Antihelix-Antihelix-Top_Backward'),
 ('96', '7-3-6-Antihelix-Antihelix-SubTopForward (cut limit with Cymba Concha)'),
 ('97', '7-3-7-Antihelix-Antihelix-SubTop_MiddleForward (cut limit with Cymba Concha)'),
 ('98', '7-3-8-Antihelix-Antihelix-SubTop_Middle'),
 ('99', '7-3-9-Antihelix-Antihelix-SubTop_MiddleBackward'),
 ('100', '7-3-10-Antihelix-Antihelix-SubTop_Backward'),
 ('101', '7-3-11-Antihelix-Antihelix-MiddleTopForward'),
 ('102', '7-3-12-Antihelix-Antihelix-MiddleTop_MiddleForward'),
 ('103', '7-3-13-Antihelix-Antihelix-MiddleTop_Middle'),
 ('104', '7-3-14-Antihelix-Antihelix-MiddleTop_MiddleBackward'),
 ('105', '7-3-15-Antihelix-Antihelix-MiddleTop_Backward'),
 ('106', '7-3-16-Antihelix-Antihelix-Middle-Forward (cut limit with posterior Concha)'),
 ('107', '7-3-17-Antihelix-Antihelix-Middle-MiddleForward (cut limit with posterior Concha)'),
 ('108', '7-3-18-Antihelix-Antihelix-Middle-Middle'),
 ('109', '7-3-19-Antihelix-Antihelix-Middle-MiddleBackward'),
 ('110', '7-3-20-Antihelix-Antihelix-Middle-Backward'),
 ('111', '7-3-21-Antihelix-Antihelix-MiddleBottom-Forward (cut limit with posterior Concha)'),
 ('112', '7-3-22-Antihelix-Antihelix-MiddleBottom-MiddleForward (cut limit with posterior Concha)'),
 ('113', '7-3-23-Antihelix-Antihelix-MiddleBottom-Middle'),
 ('114', '7-3-24-Antihelix-Antihelix-MiddleBottom-MiddleBackward'),
 ('115', '7-3-25-Antihelix-Antihelix-MiddleBottom-Backward'),
 ('116', '7-3-26-Antihelix-Antihelix-SubBottom-Forward (cut limit with cavum Concha)'),
 ('117', '7-3-27-Antihelix-Antihelix-SubBottom-MiddleForward (cut limit with cavum Concha)'),
 ('118', '7-3-28-Antihelix-Antihelix-SubBottom-Middle'),
 ('119', '7-3-29-Antihelix-Antihelix-SubBottom-MiddleBackward'),
 ('120', '7-3-30-Antihelix-Antihelix-SubBottom-Backward'),
 ('121', '7-3-31-Antihelix-Antihelix-Bottom-Forward (cut limit with cavum Concha)'),
 ('122', '7-3-32-Antihelix-Antihelix-Bottom-MiddleForward (cut limit with cavum Concha)'),
 ('123', '7-3-33-Antihelix-Antihelix-Bottom-Middle'),
 ('124', '7-3-34-Antihelix-Antihelix-Bottom-MiddleBackward'),
 ('125', '7-3-35-Antihelix-Antihelix-Bottom-Backward'),
 ('126', '8-1-Antigragus-Backward_Inner'),
 ('127', '8-2-Antigragus-Backward_Edge'),
 ('128', '8-3-Antigragus-Backward_OuterTop'),
 ('129', '8-4-Antigragus-Backward_OuterBottom'),
 ('130', '8-5-Antigragus-Middle_Inner'),
 ('131', '8-6-Antigragus-Middle_Edge'),
 ('132', '8-7-Antigragus-Middle_OuterTop'),
 ('133', '8-8-Antigragus-Middle_OuterBottom'),
 ('134', '8-9-Antigragus-Forward_Inner'),
 ('135', '8-10-Antigragus-Forward_Edge'),
 ('136', '8-11-Antigragus-Forward_OuterTop'),
 ('137', '8-12-Antigragus-Forward_OuterBottom'),
 ('138', '9-1-IntertragicNotch-Outer'),
 ('139', '9-2-IntertragicNotch-MiddleOuter'),
 ('140', '9-3-IntertragicNotch-Middle'),
 ('141', '9-4-IntertragicNotch-MiddleInner'),
 ('142', '9-5-IntertragicNotch-Inner(Near Canal)'),
 ('143', '10-1-1-Helix-LateralOuter_1 (Start otobasion anterior_superior)'),
 ('144', '10-1-2-Helix-LateralOuter_2'),
 ('145', '10-1-3-Helix-LateralOuter_3'),
 ('146', '10-1-4-Helix-LateralOuter_4 (Auricular tubercle)'),
 ('147', '10-1-5-Helix-LateralOuter_5'),
 ('148', '10-1-6-Helix-LateralOuter_6 (Near Otobasion Posterior)'),
 ('149', '10-1-7-Helix-LateralOuter_7'),
 ('150', '10-1-8-Helix-LateralOuter_8'),
 ('151', '10-1-9-Helix-LateralOuter_9'),
 ('152', '10-1-10-Helix-LateralOuter_10'),
 ('153', '10-1-11-Helix-LateralOuter_11'),
 ('154', '10-1-12-Helix-LateralOuter_12 (Start Lobe)'),
 ('155', '10-2-1-Helix-Outer_1'),
 ('156', '10-2-2-Helix-Outer_2'),
 ('157', '10-2-3-Helix-Outer_3'),
 ('158', '10-2-4-Helix-Outer_4'),
 ('159', '10-2-5-Helix-Outer_5'),
 ('160', '10-2-6-Helix-Outer_6'),
 ('161', '10-2-7-Helix-Outer_7'),
 ('162', '10-2-8-Helix-Outer_8'),
 ('163', '10-2-9-Helix-Outer_9'),
 ('164', '10-2-10-Helix-Outer_10'),
 ('165', '10-2-11-Helix-Outer_11'),
 ('166', '10-2-12-Helix-Outer_12'),
 ('167', '10-3-1-Helix-LateralInner_1'),
 ('168', '10-3-2-Helix-LateralInner_2'),
 ('169', '10-3-3-Helix-LateralInner_3'),
 ('170', '10-3-4-Helix-LateralInner_4'),
 ('171', '10-3-5-Helix-LateralInner_5'),
 ('172', '10-3-6-Helix-LateralInner_6'),
 ('173', '10-3-7-Helix-LateralInner_7'),
 ('174', '10-3-8-Helix-LateralInner_8'),
 ('175', '10-3-9-Helix-LateralInner_9'),
 ('176', '10-3-10-Helix-LateralInner_10'),
 ('177', '10-3-11-Helix-LateralInner_11'),
 ('178', '10-3-12-Helix-LateralInner_12'),
 ('179', '10-4-1-Helix-Edge_1'),
 ('180', '10-4-2-Helix-Edge_2'),
 ('181', '10-4-3-Helix-Edge_3'),
 ('182', '10-4-4-Helix-Edge_4'),
 ('183', '10-4-5-Helix-Edge_5'),
 ('184', '10-4-6-Helix-Edge_6'),
 ('185', '10-4-7-Helix-Edge_7'),
 ('186', '10-4-8-Helix-Edge_8'),
 ('187', '10-4-9-Helix-Edge_9'),
 ('188', '10-4-10-Helix-Edge_10'),
 ('189', '10-4-11-Helix-Edge_11'),
 ('190', '10-4-12-Helix-Edge_12'),
 ('191', '10-5-1-Helix-InnerFold_1'),
 ('192', '10-5-2-Helix-InnerFold_2'),
 ('193', '10-5-3-Helix-InnerFold_3'),
 ('194', '10-5-4-Helix-InnerFold_4'),
 ('195', '10-5-5-Helix-InnerFold_5'),
 ('196', '10-5-6-Helix-InnerFold_6'),
 ('197', '10-5-7-Helix-InnerFold_7'),
 ('198', '10-5-8-Helix-InnerFold_8'),
 ('199', '10-5-9-Helix-InnerFold_9'),
 ('200', '11-1-Lobe_Backward_LateralOuterTop'),
 ('201', '11-2-Lobe_Backward_LateralOuterMiddle'),
 ('202', '11-3-Lobe_Backward_LateralOuterEdge'),
 ('203', '11-4-Lobe_Backward_OuterEdge'),
 ('204', '11-5-Lobe_Backward_LateralInnerEdge'),
 ('205', '11-6-Lobe_Middle_LateralOuterTop'),
 ('206', '11-7-Lobe_Middle_LateralOuterMiddle'),
 ('207', '11-8-Lobe_Middle_LateralOuterEdge'),
 ('208', '11-9-Lobe_Middle_OuterEdge'),
 ('209', '11-10-Lobe_Middle_LateralInnerEdge'),
 ('210', '11-11-Lobe_Forward_LateralOuterTop'),
 ('211', '11-12-Lobe_Forward_LateralOuterMiddle'),
 ('212', '11-13-Lobe_Forward_LateralOuterEdge'),
 ('213', '11-14-Lobe_Forward_OuterEdge'),
 ('214', '11-15-Lobe_Forward_LateralInnerEdge'),
 ('215', '12-1-1-BackEar_AntihelixSide_1(Top, Near Eminence Scapha)'),
 ('216', '12-1-2-BackEar_AntihelixSide_2(Top, Near Eminence Scapha)'),
 ('217', '12-1-3-BackEar_AntihelixSide_3(Middle, Near Antihelical Fossa)'),
 ('218', '12-1-4-BackEar_AntihelixSide_4(Middle, Near Antihelical Fossa)'),
 ('219', '12-1-5-BackEar_AntihelixSide_5(Middle, Near Antihelical Fossa)'),
 ('220', '12-1-6-BackEar_AntihelixSide_6(Bottom, Near Antihelical Fossa)'),
 ('221', '12-1-7-BackEar_AntihelixSide_7(Bottom, Near Antihelical Fossa)'),
 ('222', '12-2-1-BackEar-CenterAntihelicalFold_1(Top)'),
 ('223', '12-2-2-BackEar-CenterAntihelicalFold_2(Top)'),
 ('224', '12-2-3-BackEar-CenterAntihelicalFold_3(Top)'),
 ('225', '12-2-4-BackEar-CenterAntihelicalFold_4(Middle)'),
 ('226', '12-2-5-BackEar-CenterAntihelicalFold_5(Middle)'),
 ('227', '12-2-6-BackEar-CenterAntihelicalFold_6(Middle)'),
 ('228', '12-3-1-BackEar-LobeFold_1(Bottom)'),
 ('229', '12-3-2-BackEar-LobeFold_2(Bottom)'),
 ('230', '12-3-3-BackEar-LobeFold_3(Bottom)'),
 ('231', '12-4-1-BackEar-ConchalBowlCut_1(Top)'),
 ('232', '12-4-2-BackEar-ConchalBowlCut_2(Middle)'),
 ('233', '12-4-3-BackEar-ConchalBowlCut_3(Middle)'),
 ('234', '12-4-4-BackEar-ConchalBowlCut_4(Middle)'),
 ('235', '12-4-5-BackEar-ConchalBowlCut_5(Bottom)'),
 ('236', '12-4-6-BackEar-ConchalBowlCut_6(Bottom)'),
 ('237', '12-4-7-BackEar-ConchalBowlCut_7(Bottom)'),
 ('238', '12-5-1-BackEar-ConchalBowlCenter_1(Top)'),
 ('239', '12-5-2-BackEar-ConchalBowlCenter_2(Middle)'),
 ('240', '12-5-3-BackEar-ConchalBowlCenter_3(Middle)'),
 ('241', '12-5-4-BackEar-ConchalBowlCenter_4(Middle)'),
 ('242', '12-5-5-BackEar-ConchalBowlCenter_5(Bottom)'),
 ('243', '12-6-1-BackEar-ExtraPointsBackScapha_1(Top)'),
 ('244', '12-6-2-BackEar-ExtraPointsBackScapha_2(MiddleTop)'),
 ('245', '12-6-3-BackEar-ExtraPointsBackScapha_3(Middle)'),
 ('246', '13-1-Otobasion_1 (Otobasion Superius - 12h)'),
 ('247', '13-2-Otobasion_2 (Otobasion Back - 12.45h)'),
 ('248', '13-3-Otobasion_3 (Otobasion Back - 1.30h)'),
 ('249', '13-4-Otobasion_4 (Otobasion Back - 2.15h)'),
 ('250', '13-5-Otobasion_5 (Otobasion Posterius - 3h)'),
 ('251', '13-6-Otobasion_6 (Otobasion Back - 3.45h)'),
 ('252', '13-7-Otobasion_7 (Otobasion Back - 4.30h)'),
 ('253', '13-8-Otobasion_8 (Otobasion Back - 5.15h)'),
 ('254', '13-9-Otobasion_9 (Otobasion Inferius - 6h)'),
 ('255', '13-10-Otobasion_10 (Forward Limit Ear near Lobe - 6.45h)'),
 ('256', '13-11-Otobasion_11 (Forward Limit Ear Intertragic Notch - 7.30h)'),
 ('257', '13-12-Otobasion_12 (Forward Limit Ear Near Tragus - 8.15h)'),
 ('258', '13-13-Otobasion_13 (Forward Limit Ear Near Tragus - 9h)'),
 ('259', '13-14-Otobasion_14 (Forward Limit Ear Between Tragus and HCrux - 9:45h)'),
 ('260', '13-15-Otobasion_15 (Forward Limit Ear start HCrux - 10:30h)'),
 ('261', '13-16-Otobasion_15 (Otobasion Anterius - 11.45h)')]


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------
class ETSE_UV__MarkupNameTools(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "ETSE-UV Markup Name Tools"
        parent.categories = ["ETSE_UV"]
        parent.dependencies = []
        parent.contributors = ["ETSE-UV"]
        parent.helpText = """
        <p><b>Markup Name Tools</b> renames MALPACA/Slicer fiducial point names using a PGD/original fiducial template.</p>

        <p><b>Main tools:</b></p>
        <ul>
          <li>Transcribe labels/descriptions on a loaded Markups Fiducial node.</li>
          <li>Transcribe one .mrk.json file directly.</li>
          <li>Batch-process folders of .mrk.json files.</li>
          <li>Clear/delete all point labels and descriptions.</li>
        </ul>

        <p>If no template node/file is selected, the embedded PGD template 261 fidutials names are used.</p>
        """
        parent.acknowledgementText = (
            "Developed at ETSE-UV for PGD/MALPACA fiducial post-processing. "
            "Thanks to the 3D Slicer community."
        )


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class ETSE_UV__MarkupNameToolsWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = ETSE_UV__MarkupNameToolsLogic()

        # ------------------------------------------------------------
        # Common mode help
        # ------------------------------------------------------------
        helpBox = ctk.ctkCollapsibleButton()
        helpBox.text = "What this module does"
        helpBox.collapsed = True
        self.layout.addWidget(helpBox)
        helpLayout = qt.QVBoxLayout(helpBox)
        helpLabel = qt.QLabel(
            "Use this after MALPACA creates labels like 'Median Predicted Landmarks-3'.\n"
            "The module copies the original PGD names by point index. By default it uses the "
            "embedded template: label='3', description='1-3-Canal_Entrance_Down'.\n\n"
            "Recommended mode if you want readable point names in Slicer: 'Anatomical label'.\n"
            "Recommended mode if your later scripts expect the old format: 'Original PGD style'."
        )
        helpLabel.wordWrap = True
        helpLayout.addWidget(helpLabel)

        # ------------------------------------------------------------
        # 1) Current scene node
        # ------------------------------------------------------------
        nodeBox = ctk.ctkCollapsibleButton()
        nodeBox.text = "1) Loaded node: transcribe / clear point names"
        nodeBox.collapsed = False
        self.layout.addWidget(nodeBox)
        nodeForm = qt.QFormLayout(nodeBox)

        self.targetNodeSelector = slicer.qMRMLNodeComboBox()
        self.targetNodeSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.targetNodeSelector.selectNodeUponCreation = True
        self.targetNodeSelector.addEnabled = False
        self.targetNodeSelector.removeEnabled = False
        self.targetNodeSelector.noneEnabled = True
        self.targetNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.targetNodeSelector.setToolTip("MALPACA/output fiducials to rename.")
        nodeForm.addRow("Target markups:", self.targetNodeSelector)

        self.templateNodeSelector = slicer.qMRMLNodeComboBox()
        self.templateNodeSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.templateNodeSelector.selectNodeUponCreation = True
        self.templateNodeSelector.addEnabled = False
        self.templateNodeSelector.removeEnabled = False
        self.templateNodeSelector.noneEnabled = True
        self.templateNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.templateNodeSelector.setToolTip(
            "Optional. If empty, the embedded template names are used."
        )
        nodeForm.addRow("Template markups:", self.templateNodeSelector)

        self.nodeModeCombo = self._makeModeCombo()
        nodeForm.addRow("Rename mode:", self.nodeModeCombo)

        self.nodeRangeEdit = qt.QLineEdit("all")
        self.nodeRangeEdit.setToolTip("1-based target point range. Examples: all, 1-261, 1-4,246-261")
        nodeForm.addRow("Target point range:", self.nodeRangeEdit)

        self.nodeParsedIndexCheck = qt.QCheckBox("Use number parsed from current label/id when possible")
        self.nodeParsedIndexCheck.checked = True
        self.nodeParsedIndexCheck.setToolTip(
            "Useful for labels like 'Median Predicted Landmarks-3'. If parsing fails, point order is used."
        )
        nodeForm.addRow(self.nodeParsedIndexCheck)

        nodeButtons = qt.QHBoxLayout()
        self.nodeTranscribeButton = qt.QPushButton("Transcribe selected node")
        self.nodeClearButton = qt.QPushButton("Clear selected node names")
        self.nodeSaveButton = qt.QPushButton("Save selected node as .mrk.json…")
        nodeButtons.addWidget(self.nodeTranscribeButton)
        nodeButtons.addWidget(self.nodeClearButton)
        nodeButtons.addWidget(self.nodeSaveButton)
        nodeButtonsWidget = qt.QWidget()
        nodeButtonsWidget.setLayout(nodeButtons)
        nodeForm.addRow(nodeButtonsWidget)

        self.nodeStatus = qt.QLabel("Ready.")
        self.nodeStatus.wordWrap = True
        nodeForm.addRow("Status:", self.nodeStatus)

        self.nodeTranscribeButton.clicked.connect(self.onTranscribeNode)
        self.nodeClearButton.clicked.connect(self.onClearNode)
        self.nodeSaveButton.clicked.connect(self.onSaveSelectedNode)

        # ------------------------------------------------------------
        # 2) Single file
        # ------------------------------------------------------------
        fileBox = ctk.ctkCollapsibleButton()
        fileBox.text = "2) Single .mrk.json file: transcribe / clear"
        fileBox.collapsed = False
        self.layout.addWidget(fileBox)
        fileForm = qt.QFormLayout(fileBox)

        self.fileInputEdit = qt.QLineEdit("")
        self.fileInputEdit.setToolTip("Input .mrk.json file.")
        self.fileInputButton = qt.QPushButton("Browse…")
        fileForm.addRow("Input file:", self._lineEditWithButton(self.fileInputEdit, self.fileInputButton))

        self.fileTemplateEdit = qt.QLineEdit("")
        self.fileTemplateEdit.setToolTip("Optional template .mrk.json. Leave empty to use embedded names.")
        self.fileTemplateButton = qt.QPushButton("Browse…")
        fileForm.addRow("Template file:", self._lineEditWithButton(self.fileTemplateEdit, self.fileTemplateButton))

        self.fileOutputEdit = qt.QLineEdit("")
        self.fileOutputEdit.setToolTip("Output .mrk.json file.")
        self.fileOutputButton = qt.QPushButton("Browse…")
        fileForm.addRow("Output file:", self._lineEditWithButton(self.fileOutputEdit, self.fileOutputButton))

        self.fileModeCombo = self._makeModeCombo()
        fileForm.addRow("Rename mode:", self.fileModeCombo)

        self.fileRangeEdit = qt.QLineEdit("all")
        self.fileRangeEdit.setToolTip("1-based target point range. Examples: all, 1-261, 1-4,246-261")
        fileForm.addRow("Target point range:", self.fileRangeEdit)

        self.fileParsedIndexCheck = qt.QCheckBox("Use number parsed from current label/id when possible")
        self.fileParsedIndexCheck.checked = True
        fileForm.addRow(self.fileParsedIndexCheck)

        fileButtons = qt.QHBoxLayout()
        self.fileTranscribeButton = qt.QPushButton("Transcribe file")
        self.fileClearButton = qt.QPushButton("Clear file names")
        fileButtons.addWidget(self.fileTranscribeButton)
        fileButtons.addWidget(self.fileClearButton)
        fileButtonsWidget = qt.QWidget()
        fileButtonsWidget.setLayout(fileButtons)
        fileForm.addRow(fileButtonsWidget)

        self.fileStatus = qt.QLabel("Ready.")
        self.fileStatus.wordWrap = True
        fileForm.addRow("Status:", self.fileStatus)

        self.fileInputButton.clicked.connect(lambda: self._browseOpenFile(self.fileInputEdit, "Select input markups"))
        self.fileTemplateButton.clicked.connect(lambda: self._browseOpenFile(self.fileTemplateEdit, "Select template markups"))
        self.fileOutputButton.clicked.connect(lambda: self._browseSaveFile(self.fileOutputEdit, "Save output markups"))
        self.fileTranscribeButton.clicked.connect(self.onTranscribeFile)
        self.fileClearButton.clicked.connect(self.onClearFile)

        # ------------------------------------------------------------
        # 3) Batch folder
        # ------------------------------------------------------------
        batchBox = ctk.ctkCollapsibleButton()
        batchBox.text = "3) Batch folder: transcribe / clear .mrk.json files"
        batchBox.collapsed = False
        self.layout.addWidget(batchBox)
        batchForm = qt.QFormLayout(batchBox)

        self.batchInputDirEdit = qt.QLineEdit("")
        self.batchInputDirButton = qt.QPushButton("Browse…")
        batchForm.addRow("Input folder:", self._lineEditWithButton(self.batchInputDirEdit, self.batchInputDirButton))

        self.batchOutputDirEdit = qt.QLineEdit("")
        self.batchOutputDirButton = qt.QPushButton("Browse…")
        batchForm.addRow("Output folder:", self._lineEditWithButton(self.batchOutputDirEdit, self.batchOutputDirButton))

        self.batchTemplateEdit = qt.QLineEdit("")
        self.batchTemplateButton = qt.QPushButton("Browse…")
        batchForm.addRow("Template file:", self._lineEditWithButton(self.batchTemplateEdit, self.batchTemplateButton))

        self.batchModeCombo = self._makeModeCombo()
        batchForm.addRow("Rename mode:", self.batchModeCombo)

        self.batchRangeEdit = qt.QLineEdit("all")
        batchForm.addRow("Target point range:", self.batchRangeEdit)

        self.batchParsedIndexCheck = qt.QCheckBox("Use number parsed from current label/id when possible")
        self.batchParsedIndexCheck.checked = True
        batchForm.addRow(self.batchParsedIndexCheck)

        self.batchRecursiveCheck = qt.QCheckBox("Recursive")
        self.batchRecursiveCheck.checked = False
        batchForm.addRow(self.batchRecursiveCheck)

        self.batchKeepRelativeCheck = qt.QCheckBox("Keep relative subfolders in output")
        self.batchKeepRelativeCheck.checked = True
        self.batchKeepRelativeCheck.setToolTip("Only relevant when Recursive is enabled.")
        batchForm.addRow(self.batchKeepRelativeCheck)

        self.batchSuffixEdit = qt.QLineEdit("_renamed")
        self.batchSuffixEdit.setToolTip(
            "Suffix inserted before .mrk.json. Leave empty to preserve filenames in output folder. "
            "Use a suffix if input and output are the same folder."
        )
        batchForm.addRow("Output filename suffix:", self.batchSuffixEdit)

        batchButtons = qt.QHBoxLayout()
        self.batchTranscribeButton = qt.QPushButton("Batch transcribe")
        self.batchClearButton = qt.QPushButton("Batch clear names")
        batchButtons.addWidget(self.batchTranscribeButton)
        batchButtons.addWidget(self.batchClearButton)
        batchButtonsWidget = qt.QWidget()
        batchButtonsWidget.setLayout(batchButtons)
        batchForm.addRow(batchButtonsWidget)

        self.batchStatus = qt.QLabel("Ready.")
        self.batchStatus.wordWrap = True
        batchForm.addRow("Status:", self.batchStatus)

        self.batchInputDirButton.clicked.connect(lambda: self._browseDir(self.batchInputDirEdit, "Select input folder"))
        self.batchOutputDirButton.clicked.connect(lambda: self._browseDir(self.batchOutputDirEdit, "Select output folder"))
        self.batchTemplateButton.clicked.connect(lambda: self._browseOpenFile(self.batchTemplateEdit, "Select template markups"))
        self.batchTranscribeButton.clicked.connect(self.onBatchTranscribe)
        self.batchClearButton.clicked.connect(self.onBatchClear)

        self.layout.addStretch(1)

    # ------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------
    def _makeModeCombo(self):
        c = qt.QComboBox()
        c.addItem("Anatomical label: label = template description, description = template description", "anatomical_label")
        c.addItem("Original PGD style: label = template label, description = template description", "original_style")
        c.addItem("Descriptions only: keep current labels, description = template description", "description_only")
        c.addItem("Numeric labels only: label = template label, clear description", "numeric_only")
        return c

    def _comboData(self, combo):
        try:
            return str(combo.itemData(combo.currentIndex))
        except Exception:
            txt = str(combo.currentText)
            if txt.startswith("Original"):
                return "original_style"
            if txt.startswith("Descriptions"):
                return "description_only"
            if txt.startswith("Numeric"):
                return "numeric_only"
            return "anatomical_label"

    def _lineEditWithButton(self, lineEdit, button):
        row = qt.QHBoxLayout()
        row.addWidget(lineEdit)
        row.addWidget(button)
        w = qt.QWidget()
        w.setLayout(row)
        return w

    def _browseOpenFile(self, edit, title):
        path = qt.QFileDialog.getOpenFileName(self.parent, title, "", "Slicer Markups (*.mrk.json);;JSON files (*.json);;All files (*)")
        if path:
            edit.setText(path)

    def _browseSaveFile(self, edit, title):
        path = qt.QFileDialog.getSaveFileName(self.parent, title, "", "Slicer Markups (*.mrk.json);;JSON files (*.json);;All files (*)")
        if path:
            path = str(path)
            if not path.lower().endswith(".mrk.json"):
                path += ".mrk.json"
            edit.setText(path)

    def _browseDir(self, edit, title):
        path = qt.QFileDialog.getExistingDirectory(self.parent, title)
        if path:
            edit.setText(path)

    def _templateRecordsFromNodeOrDefault(self):
        templateNode = self.templateNodeSelector.currentNode()
        if templateNode:
            return self.logic.template_records_from_node(templateNode)
        return self.logic.default_template_records()

    # ------------------------------------------------------------
    # Node callbacks
    # ------------------------------------------------------------
    def onTranscribeNode(self):
        target = self.targetNodeSelector.currentNode()
        if not target:
            slicer.util.errorDisplay("Select a target markups node.")
            return
        try:
            modified = self.logic.transcribe_node(
                targetNode=target,
                templateRecords=self._templateRecordsFromNodeOrDefault(),
                mode=self._comboData(self.nodeModeCombo),
                rangeText=self.nodeRangeEdit.text,
                useParsedIndex=bool(self.nodeParsedIndexCheck.checked),
            )
            msg = f"Updated node '{target.GetName()}': {modified} control points."
            self.nodeStatus.setText(msg)
            slicer.util.infoDisplay(msg)
        except Exception as e:
            self.nodeStatus.setText(f"Error: {e}")
            slicer.util.errorDisplay(str(e))

    def onClearNode(self):
        target = self.targetNodeSelector.currentNode()
        if not target:
            slicer.util.errorDisplay("Select a target markups node.")
            return
        try:
            modified = self.logic.clear_node_names(target, rangeText=self.nodeRangeEdit.text)
            msg = f"Cleared node '{target.GetName()}': {modified} control points."
            self.nodeStatus.setText(msg)
            slicer.util.infoDisplay(msg)
        except Exception as e:
            self.nodeStatus.setText(f"Error: {e}")
            slicer.util.errorDisplay(str(e))

    def onSaveSelectedNode(self):
        target = self.targetNodeSelector.currentNode()
        if not target:
            slicer.util.errorDisplay("Select a target markups node.")
            return
        path = qt.QFileDialog.getSaveFileName(self.parent, "Save selected markups", "", "Slicer Markups (*.mrk.json)")
        if not path:
            return
        path = str(path)
        if not path.lower().endswith(".mrk.json"):
            path += ".mrk.json"
        try:
            ok = slicer.util.saveNode(target, path)
            if not ok:
                raise RuntimeError("slicer.util.saveNode returned False.")
            msg = f"Saved: {path}"
            self.nodeStatus.setText(msg)
            slicer.util.infoDisplay(msg)
        except Exception as e:
            self.nodeStatus.setText(f"Error: {e}")
            slicer.util.errorDisplay(str(e))

    # ------------------------------------------------------------
    # File callbacks
    # ------------------------------------------------------------
    def onTranscribeFile(self):
        inPath = self.fileInputEdit.text.strip()
        outPath = self.fileOutputEdit.text.strip()
        if not inPath or not os.path.isfile(inPath):
            slicer.util.errorDisplay("Select a valid input .mrk.json file.")
            return
        if not outPath:
            slicer.util.errorDisplay("Select an output .mrk.json file.")
            return
        try:
            templateRecords = self.logic.template_records_from_file_or_default(self.fileTemplateEdit.text.strip())
            modified = self.logic.transcribe_file(
                inputPath=inPath,
                outputPath=outPath,
                templateRecords=templateRecords,
                mode=self._comboData(self.fileModeCombo),
                rangeText=self.fileRangeEdit.text,
                useParsedIndex=bool(self.fileParsedIndexCheck.checked),
            )
            msg = f"Wrote: {outPath} ({modified} control points updated)."
            self.fileStatus.setText(msg)
            slicer.util.infoDisplay(msg)
        except Exception as e:
            self.fileStatus.setText(f"Error: {e}")
            slicer.util.errorDisplay(str(e))

    def onClearFile(self):
        inPath = self.fileInputEdit.text.strip()
        outPath = self.fileOutputEdit.text.strip()
        if not inPath or not os.path.isfile(inPath):
            slicer.util.errorDisplay("Select a valid input .mrk.json file.")
            return
        if not outPath:
            slicer.util.errorDisplay("Select an output .mrk.json file.")
            return
        try:
            modified = self.logic.clear_file_names(inPath, outPath, rangeText=self.fileRangeEdit.text)
            msg = f"Wrote: {outPath} ({modified} control points cleared)."
            self.fileStatus.setText(msg)
            slicer.util.infoDisplay(msg)
        except Exception as e:
            self.fileStatus.setText(f"Error: {e}")
            slicer.util.errorDisplay(str(e))

    # ------------------------------------------------------------
    # Batch callbacks
    # ------------------------------------------------------------
    def onBatchTranscribe(self):
        inDir = self.batchInputDirEdit.text.strip()
        outDir = self.batchOutputDirEdit.text.strip()
        if not inDir or not os.path.isdir(inDir):
            slicer.util.errorDisplay("Select a valid input folder.")
            return
        if not outDir:
            slicer.util.errorDisplay("Select an output folder.")
            return
        try:
            templateRecords = self.logic.template_records_from_file_or_default(self.batchTemplateEdit.text.strip())
            result = self.logic.batch_transcribe_folder(
                inputDir=inDir,
                outputDir=outDir,
                templateRecords=templateRecords,
                mode=self._comboData(self.batchModeCombo),
                rangeText=self.batchRangeEdit.text,
                useParsedIndex=bool(self.batchParsedIndexCheck.checked),
                recursive=bool(self.batchRecursiveCheck.checked),
                keepRelativeFolders=bool(self.batchKeepRelativeCheck.checked),
                suffix=self.batchSuffixEdit.text,
            )
            msg = f"Batch transcribe done. Processed: {result['processed']}, failed: {result['failed']}. Output: {outDir}"
            if result.get("errors"):
                msg += "\nFirst errors:\n" + "\n".join(result["errors"][:5])
            self.batchStatus.setText(msg)
            slicer.util.infoDisplay(msg)
        except Exception as e:
            self.batchStatus.setText(f"Error: {e}")
            slicer.util.errorDisplay(str(e))

    def onBatchClear(self):
        inDir = self.batchInputDirEdit.text.strip()
        outDir = self.batchOutputDirEdit.text.strip()
        if not inDir or not os.path.isdir(inDir):
            slicer.util.errorDisplay("Select a valid input folder.")
            return
        if not outDir:
            slicer.util.errorDisplay("Select an output folder.")
            return
        try:
            suffix = self.batchSuffixEdit.text
            if str(suffix or "") == "_renamed":
                suffix = "_cleared"
            result = self.logic.batch_clear_folder(
                inputDir=inDir,
                outputDir=outDir,
                rangeText=self.batchRangeEdit.text,
                recursive=bool(self.batchRecursiveCheck.checked),
                keepRelativeFolders=bool(self.batchKeepRelativeCheck.checked),
                suffix=suffix,
            )
            msg = f"Batch clear done. Processed: {result['processed']}, failed: {result['failed']}. Output: {outDir}"
            if result.get("errors"):
                msg += "\nFirst errors:\n" + "\n".join(result["errors"][:5])
            self.batchStatus.setText(msg)
            slicer.util.infoDisplay(msg)
        except Exception as e:
            self.batchStatus.setText(f"Error: {e}")
            slicer.util.errorDisplay(str(e))


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class ETSE_UV__MarkupNameToolsLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        try:
            super(ETSE_UV__MarkupNameToolsLogic, self).__init__()
        except Exception:
            pass

    # ------------------------------------------------------------
    # Template readers
    # ------------------------------------------------------------
    def default_template_records(self):
        return list(DEFAULT_TEMPLATE_RECORDS)

    def template_records_from_node(self, fidNode):
        if fidNode is None:
            return self.default_template_records()
        n = fidNode.GetNumberOfControlPoints()
        records = []
        for i in range(n):
            label = self._safe_get_node_label(fidNode, i)
            desc = self._safe_get_node_description(fidNode, i)
            records.append((label, desc))
        if not records:
            raise ValueError("Template node has no control points.")
        return records

    def template_records_from_file_or_default(self, templatePath):
        templatePath = (templatePath or "").strip()
        if templatePath:
            if not os.path.isfile(templatePath):
                raise ValueError(f"Template file does not exist: {templatePath}")
            return self.template_records_from_file(templatePath)
        return self.default_template_records()

    def template_records_from_file(self, templatePath):
        data = self._read_json(templatePath)
        cps = self._first_control_points(data)
        if not cps:
            raise ValueError(f"Template file has no control points: {templatePath}")
        return [(str(cp.get("label", "")), str(cp.get("description", ""))) for cp in cps]

    # ------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------
    def transcribe_node(self, targetNode, templateRecords=None, mode="anatomical_label", rangeText="all", useParsedIndex=True):
        if targetNode is None:
            raise ValueError("Target node is None.")
        templateRecords = templateRecords or self.default_template_records()
        n = targetNode.GetNumberOfControlPoints()
        if n <= 0:
            raise ValueError("Target node has no control points.")
        target_indices = self._parse_range_1based(rangeText, n)
        modified = 0

        wasModified = targetNode.StartModify() if hasattr(targetNode, "StartModify") else None
        try:
            for oneBasedTarget in target_indices:
                i = oneBasedTarget - 1
                template_index = self._template_index_for_node_point(targetNode, i, useParsedIndex)
                if template_index < 1 or template_index > len(templateRecords):
                    continue
                newLabel, newDesc = self._make_names(templateRecords[template_index - 1], mode)
                self._safe_set_node_label(targetNode, i, newLabel)
                self._safe_set_node_description(targetNode, i, newDesc)
                modified += 1
        finally:
            if wasModified is not None and hasattr(targetNode, "EndModify"):
                targetNode.EndModify(wasModified)
        return modified

    def clear_node_names(self, targetNode, rangeText="all"):
        if targetNode is None:
            raise ValueError("Target node is None.")
        n = targetNode.GetNumberOfControlPoints()
        target_indices = self._parse_range_1based(rangeText, n)
        modified = 0
        wasModified = targetNode.StartModify() if hasattr(targetNode, "StartModify") else None
        try:
            for oneBasedTarget in target_indices:
                i = oneBasedTarget - 1
                self._safe_set_node_label(targetNode, i, "")
                self._safe_set_node_description(targetNode, i, "")
                modified += 1
        finally:
            if wasModified is not None and hasattr(targetNode, "EndModify"):
                targetNode.EndModify(wasModified)
        return modified

    # ------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------
    def transcribe_file(self, inputPath, outputPath, templateRecords=None, mode="anatomical_label", rangeText="all", useParsedIndex=True):
        if not os.path.isfile(inputPath):
            raise ValueError(f"Input file does not exist: {inputPath}")
        templateRecords = templateRecords or self.default_template_records()
        data = self._read_json(inputPath)
        modified = self._transcribe_json_data(data, templateRecords, mode, rangeText, useParsedIndex)
        self._write_json(outputPath, data)
        return modified

    def clear_file_names(self, inputPath, outputPath, rangeText="all"):
        if not os.path.isfile(inputPath):
            raise ValueError(f"Input file does not exist: {inputPath}")
        data = self._read_json(inputPath)
        modified = self._clear_json_data(data, rangeText)
        self._write_json(outputPath, data)
        return modified

    # ------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------
    def batch_transcribe_folder(self, inputDir, outputDir, templateRecords=None, mode="anatomical_label",
                                rangeText="all", useParsedIndex=True, recursive=False,
                                keepRelativeFolders=True, suffix="_renamed"):
        templateRecords = templateRecords or self.default_template_records()
        paths = self._find_mrk_json_files(inputDir, recursive=recursive)
        os.makedirs(outputDir, exist_ok=True)
        result = {"processed": 0, "failed": 0, "errors": []}
        for inputPath in paths:
            try:
                outputPath = self._batch_output_path(inputPath, inputDir, outputDir, suffix, recursive, keepRelativeFolders)
                self.transcribe_file(inputPath, outputPath, templateRecords, mode, rangeText, useParsedIndex)
                result["processed"] += 1
            except Exception as e:
                result["failed"] += 1
                result["errors"].append(f"{os.path.basename(inputPath)}: {e}")
        return result

    def batch_clear_folder(self, inputDir, outputDir, rangeText="all", recursive=False,
                           keepRelativeFolders=True, suffix="_cleared"):
        paths = self._find_mrk_json_files(inputDir, recursive=recursive)
        os.makedirs(outputDir, exist_ok=True)
        result = {"processed": 0, "failed": 0, "errors": []}
        for inputPath in paths:
            try:
                outputPath = self._batch_output_path(inputPath, inputDir, outputDir, suffix, recursive, keepRelativeFolders)
                self.clear_file_names(inputPath, outputPath, rangeText)
                result["processed"] += 1
            except Exception as e:
                result["failed"] += 1
                result["errors"].append(f"{os.path.basename(inputPath)}: {e}")
        return result

    # ------------------------------------------------------------
    # Core JSON transforms
    # ------------------------------------------------------------
    def _transcribe_json_data(self, data, templateRecords, mode, rangeText, useParsedIndex):
        modified = 0
        for markup in data.get("markups", []):
            cps = markup.get("controlPoints", []) or []
            target_indices = self._parse_range_1based(rangeText, len(cps))
            for oneBasedTarget in target_indices:
                i = oneBasedTarget - 1
                cp = cps[i]
                template_index = self._template_index_for_json_point(cp, i, useParsedIndex)
                if template_index < 1 or template_index > len(templateRecords):
                    continue
                newLabel, newDesc = self._make_names(templateRecords[template_index - 1], mode)
                if newLabel is not None:
                    cp["label"] = newLabel
                if newDesc is not None:
                    cp["description"] = newDesc
                modified += 1
        return modified

    def _clear_json_data(self, data, rangeText):
        modified = 0
        for markup in data.get("markups", []):
            cps = markup.get("controlPoints", []) or []
            target_indices = self._parse_range_1based(rangeText, len(cps))
            for oneBasedTarget in target_indices:
                i = oneBasedTarget - 1
                cps[i]["label"] = ""
                cps[i]["description"] = ""
                modified += 1
        return modified

    # ------------------------------------------------------------
    # Naming policy
    # ------------------------------------------------------------
    def _make_names(self, record, mode):
        templateLabel = str(record[0] if len(record) > 0 else "")
        templateDesc = str(record[1] if len(record) > 1 else "")
        anatomical = templateDesc if templateDesc else templateLabel

        if mode == "original_style":
            return templateLabel, templateDesc
        if mode == "description_only":
            return None, templateDesc
        if mode == "numeric_only":
            return templateLabel, ""
        # anatomical_label
        return anatomical, templateDesc

    # ------------------------------------------------------------
    # Safe node helpers
    # ------------------------------------------------------------
    def _safe_get_node_label(self, node, i):
        try:
            return str(node.GetNthControlPointLabel(i))
        except Exception:
            return ""

    def _safe_get_node_description(self, node, i):
        try:
            return str(node.GetNthControlPointDescription(i))
        except Exception:
            return ""

    def _safe_set_node_label(self, node, i, label):
        if label is None:
            return
        try:
            node.SetNthControlPointLabel(i, str(label))
        except Exception:
            pass

    def _safe_set_node_description(self, node, i, desc):
        if desc is None:
            return
        try:
            node.SetNthControlPointDescription(i, str(desc))
        except Exception:
            pass

    # ------------------------------------------------------------
    # Index parsing / mapping
    # ------------------------------------------------------------
    def _parse_range_1based(self, text, n):
        text = str(text or "all").strip().lower()
        if text in ("", "all", "*"):
            return list(range(1, int(n) + 1))
        out = []
        clean = text.replace(" ", "")
        for chunk in clean.split(","):
            if not chunk:
                continue
            chunk = chunk.replace("..", "-")
            if "-" in chunk:
                parts = chunk.split("-")
                if len(parts) != 2:
                    raise ValueError(f"Bad range chunk: {chunk}")
                a = self._range_token_to_int(parts[0], n)
                b = self._range_token_to_int(parts[1], n)
                step = 1 if b >= a else -1
                out.extend(range(a, b + step, step))
            else:
                out.append(self._range_token_to_int(chunk, n))
        # Keep user order, remove duplicates, clamp to valid range.
        seen = set()
        final = []
        for x in out:
            x = int(x)
            if x < 1 or x > n:
                continue
            if x not in seen:
                final.append(x)
                seen.add(x)
        return final

    def _range_token_to_int(self, token, n):
        token = str(token).strip().lower()
        if token == "n":
            return int(n)
        return int(token)

    def _template_index_for_node_point(self, node, zeroBasedIndex, useParsedIndex):
        if useParsedIndex:
            label = self._safe_get_node_label(node, zeroBasedIndex)
            idx = self._parse_last_integer(label)
            if idx is not None:
                return idx
            try:
                cpID = str(node.GetNthControlPointID(zeroBasedIndex))
                idx = self._parse_last_integer(cpID)
                if idx is not None:
                    return idx
            except Exception:
                pass
        return int(zeroBasedIndex) + 1

    def _template_index_for_json_point(self, cp, zeroBasedIndex, useParsedIndex):
        if useParsedIndex:
            label = str(cp.get("label", ""))
            idx = self._parse_last_integer(label)
            if idx is not None:
                return idx
            cpID = str(cp.get("id", ""))
            idx = self._parse_last_integer(cpID)
            if idx is not None:
                return idx
        return int(zeroBasedIndex) + 1

    def _parse_last_integer(self, text):
        if text is None:
            return None
        matches = re.findall(r"(\d+)", str(text))
        if not matches:
            return None
        try:
            return int(matches[-1])
        except Exception:
            return None

    # ------------------------------------------------------------
    # Filesystem / JSON helpers
    # ------------------------------------------------------------
    def _read_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path, data):
        outDir = os.path.dirname(os.path.abspath(path))
        if outDir:
            os.makedirs(outDir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _first_control_points(self, data):
        markups = data.get("markups", [])
        if not markups:
            return []
        return markups[0].get("controlPoints", []) or []

    def _find_mrk_json_files(self, inputDir, recursive=False):
        if not os.path.isdir(inputDir):
            raise ValueError(f"Input folder does not exist: {inputDir}")
        out = []
        if recursive:
            for root, _dirs, files in os.walk(inputDir):
                for name in files:
                    if name.lower().endswith(".mrk.json"):
                        out.append(os.path.join(root, name))
        else:
            for name in os.listdir(inputDir):
                p = os.path.join(inputDir, name)
                if os.path.isfile(p) and name.lower().endswith(".mrk.json"):
                    out.append(p)
        out.sort()
        return out

    def _batch_output_path(self, inputPath, inputDir, outputDir, suffix, recursive, keepRelativeFolders):
        inputPath = os.path.abspath(inputPath)
        inputDir = os.path.abspath(inputDir)
        outputDir = os.path.abspath(outputDir)
        relDir = ""
        if recursive and keepRelativeFolders:
            relDir = os.path.dirname(os.path.relpath(inputPath, inputDir))
            if relDir == ".":
                relDir = ""
        outDir = os.path.join(outputDir, relDir)
        os.makedirs(outDir, exist_ok=True)
        name = os.path.basename(inputPath)
        suffix = str(suffix or "")
        if suffix:
            if name.lower().endswith(".mrk.json"):
                name = name[:-9] + suffix + ".mrk.json"
            else:
                base, ext = os.path.splitext(name)
                name = base + suffix + ext
        return os.path.join(outDir, name)
