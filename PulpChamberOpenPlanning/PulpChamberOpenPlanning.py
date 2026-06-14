import logging
import os
from typing import Annotated, Optional

import slicer.logic
import slicer.util
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

import qt
import nibabel as nib
from nibabel import processing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import shutil
from scipy import ndimage
import json
import time
from ortools.graph import pywrapgraph
from scipy.spatial import distance
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt, binary_fill_holes
from skimage.morphology import skeletonize_3d, ball
import copy
import SimpleITK as sitk
import matplotlib.pyplot as plt
import SegmentEditorEffects
import numba

# from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode, vtkMRMLMarkupsROINode

from MRMLCorePython import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode
from vtkSlicerMarkupsModuleMRMLPython import vtkMRMLMarkupsROINode

#
# PulpChamberOpenPlanning
#


class PulpChamberOpenPlanning(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("EndoPlanner")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Endodontics")]
        self.parent.dependencies = []
        self.parent.contributors = ["Yi Zhang (SJTU)", "Xiaojun Chen (SJTU)"]
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
<b>EndoPlanner</b> is an adaptive preoperative planning framework for Guided Endodontics (root canal therapy).
It integrates automated root canal landmark detection, minimally invasive access cavity planning, and surgical
template generation directly from dental CBCT scans.
See more information in the <a href="https://github.com/ZhyBrian/SlicerEndoPlanner">project repository</a>.
""")
        self.parent.acknowledgementText = _("""
This module was developed by Yi Zhang and collaborators at Shanghai Jiao Tong University and Shanghai Ninth
People's Hospital. It is built on the 3D Slicer platform and makes use of the DentalSegmentator and STU-Net
open-source projects.
""")


#
# PulpChamberOpenPlanningParameterNode
#


@parameterNodeWrapper
class PulpChamberOpenPlanningParameterNode:
    """
    The parameters needed by module.

    inputVolume - The input volume.
    outputSegmentation - The output segmentation.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    outputSegmentation: vtkMRMLSegmentationNode
    rootCanalPathNum: Annotated[int, WithinRange(0, 4)] = 0
    sliderLocalMaxCandidateNum: Annotated[int, WithinRange(0, 10)] = 0
    sliderClosestPointsNum: Annotated[int, WithinRange(0, 10)] = 5
    sliderHeatmapFilterThres: Annotated[float, WithinRange(0.0, 1.0)] = 0.0
    sliderDirCoincidenceCoeff: Annotated[float, WithinRange(0.0, 1.0)] = 0.4
    sliderSegProximityCoeff: Annotated[float, WithinRange(0.0, 1.0)] = 0.3
    sliderHeatmapSigniCoeff: Annotated[float, WithinRange(0.0, 1.0)] = 0.3
    predRootCanalPathNum: Annotated[int, WithinRange(0, 4)] = 0
    inputSegmentationOPT: vtkMRMLSegmentationNode
    zCrownIndex: Annotated[int, WithinRange(-1000000, 1000000)] = -1000000
    zPulpIndex: Annotated[int, WithinRange(-1000000, 1000000)] = -1000000
    outputPulpSection: vtkMRMLSegmentationNode
    sliderTermContour: Annotated[float, WithinRange(0.0, 20.0)] = 10.0
    sliderTermRegularization1: Annotated[float, WithinRange(0.0, 20.0)] = 0.0  # optional regularizer toward the initial access points; disabled (0.0) by default
    sliderTermRegularization2: Annotated[float, WithinRange(0.0, 20.0)] = 1.0
    sliderTermDistanceKeepFar: Annotated[float, WithinRange(0.0, 20.0)] = 7.5
    sliderTermMutualDistanceIntensity: Annotated[float, WithinRange(0.0, 4.0)] = 0.75
    sliderDistanceKeepUniform: Annotated[float, WithinRange(0.0, 20.0)] = 3.0
    sliderTermCenter: Annotated[float, WithinRange(0.0, 20.0)] = 1.0
    inputSuperVolume: vtkMRMLScalarVolumeNode
    inputTotalDentalSegmentation: vtkMRMLSegmentationNode
    upperTeethOrLowerTeeth: Annotated[int, WithinRange(0, 2)] = 0         # 0: None, 1: Upper teeth, 2: Lower teeth
    outputBottomGuideBeforeCropSegmentation: vtkMRMLSegmentationNode
    inputGuidePlateCoverRegionROI: vtkMRMLMarkupsROINode
    inputBottomGuideBeforeCropSegmentation: vtkMRMLSegmentationNode
    outputBottomGuidePlateSegmentation: vtkMRMLSegmentationNode
    outputTopGuidePlateSegmentation: vtkMRMLSegmentationNode
    accessDesignPreference: Annotated[int, WithinRange(0, 1)] = 0         # 0: More consistent with clinical experience, 1: More consistent with specific root canal anatomy
    
    
#
# PulpChamberOpenPlanningWidget
#


class PulpChamberOpenPlanningWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/PulpChamberOpenPlanning.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = PulpChamberOpenPlanningLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButtonLDM.connect("clicked(bool)", self.onapplyButtonLDM)
        self.ui.applyButtonOPT.connect("clicked(bool)", self.onapplyButtonOPT)
        self.ui.pushButtonSetZCrownIndex.connect("clicked(bool)", self.onPushButtonSetZCrownIndex)
        self.ui.pushButtonSetZPulpIndex.connect("clicked(bool)", self.onPushButtonSetZPulpIndex)
        self.ui.pushButtonDecideAutoPulpOPT.connect("clicked(bool)", self.onPushButtonDecideAutoPulpOPT)
        self.ui.applyButtonGPDS1.connect("clicked(bool)", self.onapplyButtonGPDS1)
        self.ui.applyButtonGPDS2.connect("clicked(bool)", self.onapplyButtonGPDS2)
        
        # Selectors
        self.ui.rootCanalPathSelector.connect("currentIndexChanged(int)", self.onRootCanalPathSelectorIndexChanged)
        self.ui.upperLowerTeethSelector.connect("currentIndexChanged(int)", self.onUpperLowerTeethSelectorIndexChanged)
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUIinput)
        self.ui.inputROISelectorGPD.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUIinputROI)
        self.ui.comboBoxDesignPreference.connect("currentIndexChanged(int)", self.onComboBoxDesignPreferenceIndexChanged)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self._parameterNode.zCrownIndex = -1000000
        self._parameterNode.zPulpIndex = -1000000
        
        self.ui.rootCanalPathSelector.addItems(["Decide automatically", "1 root canal path", "2 root canal path", "3 root canal path", "4 root canal path"])
        self.ui.comboBoxDesignPreference.addItems(["More consistent with clinical experience", "More consistent with specific root canal anatomy"])
        self.ui.upperLowerTeethSelector.addItems(["None", "Upper teeth", "Lower teeth"])
        

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[PulpChamberOpenPlanningParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.outputSegmentation:
            self.ui.applyButtonLDM.toolTip = _("Compute outputs")
            self.ui.applyButtonLDM.enabled = True
        else:
            self.ui.applyButtonLDM.toolTip = _("Select input volume and output segmentation nodes")
            self.ui.applyButtonLDM.enabled = False
        
        if self._parameterNode and self._parameterNode.inputSegmentationOPT and self._parameterNode.zCrownIndex != -1000000 and self._parameterNode.outputPulpSection:
            self.ui.applyButtonOPT.toolTip = _("Compute outputs")
            self.ui.applyButtonOPT.enabled = True
        else:
            self.ui.applyButtonOPT.toolTip = _("Select input segmentation, z crown index, and output pulp section nodes")
            self.ui.applyButtonOPT.enabled = False
        
        if self._parameterNode and self._parameterNode.inputSuperVolume and self._parameterNode.inputTotalDentalSegmentation and self._parameterNode.outputBottomGuideBeforeCropSegmentation:
            self.ui.applyButtonGPDS1.toolTip = _("Compute outputs")
            self.ui.applyButtonGPDS1.enabled = True
        else:
            self.ui.applyButtonGPDS1.toolTip = _("Select input super volume, input total dental segmentation, upper teeth or lower teeth, and output bottom guide before crop segmentation nodes")
            self.ui.applyButtonGPDS1.enabled = False
        
        if self._parameterNode and self._parameterNode.inputGuidePlateCoverRegionROI and self._parameterNode.outputBottomGuidePlateSegmentation and self._parameterNode.outputTopGuidePlateSegmentation and self._parameterNode.inputSuperVolume and self._parameterNode.inputTotalDentalSegmentation and self._parameterNode.inputBottomGuideBeforeCropSegmentation:
            self.ui.applyButtonGPDS2.toolTip = _("Compute outputs")
            self.ui.applyButtonGPDS2.enabled = True
        else:
            self.ui.applyButtonGPDS2.toolTip = _("Select input guide plate cover region ROI, output bottom guide plate segmentation, output top guide plate segmentation, input super volume, input total dental segmentation, and input bottom guide before crop segmentation nodes")
            self.ui.applyButtonGPDS2.enabled = False


    def onRootCanalPathSelectorIndexChanged(self, index: int) -> None:
        """Handle change in root canal path selector."""
        if index == 0:
            self.ui.SliderLocalMaxCandidateNum.enabled = False
        else:
            self.ui.SliderLocalMaxCandidateNum.enabled = True
            
        self._parameterNode.rootCanalPathNum = int(index)
        self.ui.SliderLocalMaxCandidateNum.setValue(index * 2)
        self._parameterNode.sliderLocalMaxCandidateNum = int(index * 2)

    def onComboBoxDesignPreferenceIndexChanged(self, index: int) -> None:
        """Handle change in design preference combo box."""
        self._parameterNode.accessDesignPreference = int(index)
    
    def onUpperLowerTeethSelectorIndexChanged(self, index: int) -> None:
        """Handle change in upper lower teeth selector."""
        self._parameterNode.upperTeethOrLowerTeeth = int(index)
    
    def updateParameterNodeFromGUIinput(self) -> None:
        """Handle change in input volume selector."""
        self._parameterNode.predRootCanalPathNum = int(0)
        self.ui.labelPredRootCanalNum.setText("None")
        # self._parameterNode.inputVolume = self.ui.inputSelector.currentNode()
        slicer.util.setSliceViewerLayers(background=self.ui.inputSelector.currentNodeID)
        slicer.util.resetSliceViews()
    
    def onPushButtonSetZCrownIndex(self) -> None:
        """Handle click on set z crown index button."""
        currentOffset = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset()
        bounds = [0,] * 6
        slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceBounds(bounds)
        LowerBound = bounds[4]
        UpperBound = bounds[5]
        numberOfChannels = self.ui.inputSelector.currentNode().GetImageData().GetDimensions()[2]
        self._parameterNode.zCrownIndex = int(fromCurrentOffsettoCurrentSlice(currentOffset, LowerBound, UpperBound, numberOfChannels))
        self.ui.labelZCrownIndex.setText(str(self._parameterNode.zCrownIndex))
    
    def onPushButtonSetZPulpIndex(self) -> None:
        """Handle click on set z pulp index button."""
        currentOffset = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset()
        bounds = [0,] * 6
        slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceBounds(bounds)
        LowerBound = bounds[4]
        UpperBound = bounds[5]
        numberOfChannels = self.ui.inputSelector.currentNode().GetImageData().GetDimensions()[2]
        self._parameterNode.zPulpIndex = int(fromCurrentOffsettoCurrentSlice(currentOffset, LowerBound, UpperBound, numberOfChannels))
        self.ui.labelZPulpIndex.setText(str(self._parameterNode.zPulpIndex))
        
    def onPushButtonDecideAutoPulpOPT(self) -> None:
        self._parameterNode.zPulpIndex = -1000000
        self.ui.labelZPulpIndex.setText("Decide automatically")
    
    def updateParameterNodeFromGUIinputROI(self) -> None:
        """Handle change in input ROI selector."""
        if self.ui.inputROISelectorGPD.currentNode() and self.ui.inputSuperVolumeSelector.currentNode():
            # if the roi is just created, set the size and center of the roi
            isSizeAllZero = self.ui.inputROISelectorGPD.currentNode().GetSize()[0] == 0.0 and self.ui.inputROISelectorGPD.currentNode().GetSize()[1] == 0.0 and self.ui.inputROISelectorGPD.currentNode().GetSize()[2] == 0.0
            if isSizeAllZero:
                originWorld = self.ui.inputSuperVolumeSelector.currentNode().GetOrigin()
                size = self.ui.inputSuperVolumeSelector.currentNode().GetImageData().GetDimensions()
                spacing = self.ui.inputSuperVolumeSelector.currentNode().GetSpacing()
                sizeWorld = [size[0] * spacing[0], size[1] * spacing[1], size[2] * spacing[2]]
                dirMat = [[0.0, 0.0, 0.0] for i in range(3)]
                self.ui.inputSuperVolumeSelector.currentNode().GetIJKToRASDirections(dirMat)
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            assert dirMat[i][j] == 0.0
                centerWorld = [originWorld[i] + sizeWorld[i] * dirMat[i][i] / 2 for i in range(3)]
                self.ui.inputROISelectorGPD.currentNode().SetSizeWorld(sizeWorld[0], sizeWorld[1], sizeWorld[2])
                self.ui.inputROISelectorGPD.currentNode().SetCenterWorld(centerWorld[0], centerWorld[1], centerWorld[2])
            
            ROIDisplayNode = self.ui.inputROISelectorGPD.currentNode().GetDisplayNode()
            ROIDisplayNode.FillVisibilityOn()

        else:
            if self.ui.inputSuperVolumeSelector.currentNode() is None:
                node_temp = self.ui.inputROISelectorGPD.currentNode()
                self.ui.inputROISelectorGPD.setCurrentNode(None)
                slicer.mrmlScene.RemoveNode(node_temp)
                raise ValueError("Please select input super volume node.")
        

    def onapplyButtonLDM(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            
            make_dirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), "TmpFilesLDM"))
            
            self._parameterNode.rootCanalPathNum = int(self.ui.rootCanalPathSelector.currentIndex)
            self._parameterNode.sliderLocalMaxCandidateNum = int(self.ui.SliderLocalMaxCandidateNum.value)
            self._parameterNode.sliderClosestPointsNum = int(self.ui.SliderClosestPointsNum.value)
            self._parameterNode.sliderHeatmapFilterThres = self.ui.SliderHeatmapFilterThres.value
            self._parameterNode.sliderDirCoincidenceCoeff = self.ui.SliderDirCoincidenceCoeff.value
            self._parameterNode.sliderSegProximityCoeff = self.ui.SliderSegProximityCoeff.value
            self._parameterNode.sliderHeatmapSigniCoeff = self.ui.SliderHeatmapSigniCoeff.value
            
            predRootCanalPathNum = self.logic.processLDM(self.ui.inputSelector.currentNode(), self.ui.outputSelectorLDM.currentNode(),
                               self._parameterNode.rootCanalPathNum,
                               self._parameterNode.sliderLocalMaxCandidateNum,
                               self._parameterNode.sliderClosestPointsNum,
                               self._parameterNode.sliderHeatmapFilterThres,
                               self._parameterNode.sliderDirCoincidenceCoeff,
                               self._parameterNode.sliderSegProximityCoeff,
                               self._parameterNode.sliderHeatmapSigniCoeff,
                               self.ui.inputTotalDentalSegmentSelector.currentNode())
            
            self._parameterNode.predRootCanalPathNum = int(predRootCanalPathNum)
            self.ui.labelPredRootCanalNum.setText(str(self._parameterNode.predRootCanalPathNum))
            self.ui.inputSelectorOPT.setCurrentNode(self.ui.outputSelectorLDM.currentNode())
    
    
    def onapplyButtonOPT(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            
            make_dirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), "TmpFilesOPT"))
            
            self._parameterNode.sliderTermContour = self.ui.SliderTermContour.value
            self._parameterNode.sliderTermRegularization1 = self.ui.SliderTermRegularization1.value
            self._parameterNode.sliderTermRegularization2 = self.ui.SliderTermRegularization2.value
            self._parameterNode.sliderTermDistanceKeepFar = self.ui.SliderTermDistanceKeepFar.value
            self._parameterNode.sliderTermMutualDistanceIntensity = self.ui.SliderTermMutualDistanceIntensity.value
            self._parameterNode.sliderDistanceKeepUniform = self.ui.SliderDistanceKeepUniform.value
            self._parameterNode.sliderTermCenter = self.ui.SliderTermCenter.value
            self._parameterNode.accessDesignPreference = int(self.ui.comboBoxDesignPreference.currentIndex)
            
            self.logic.processOPT(self.ui.inputSelector.currentNode(),
                               self.ui.inputSelectorOPT.currentNode(), self._parameterNode.zCrownIndex, self._parameterNode.zPulpIndex, 
                               self.ui.outputSelectorOPT.currentNode(),
                               self._parameterNode.sliderTermContour,
                               self._parameterNode.sliderTermRegularization1,
                               self._parameterNode.sliderTermRegularization2,
                               self._parameterNode.sliderTermDistanceKeepFar,
                               self._parameterNode.sliderTermMutualDistanceIntensity,
                               self._parameterNode.sliderDistanceKeepUniform,
                               self._parameterNode.sliderTermCenter,
                               self._parameterNode.accessDesignPreference)
    
    
    def onapplyButtonGPDS1(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            
            self._parameterNode.upperTeethOrLowerTeeth = int(self.ui.upperLowerTeethSelector.currentIndex)
            
            self.logic.processGPDS1(self.ui.inputSuperVolumeSelector.currentNode(),
                               self.ui.inputTotalDentalSegmentSelector.currentNode(),
                               self._parameterNode.upperTeethOrLowerTeeth,
                               self.ui.outputBottomGuideBeforeCropSelectorGPD.currentNode())
            
            self.ui.inputBottomGuideBeforeCropSelectorGPD.setCurrentNode(self.ui.outputBottomGuideBeforeCropSelectorGPD.currentNode())
    
    
    def onapplyButtonGPDS2(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            
            make_dirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), "TmpFilesGPD"))

            self._parameterNode.upperTeethOrLowerTeeth = int(self.ui.upperLowerTeethSelector.currentIndex)
            
            self.logic.processGPDS2(self.ui.inputSuperVolumeSelector.currentNode(),
                               self.ui.inputTotalDentalSegmentSelector.currentNode(),
                               self.ui.inputROISelectorGPD.currentNode(),
                               self.ui.outputBottomGuidePlateSelectorGPD.currentNode(),
                               self.ui.outputTopGuidePlateSelectorGPD.currentNode(),
                               self.ui.inputBottomGuideBeforeCropSelectorGPD.currentNode(),
                               self._parameterNode.upperTeethOrLowerTeeth)
            
            
            



#
# PulpChamberOpenPlanningLogic
#


class PulpChamberOpenPlanningLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return PulpChamberOpenPlanningParameterNode(super().getParameterNode())


    def processLDM(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputSegmentation: vtkMRMLSegmentationNode,
                rootCanalPathNum,
                sliderLocalMaxCandidateNum,
                sliderClosestPointsNum,
                sliderHeatmapFilterThres,
                sliderDirCoincidenceCoeff,
                sliderSegProximityCoeff,
                sliderHeatmapSigniCoeff,
                inputTotalDentalSegmentation) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: input CBCT scalar volume
        :param outputSegmentation: vtkMRMLSegmentationNode
        """
        
        if inputTotalDentalSegmentation:
            inputTotalDentalSegmentationDisplayNode = inputTotalDentalSegmentation.GetDisplayNode()
            inputTotalDentalSegmentationDisplayNode.SetOpacity(0.15)
         
        # remove all the markups
        markupsNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsCurveNode")
        for markupsNode in markupsNodes:
            slicer.mrmlScene.RemoveNode(markupsNode)

        if not inputVolume or not outputSegmentation:
            raise ValueError("Input volume or output segmentation is invalid")
        
        inputVolumeNodeName = inputVolume.GetName()
        print(inputVolumeNodeName)
        
        print("rootCanalPathNum: ", rootCanalPathNum)
        print("sliderLocalMaxCandidateNum: ", sliderLocalMaxCandidateNum)
        print("sliderClosestPointsNum: ", sliderClosestPointsNum)
        print("sliderHeatmapFilterThres: ", sliderHeatmapFilterThres)
        print("sliderDirCoincidenceCoeff: ", sliderDirCoincidenceCoeff)
        print("sliderSegProximityCoeff: ", sliderSegProximityCoeff)
        print("sliderHeatmapSigniCoeff: ", sliderHeatmapSigniCoeff)
        
        thisDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "TmpFilesLDM")
        
        # save input volume to nii.gz
        saveInputVolumeDir = os.path.join(thisDir, inputVolumeNodeName + ".nii.gz")
        slicer.util.exportNode(inputVolume, saveInputVolumeDir)        #  slicer.util.exportNode(inputVolume, saveInputVolumeDir, {}, True)
        
        # create curve json template
        origin = inputVolume.GetOrigin()
        pointPositions = np.random.randn(3, 3) * 5 + np.array(origin)[np.newaxis, :]
        curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
        curveNode.SetName("PredictedRootCanalPath")
        slicer.util.updateMarkupsControlPointsFromArray(curveNode, pointPositions)
        slicer.util.saveNode(curveNode, os.path.join(thisDir, "template.json"))
        slicer.mrmlScene.RemoveNode(curveNode)
          
        # Load the multi-task landmark-detection network (`model`): a STUNet backbone with heatmap,
        # pulp-segmentation, and root-canal-count heads, corresponding to the network described in the
        # paper. `model_seg` is a segmentation checkpoint of the same STUNet backbone, used to produce
        # the pulp mask in this preview pipeline.
        in_channels = 1
        num_classes_seg = 2
        num_classes_hm = 3
        num_classes_cls = 4

        net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]]
        net_conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

        model_seg = STUNet_v2(input_channels=in_channels, num_classes=num_classes_seg, depth=[1, 1, 1, 1, 1, 1], dims=[16, 32, 64, 128, 256, 256],
                       pool_op_kernel_sizes=net_num_pool_op_kernel_sizes, conv_kernel_sizes=net_conv_kernel_sizes, use_output_v2=True)
        model = STUNet_hm_cls_v2(input_channels=in_channels, num_classes_seg=num_classes_seg,
                              num_classes_hm=num_classes_hm, num_classes_cls=num_classes_cls,
                              depth=[1, 1, 1, 1, 1, 1], dims=[16, 32, 64, 128, 256, 256],
                              pool_op_kernel_sizes=net_num_pool_op_kernel_sizes,
                              conv_kernel_sizes=net_conv_kernel_sizes, use_output_v2=True)
        
        model_seg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ModelWeights", "STU-NET-S-V2.pth")
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ModelWeights", "STU-NET-S-HM-CLS-V2.pth")
        model_seg.load_state_dict(torch.load(model_seg_path, map_location=torch.device('cpu'))['model_state_dict'])
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model_seg.to(device)
        model.to(device)
        model_seg.eval()
        model.eval()
          
        # preprocess the input volume
        uni_size = (96, 96, 160)
        ctimg = load_medical_image_normalize_np(saveInputVolumeDir, type='ctimg', 
                                                rescale=uni_size, rescale_order=1,
                                                window_center=1500, window_width=3000)
        affine, size = load_medical_image_affine_and_size(saveInputVolumeDir)
        ctimg, start_pad_idx, end_pad_idx = pad_volume_to_center(ctimg, uni_size)         
        ctimg = ctimg.transpose((2, 0, 1)) 
        
        scale = min([uni_size[0] * 1.0 / size[0], uni_size[1] * 1.0 / size[1], uni_size[2] * 1.0 / size[2]])
        pre_affine_matrix = np.eye(4)
        pre_affine_matrix[0, 0] = scale
        pre_affine_matrix[1, 1] = scale
        pre_affine_matrix[2, 2] = scale
        pre_affine_matrix[0, 3] = start_pad_idx[0]
        pre_affine_matrix[1, 3] = start_pad_idx[1]
        pre_affine_matrix[2, 3] = start_pad_idx[2]
        
        input_tensor = torch.FloatTensor(ctimg.copy()).unsqueeze(0).unsqueeze(0).to(device)
        input_tensor.requires_grad = False
        
        # model inference
        __, output_hm, output_cls = model(input_tensor)
        output_seg = model_seg(input_tensor)

        output_hm = torch.sigmoid(output_hm)
        
        # postprocess the output: cls
        pred_root_canal_num = torch.argmax(output_cls.squeeze()).item() + 1
        
        # postprocess the output: seg
        thres = 0.5
        output_seg = nn.Softmax(dim=1)(output_seg) 
        
        output_seg_m = output_seg.clone()
        output_seg_m = output_seg_m.squeeze()[1, :, :, :]
        output_seg_m = torch.where(output_seg_m >= thres, torch.tensor(1, device=device), torch.tensor(0, device=device))
        output_seg_m = largest_connected_component(output_seg_m)
        
        output_seg_p = output_seg.squeeze().detach().cpu().numpy()
        output_seg_p = output_seg_p[1, :, :, :]
        output_seg_p = (output_seg_p >= thres).astype(np.uint8)
        output_seg_p = largest_connected_component(output_seg_p)
        output_seg_p = output_seg_p.transpose(1, 2, 0)
        output_seg_p = output_seg_p[start_pad_idx[0]:end_pad_idx[0], start_pad_idx[1]:end_pad_idx[1], start_pad_idx[2]:end_pad_idx[2]]
        height_, width_, depth_ = output_seg_p.shape
        scale_reverse = [size[0] * 1.0 / height_, size[1] * 1.0 / width_, size[2] * 1.0 / depth_]  
        output_seg_p = ndimage.interpolation.zoom(output_seg_p, scale_reverse, order=0, mode='nearest')
        mask_nii = nib.Nifti1Image(output_seg_p, affine)
        nib.save(mask_nii, os.path.join(thisDir, 'Seg_pred_' + inputVolumeNodeName + '.nii.gz'))
        
        # slicer.util.loadSegmentation(os.path.join(thisDir, 'Seg_pred_' + inputVolumeNodeName + '.nii.gz'), returnNode=True)
        outputSegmentation.CreateDefaultDisplayNodes()
        if outputSegmentation.GetSegmentation().GetNumberOfSegments() >= 1:
            currentSegmentID = outputSegmentation.GetSegmentation().GetNthSegmentID(0)
        else:
            currentSegmentID = outputSegmentation.GetSegmentation().AddEmptySegment()
        output_seg_p_to_slicer = output_seg_p.copy()
        output_seg_p_to_slicer = np.transpose(output_seg_p_to_slicer, axes=(2, 1, 0))
        slicer.util.updateSegmentBinaryLabelmapFromArray(output_seg_p_to_slicer, outputSegmentation, currentSegmentID, inputVolume)
        segmentationDisplayNode = outputSegmentation.GetDisplayNode()
        segmentationDisplayNode.SetSegmentOverrideColor(currentSegmentID, 1.0, 1.0, 0.0)
        segmentationDisplayNode.SetOpacity(0.3)
        # https://apidocs.slicer.org/master/classvtkMRMLSegmentationDisplayNode.html
        outputSegmentation.CreateClosedSurfaceRepresentation()
        
        # postprocess the output: hm
        if rootCanalPathNum == 0:
            DecideRootCanalPathNum = pred_root_canal_num
            sliderLocalMaxCandidateNum = pred_root_canal_num * 2
        else:
            DecideRootCanalPathNum = rootCanalPathNum
        datas = []
        with open(os.path.join(thisDir, "template.json"), 'r') as f:
            data = json.load(f)
        for i in range(DecideRootCanalPathNum):
            datas.append(data)
        matrix_orientation = np.array(datas[0]["markups"][0]["controlPoints"][0]["orientation"], dtype=float).reshape(3, 3)
        
        decode_points_list = decode_heatmap_v2(output_hm=output_hm, root_canal_num=DecideRootCanalPathNum, 
                                                seg_mask=output_seg_m, 
                                                k=sliderLocalMaxCandidateNum, 
                                                threshold=sliderHeatmapFilterThres, 
                                                n=sliderClosestPointsNum,
                                                c1=sliderDirCoincidenceCoeff,
                                                c2=sliderSegProximityCoeff,
                                                c3=sliderHeatmapSigniCoeff)
        
        t_mat = np.matmul(affine, np.linalg.inv(pre_affine_matrix))
        mat_ori_4x4 = np.eye(4)
        mat_ori_4x4[:3, :3] = matrix_orientation
        t_mat = np.matmul(np.linalg.inv(mat_ori_4x4), t_mat)
        decode_points_list_world = matrix4x4_multiply_multidim_vector3(t_mat, decode_points_list)

        print('decode_points_list_world: ', decode_points_list_world)
        
        make_dirs(thisDir + ' - Out Json')
        save_json_data_ldm(datas, 
                        thisDir, 
                        decode_points_list_world, 
                        'pred_root_canal_ldms', 
                        color=[0.5, 0.5, 1.0], 
                        return_json_dict=False)
        for json_name in os.listdir(thisDir + ' - Out Json'):
            slicer.util.loadMarkups(os.path.join(thisDir + ' - Out Json', json_name))
        
        
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()
        
        
        predRootCanalPathNum = pred_root_canal_num
        
        return predRootCanalPathNum
    
    
    def processOPT(self,
                   inputVolume: vtkMRMLScalarVolumeNode,
                   inputSegmentationOPT: vtkMRMLSegmentationNode,
                   zCrownIndex,
                   zPulpIndex,
                   outputPulpSection: vtkMRMLSegmentationNode,
                   sliderTermContour,
                   sliderTermRegularization1,
                   sliderTermRegularization2,
                   sliderTermDistanceKeepFar,
                   sliderTermMutualDistanceIntensity,
                   sliderDistanceKeepUniform,
                   sliderTermCenter,
                   accessDesignPreference) -> None:
        
        if not inputVolume or not inputSegmentationOPT or zCrownIndex == -1000000 or not outputPulpSection:
            raise ValueError("Input volume, input segmentation, z crown index, or output pulp section is invalid")
        
        if inputSegmentationOPT is outputPulpSection:
            raise ValueError("Input segmentation and output pulp section cannot be the same")
        
        inputVolumeNodeName = inputVolume.GetName()
        print(inputVolumeNodeName)
        
        print("zCrownIndex: ", zCrownIndex)
        print("zPulpIndex: ", zPulpIndex)
        print("sliderTermContour: ", sliderTermContour)
        print("sliderTermRegularization1: ", sliderTermRegularization1)
        print("sliderTermRegularization2: ", sliderTermRegularization2)
        print("sliderTermDistanceKeepFar: ", sliderTermDistanceKeepFar)
        print("sliderTermMutualDistanceIntensity: ", sliderTermMutualDistanceIntensity)
        print("sliderDistanceKeepUniform: ", sliderDistanceKeepUniform)
        print("sliderTermCenter: ", sliderTermCenter)
        
        thisDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "TmpFilesOPT")
        
        # save input volume to nii.gz
        saveInputVolumeDir = os.path.join(thisDir, inputVolumeNodeName + ".nii.gz")
        slicer.util.exportNode(inputVolume, saveInputVolumeDir)        #  slicer.util.exportNode(inputVolume, saveInputVolumeDir, {}, True)
        
        # save input segmentation to nii.gz
        saveInputSegmentationOPTDir = os.path.join(thisDir, "Segment_" + inputVolumeNodeName + ".nii.gz")
        SegmentID = inputSegmentationOPT.GetSegmentation().GetNthSegmentID(0)
        segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(inputSegmentationOPT, SegmentID, inputVolume)
        segmentArray = segmentArray.transpose(2, 1, 0)

        affine, size = load_medical_image_affine_and_size(saveInputVolumeDir)
        mask_nii = nib.Nifti1Image(segmentArray, affine)
        nib.save(mask_nii, saveInputSegmentationOPTDir)
        
        # save markups to json
        markupsNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsCurveNode")
        for markupsNode in markupsNodes:
            slicer.util.saveNode(markupsNode, os.path.join(thisDir, markupsNode.GetName() + ".json"))
        
        
        make_dirs(thisDir + ' - Out Json')
        
        if zPulpIndex == -1000000:
            json_points_processer = JsonPointsProcesser(thisDir, start_at_root=False, z_crown_index=zCrownIndex)
        else:
            json_points_processer = JsonPointsProcesser(thisDir, start_at_root=False, z_crown_index=zCrownIndex, z_pulp_index=zPulpIndex)
    
        ######################################################################

        boundary_image, map_centroid, distance_map, sum_area = region_image_to_boundary_image(json_points_processer.region_image_at_z_crown.astype(int),
                                                                                            show=False)
        print('distance_map:', distance_map.shape, distance_map.dtype, distance_map.min(), distance_map.max())
        
        root_canal_near_points = json_points_processer.all_points_index[:, 0, :]
        root_canal_central_directions = json_points_processer.all_central_tangent_index
        mean_root_canal_central_direction = json_points_processer.mean_root_direction_index
        z_crown = json_points_processer.z_crown_index
        print('z_crown:', z_crown)
        project_points = []
        for near_point, central_direction in zip(root_canal_near_points, root_canal_central_directions):
            if accessDesignPreference == 0:
                x = (z_crown - near_point[2]) / mean_root_canal_central_direction[2]
                project_points.append(near_point + x * mean_root_canal_central_direction)
            elif accessDesignPreference == 1:
                x = (z_crown - near_point[2]) / central_direction[2]
                project_points.append(near_point + x * central_direction)
            else:
                raise ValueError("Access design preference is invalid")
        project_points = np.array(project_points, dtype=float)
        project_points = project_points[:, :2]
        project_points = torch.from_numpy(project_points).float()
        project_points.requires_grad_(False)
        print('project_points:', project_points, project_points.requires_grad)
        
        
        points = torch.from_numpy(json_points_processer.crown_points_index).float()
        points = points[:, :2]
        points.requires_grad_(True)
        points_initial = points.clone().detach()  
        points_initial = points_initial[:, :2]
        points_initial.requires_grad_(False) 
        print('points:', points, points.requires_grad)
        print('points_initial:', points_initial, points_initial.requires_grad)
        print('sum_area:', sum_area)

        data = torch.tensor(distance_map, 
                            requires_grad=False, dtype=torch.float32)#.unsqueeze(0).unsqueeze(0)
        map_centroid = torch.tensor(map_centroid, dtype=torch.float32, requires_grad=False)
        print('map_centroid: ', map_centroid, map_centroid.requires_grad)

        ############################parameters################################
        lr = 0.1
        epochs = 4500
        # Constrained (primal-dual) optimization of the access points P_a under the containment
        # constraint B(P_a) = 0: the access points descend on the scalarized objective while the
        # non-negative constraint multiplier `delta` ascends. The "Points in region" slider seeds
        # the initial multiplier value.
        delta = torch.tensor(float(sliderTermContour), requires_grad=True)
        optimizer = torch.optim.Adam([points], lr=lr)
        dual_optimizer = torch.optim.Adam([delta], lr=lr / 10, maximize=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs / 9, eta_min=lr / 1000)
        ######################################################################
        
        min_loss = 1e20
        best_points = points.clone()

        sliderTermMutualDistanceIntensityIndex = world_distance_to_index_scalar(json_points_processer.affine_matrix,
                                                                                json_points_processer.matrix_orientation,
                                                                                sliderTermMutualDistanceIntensity)
        print('sliderTermMutualDistanceIntensityIndex:', sliderTermMutualDistanceIntensityIndex)

        for step in range(epochs):
            optimizer.zero_grad()
            dual_optimizer.zero_grad()
            loss = objective_function(data, points, points_initial, map_centroid, sum_area,
                                    project_points, show=False,
                                    term_contour_coeff=delta,
                                    term_regularization_1_coeff=sliderTermRegularization1,
                                    term_regularization_2_coeff=sliderTermRegularization2,
                                    term_distance_keepfar_coeff=sliderTermDistanceKeepFar,
                                    term_mutual_distance_intensity=sliderTermMutualDistanceIntensityIndex,
                                    term_distance_keepuniform_coeff=sliderDistanceKeepUniform,
                                    term_center_coeff=sliderTermCenter)
            
            loss.backward()
            optimizer.step()              # primal step: gradient descent on the access points P_a
            dual_optimizer.step()         # dual step: gradient ascent on the constraint multiplier delta
            with torch.no_grad():
                delta.clamp_(min=0.0)     # project the multiplier onto the feasible set delta >= 0

            if loss.item() < min_loss:
                min_loss = loss.item()  
                best_points = points.clone()

            if step % 500 == 0:
                print(f'Step {step}: points = {points.detach()}, loss = {loss.item()}')
                
            lr_scheduler.step()

        print(f'Best loss: {min_loss}')
        print(f'Best points: {best_points.detach()}')
        objective_function(data, best_points, points_initial, map_centroid, sum_area, 
                        project_points, show=True,
                        term_contour_coeff=delta.detach(),
                        term_regularization_1_coeff=sliderTermRegularization1,
                        term_regularization_2_coeff=sliderTermRegularization2,
                        term_distance_keepfar_coeff=sliderTermDistanceKeepFar,
                        term_mutual_distance_intensity=sliderTermMutualDistanceIntensityIndex,
                        term_distance_keepuniform_coeff=sliderDistanceKeepUniform,
                        term_center_coeff=sliderTermCenter)
        
        best_points = best_points.detach().numpy()
        json_points_processer.update_optimized_results(best_points, show=False)
        
        
        for json_name in os.listdir(thisDir + ' - Out Json'):
            if json_name.endswith('.json'):
                node_temp = slicer.util.loadMarkups(os.path.join(thisDir + ' - Out Json', json_name))
                if 'origin' in json_name:
                    node_temp.GetDisplayNode().SetVisibility(False)
        
        outputPulpSection.CreateDefaultDisplayNodes()
        if outputPulpSection.GetSegmentation().GetNumberOfSegments() >= 1:
            currentSegmentID = outputPulpSection.GetSegmentation().GetNthSegmentID(0)
        else:
            currentSegmentID = outputPulpSection.GetSegmentation().AddEmptySegment()
        output_pulp_section_to_slicer = json_points_processer.segment_volume_new.copy()
        output_pulp_section_to_slicer = np.transpose(output_pulp_section_to_slicer, axes=(2, 1, 0))
        slicer.util.updateSegmentBinaryLabelmapFromArray(output_pulp_section_to_slicer, outputPulpSection, currentSegmentID, inputVolume)
        segmentationDisplayNode = outputPulpSection.GetDisplayNode()
        segmentationDisplayNode.SetSegmentOverrideColor(currentSegmentID, 0.8, 0.0, 0.8)
        segmentationDisplayNode.SetOpacity(0.5)
        segmentationDisplayNode.SetVisibility(False)
        # https://apidocs.slicer.org/master/classvtkMRMLSegmentationDisplayNode.html
        outputPulpSection.CreateClosedSurfaceRepresentation()
        
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()
        
        red_slice = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeRed')
        red_slice.SetSliceVisible(True)
    
    
    def processGPDS1(self,
                     inputSuperVolume,
                     inputTotalDentalSegmentation,
                     upperTeethOrLowerTeeth,
                     outputBottomGuideBeforeCropSegmentation) -> None:
        
        if not inputSuperVolume or not inputTotalDentalSegmentation or not outputBottomGuideBeforeCropSegmentation:
            raise ValueError("Input super volume, input total dental segmentation, output bottom guide before crop, or upper teeth or lower teeth is invalid")
        
        inputTotalDentalSegmentationDisplayNode = inputTotalDentalSegmentation.GetDisplayNode()
        inputTotalDentalSegmentationDisplayNode.SetOpacity(0.15)
        
        inputSuperVolumeNodeName = inputSuperVolume.GetName()
        print(inputSuperVolumeNodeName)
        
        if upperTeethOrLowerTeeth == 1:
            SegmentID = inputTotalDentalSegmentation.GetSegmentation().GetNthSegmentID(2)  
        elif upperTeethOrLowerTeeth == 2:
            SegmentID = inputTotalDentalSegmentation.GetSegmentation().GetNthSegmentID(3)
        else:
            try:
                SegmentID = inputTotalDentalSegmentation.GetSegmentation().GetNthSegmentID(0)
            except:
                raise ValueError("Upper teeth or lower teeth is invalid")
        segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(inputTotalDentalSegmentation, SegmentID, inputSuperVolume)

        outputBottomGuideBeforeCropSegmentation.CreateDefaultDisplayNodes()
        if outputBottomGuideBeforeCropSegmentation.GetSegmentation().GetNumberOfSegments() >= 1:
            currentSegmentID = outputBottomGuideBeforeCropSegmentation.GetSegmentation().GetNthSegmentID(0)
        else:
            currentSegmentID = outputBottomGuideBeforeCropSegmentation.GetSegmentation().AddEmptySegment()
        slicer.util.updateSegmentBinaryLabelmapFromArray(segmentArray, outputBottomGuideBeforeCropSegmentation, currentSegmentID, inputSuperVolume)
        segmentationDisplayNode = outputBottomGuideBeforeCropSegmentation.GetDisplayNode()
        segmentationDisplayNode.SetSegmentOverrideColor(currentSegmentID, 0.8, 0.8, 0.8)
        segmentationDisplayNode.SetOpacity(0.6)
        segmentationDisplayNode.SetVisibility(True)
        
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        segmentEditorNode.SetAndObserveSegmentationNode(outputBottomGuideBeforeCropSegmentation)
        segmentEditorNode.SetAndObserveSourceVolumeNode(inputSuperVolume)
        
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(outputBottomGuideBeforeCropSegmentation)
        segmentEditorWidget.setSourceVolumeNode(inputSuperVolume)
        
        segmentEditorNode.SetSelectedSegmentID(currentSegmentID)
        segmentEditorWidget.setCurrentSegmentID(currentSegmentID)
        
        thickness = 1.5        
        print('thickness:', thickness)
        effect1 = segmentEditorWidget.effectByName("Hollow")
        effect1.setParameter('ShellMode', 'INSIDE_SURFACE')          # 'INSIDE_SURFACE' or 'OUTSIDE_SURFACE'
        effect1.setParameter('ShellThicknessMm', thickness)                # input parameter
        effect1.self().onApply()
        
        effect2 = segmentEditorWidget.effectByName("Islands")
        effect2.setParameter("Operation", SegmentEditorEffects.KEEP_LARGEST_ISLAND)
        effect2.self().onApply()

        segmentEditorWidget = None
        slicer.mrmlScene.RemoveNode(segmentEditorNode)
        
        outputBottomGuideBeforeCropSegmentation.CreateClosedSurfaceRepresentation()
         
    
    def processGPDS2(self,
                   inputSuperVolume,
                   inputTotalDentalSegmentation,
                   inputGuidePlateCoverRegionROI,
                   outputBottomGuidePlateSegmentation,
                   outputTopGuidePlateSegmentation,
                   inputBottomGuideBeforeCropSegmentation,
                   upperTeethOrLowerTeeth) -> None:
        

        if not inputSuperVolume or not inputTotalDentalSegmentation or not inputGuidePlateCoverRegionROI or not outputBottomGuidePlateSegmentation or not outputTopGuidePlateSegmentation or not inputBottomGuideBeforeCropSegmentation:
            raise ValueError("Input super volume, input total dental segmentation, input guide plate cover region ROI, output bottom guide plate segmentation, output top guide plate segmentation, or input bottom guide before crop segmentation is invalid")
        
        inputTotalDentalSegmentationDisplayNode = inputTotalDentalSegmentation.GetDisplayNode()
        inputTotalDentalSegmentationDisplayNode.SetOpacity(0.15)
        
        inputSuperVolumeNodeName = inputSuperVolume.GetName()
        print(inputSuperVolumeNodeName)
        
        thisDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "TmpFilesGPD")


        if upperTeethOrLowerTeeth == 1:
            ulSegmentID = inputTotalDentalSegmentation.GetSegmentation().GetNthSegmentID(2)
        elif upperTeethOrLowerTeeth == 2:
            ulSegmentID = inputTotalDentalSegmentation.GetSegmentation().GetNthSegmentID(3)
        else:
            try:
                ulSegmentID = inputTotalDentalSegmentation.GetSegmentation().GetNthSegmentID(0)
            except:
                raise ValueError("Upper teeth or lower teeth is invalid")
        teethSegmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(inputTotalDentalSegmentation, ulSegmentID, inputSuperVolume)

        
        ###################################################crop bottom guide plate###################################################
        
        SegmentBottomGuideBeforeCropID = inputBottomGuideBeforeCropSegmentation.GetSegmentation().GetNthSegmentID(0)  
        segmentBottomGuideBeforeCropArray = slicer.util.arrayFromSegmentBinaryLabelmap(inputBottomGuideBeforeCropSegmentation, SegmentBottomGuideBeforeCropID, inputSuperVolume)
        inputBottomGuideBeforeCropSegmentationDisplayNode = inputBottomGuideBeforeCropSegmentation.GetDisplayNode()
        inputBottomGuideBeforeCropSegmentationDisplayNode.SetVisibility3D(False)
        
        superVolumeShape = slicer.util.arrayFromVolume(inputSuperVolume).shape
        roiArray = np.zeros(superVolumeShape, dtype=np.uint8)   
        
        roiBounds = [0]*6
        inputGuidePlateCoverRegionROI.GetRASBounds(roiBounds)
        rasToIJKMatrix = vtk.vtkMatrix4x4()
        inputSuperVolume.GetRASToIJKMatrix(rasToIJKMatrix)
        ijkToRASMatrix = vtk.vtkMatrix4x4()
        inputSuperVolume.GetIJKToRASMatrix(ijkToRASMatrix)
        
        def rasToIjk(rasPoint, rasToIJKMatrix):
            rasPointHomogeneous = [rasPoint[0], rasPoint[1], rasPoint[2], 1.0]
            ijkPointHomogeneous = rasToIJKMatrix.MultiplyPoint(rasPointHomogeneous)
            ijkPoint = [ijkPointHomogeneous[i] for i in range(3)]
            return ijkPoint

        # Convert ROI bounds to IJK coordinates
        rasMin = [roiBounds[0], roiBounds[2], roiBounds[4]]
        rasMax = [roiBounds[1], roiBounds[3], roiBounds[5]]
        ijkMin = rasToIjk(rasMin, rasToIJKMatrix)
        ijkMax = rasToIjk(rasMax, rasToIJKMatrix)
        ijkMin = [int(np.floor(c)) for c in ijkMin]
        ijkMax = [int(np.ceil(c)) for c in ijkMax]
        
        # Ensure indices are within valid range
        maxIJK = roiArray.shape  # (Z, Y, X)
        def clamp(value, minValue, maxValue):
            return max(minValue, min(value, maxValue))

        ijkMin = [clamp(ijkMin[i], 0, maxIJK[i]-1) for i in range(3)]
        ijkMax = [clamp(ijkMax[i], 0, maxIJK[i]-1) for i in range(3)]
        ijkMinf = [min(ijkMin[0], ijkMax[0]), min(ijkMin[1], ijkMax[1]), min(ijkMin[2], ijkMax[2])]
        ijkMaxf = [max(ijkMin[0], ijkMax[0]), max(ijkMin[1], ijkMax[1]), max(ijkMin[2], ijkMax[2])]
        roiArray[ijkMinf[2]:ijkMaxf[2]+1, ijkMinf[1]:ijkMaxf[1]+1, ijkMinf[0]:ijkMaxf[0]+1] = 1
        
        resultBottomGuideArray = np.logical_and(segmentBottomGuideBeforeCropArray, roiArray).astype(np.uint8)
        
        
        
        
        ###################################################crop top guide plate###################################################
        
        
        all_start_points = []
        all_end_points = []
        
        markupsNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsCurveNode")
        for markupsNode in markupsNodes:
            if "optimized" in markupsNode.GetName():
                markupsNodeControlPoints = slicer.util.arrayFromMarkupsCurvePoints(markupsNode)
                all_start_points.append(np.array(markupsNodeControlPoints[0]))
                print('start:', markupsNodeControlPoints[0])
                all_end_points.append(np.array(markupsNodeControlPoints[-1]))
                print('end:', markupsNodeControlPoints[-1])
        
        all_start_points = np.array(all_start_points)  # (n, 3)
        all_end_points = np.array(all_end_points)      # (n, 3)

        direction_vectors = all_end_points - all_start_points  # (n, 3)
        all_norm_directions = direction_vectors / np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]
        offset = 1.5
        all_base_centers = all_end_points + offset * all_norm_directions
        
        base_radius = 2.2   
        cylinder_height = 2.5  
        small_cylinder_radius = 0.75  
        
        ijkToRAS = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                ijkToRAS[i, j] = ijkToRASMatrix.GetElement(i, j)
        
        
        
        def morphological_close_transform(image_3d, dilation_iterations=1, erosion_iterations=1, structure_element=np.ones((3, 3, 3), dtype=np.uint8)):
            dilated_image = image_3d.copy()
            dilated_image = binary_dilation(dilated_image, structure=structure_element, iterations=dilation_iterations)

            eroded_image = dilated_image.copy()
            eroded_image = binary_erosion(eroded_image, structure=structure_element, iterations=erosion_iterations)

            return eroded_image
        
        
        
        main_geometry_array = np.zeros(superVolumeShape, dtype=np.uint8)
        inside_subtract_geometry_array = np.zeros(superVolumeShape, dtype=np.uint8)
        upward_substract_geometry_array = np.zeros(superVolumeShape, dtype=np.uint8)
        downward_substract_geometry_array = np.zeros(superVolumeShape, dtype=np.uint8)
        upward_add_geometry_array = np.zeros(superVolumeShape, dtype=np.uint8)
        for (base_center, norm_direction) in zip(all_base_centers, all_norm_directions):
            main_geometry_array = np.logical_or(main_geometry_array, create_cylinder_optimized(superVolumeShape, base_radius, base_center, base_radius, norm_direction,
                                                cylinder_height, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=False)).astype(np.uint8)
            inside_subtract_geometry_array = np.logical_or(inside_subtract_geometry_array, create_cylinder_optimized(superVolumeShape, base_radius, base_center-0.8*offset*norm_direction, small_cylinder_radius, norm_direction,
                                                cylinder_height*2.5, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=False)).astype(np.uint8)
            upward_substract_geometry_array = np.logical_or(upward_substract_geometry_array, create_cylinder_optimized(superVolumeShape, base_radius, base_center, base_radius+0.15, norm_direction,
                                                cylinder_height*3.0, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=False)).astype(np.uint8)
            downward_substract_geometry_array = np.logical_or(downward_substract_geometry_array, create_cylinder_optimized(superVolumeShape, base_radius, base_center+0.5*offset*norm_direction, base_radius*0.7, -norm_direction,
                                                cylinder_height*3.0, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=False)).astype(np.uint8)
            upward_add_geometry_array = np.logical_or(upward_add_geometry_array, create_cylinder_optimized(superVolumeShape, base_radius, base_center-1.0*offset*norm_direction, base_radius*1.8, norm_direction,
                                                cylinder_height+1.0*offset, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=False)).astype(np.uint8)
        upward_substract_geometry_array = morphological_close_transform(upward_substract_geometry_array, dilation_iterations=5, erosion_iterations=5)
        downward_substract_geometry_array = morphological_close_transform(downward_substract_geometry_array, dilation_iterations=5, erosion_iterations=5)

        main_geometry_array_f = main_geometry_array & (~inside_subtract_geometry_array)
        main_geometry_array_f = (main_geometry_array_f > 0).astype(np.uint8)

        intersect_check_t = np.logical_and(main_geometry_array_f, teethSegmentArray)
        if np.sum(intersect_check_t) > 0:
            raise ValueError("The top guide plate intersects with the teeth.")
        
        outputTopGuidePlateSegmentation.CreateDefaultDisplayNodes()
        if outputTopGuidePlateSegmentation.GetSegmentation().GetNumberOfSegments() >= 1:
            currentSegmentOutputTopGuideID = outputTopGuidePlateSegmentation.GetSegmentation().GetNthSegmentID(0)
        else:
            currentSegmentOutputTopGuideID = outputTopGuidePlateSegmentation.GetSegmentation().AddEmptySegment()
        slicer.util.updateSegmentBinaryLabelmapFromArray(main_geometry_array_f, outputTopGuidePlateSegmentation, currentSegmentOutputTopGuideID, inputSuperVolume)
        outputTopGuidePlateSegmentationDisplayNode = outputTopGuidePlateSegmentation.GetDisplayNode()
        outputTopGuidePlateSegmentationDisplayNode.SetSegmentOverrideColor(currentSegmentOutputTopGuideID, 0.8, 0.0, 0.8)
        outputTopGuidePlateSegmentationDisplayNode.SetOpacity(0.75)
        outputTopGuidePlateSegmentationDisplayNode.SetVisibility(True)
        outputTopGuidePlateSegmentation.CreateClosedSurfaceRepresentation()
        
        
        
        
        
        resultBottomGuideArray = np.logical_or(upward_add_geometry_array, resultBottomGuideArray).astype(np.uint8)
        intersect_part_upward = np.logical_and(upward_substract_geometry_array, resultBottomGuideArray)
        intersect_part_downward = np.logical_and(downward_substract_geometry_array, resultBottomGuideArray)
        intersect_part = np.logical_or(intersect_part_upward, intersect_part_downward)
        intersect_part_from_add = np.logical_and(upward_add_geometry_array, teethSegmentArray)
        intersect_part = np.logical_or(intersect_part, intersect_part_from_add)
        resultBottomGuideArray_f = resultBottomGuideArray & (~intersect_part)
        # ensure binary
        resultBottomGuideArray_f = (resultBottomGuideArray_f > 0).astype(np.uint8)
        resultBottomGuideArray_f = largest_connected_component(resultBottomGuideArray_f)
        print('resultBottomGuideArray_f:', resultBottomGuideArray_f.shape)

        intersect_check_b = np.logical_and(resultBottomGuideArray_f, teethSegmentArray)
        if np.sum(intersect_check_b) > 0:
            raise ValueError("The bottom guide plate intersects with the teeth.")
          
        # temp
        outputBottomGuidePlateSegmentation.CreateDefaultDisplayNodes()
        if outputBottomGuidePlateSegmentation.GetSegmentation().GetNumberOfSegments() >= 1:
            currentSegmentOutputBottomGuideID = outputBottomGuidePlateSegmentation.GetSegmentation().GetNthSegmentID(0)
        else:
            currentSegmentOutputBottomGuideID = outputBottomGuidePlateSegmentation.GetSegmentation().AddEmptySegment()
        slicer.util.updateSegmentBinaryLabelmapFromArray(resultBottomGuideArray_f, outputBottomGuidePlateSegmentation, currentSegmentOutputBottomGuideID, inputSuperVolume)
        outputBottomGuidePlateSegmentationDisplayNode = outputBottomGuidePlateSegmentation.GetDisplayNode()
        outputBottomGuidePlateSegmentationDisplayNode.SetSegmentOverrideColor(currentSegmentOutputBottomGuideID, 0.8, 0.8, 0.8)
        outputBottomGuidePlateSegmentationDisplayNode.SetOpacity(0.6)
        outputBottomGuidePlateSegmentationDisplayNode.SetVisibility(True)
        outputBottomGuidePlateSegmentation.CreateClosedSurfaceRepresentation()
        


        slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsClosedSurfaceRepresentationToFiles(
            thisDir,
            outputBottomGuidePlateSegmentation,
            None,
            "STL",
            True,
            1.0,
            False
        )
        slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsClosedSurfaceRepresentationToFiles(
            thisDir,
            outputTopGuidePlateSegmentation,
            None,
            "STL",
            True,
            1.0,
            False
        )
        
        
        
        inputGuidePlateCoverRegionROIDisplayNode = inputGuidePlateCoverRegionROI.GetDisplayNode()
        inputGuidePlateCoverRegionROIDisplayNode.FillVisibilityOff()











########################################################GuidePlateDesign########################################################


########################################################GuidePlateDesign########################################################





#####################################################Optimization#####################################################


#####################################################Optimization#####################################################




#####################################################decode#####################################################


#####################################################decode#####################################################




####################################################utils####################################################


###################################################utils####################################################





#####################################################network#####################################################


#####################################################network#####################################################







#
# PulpChamberOpenPlanningTest
#


class PulpChamberOpenPlanningTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_PulpChamberOpenPlanning1()

    def test_PulpChamberOpenPlanning1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        self.delayDisplay("Test passed")
