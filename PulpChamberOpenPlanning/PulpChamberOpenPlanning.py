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
        self.parent.title = _("PulpChamberOpenPlanning")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Utilities")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Yi Zhang (SJTU)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#PulpChamberOpenPlanning">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # PulpChamberOpenPlanning1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="PulpChamberOpenPlanning",
        sampleName="PulpChamberOpenPlanning1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "PulpChamberOpenPlanning1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="PulpChamberOpenPlanning1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="PulpChamberOpenPlanning1",
    )

    # PulpChamberOpenPlanning2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="PulpChamberOpenPlanning",
        sampleName="PulpChamberOpenPlanning2",
        thumbnailFileName=os.path.join(iconsPath, "PulpChamberOpenPlanning2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="PulpChamberOpenPlanning2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="PulpChamberOpenPlanning2",
    )


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
    sliderTermRegularization1: Annotated[float, WithinRange(0.0, 20.0)] = 0.0
    sliderTermRegularization2: Annotated[float, WithinRange(0.0, 20.0)] = 2.0
    sliderTermDistanceKeepFar: Annotated[float, WithinRange(0.0, 20.0)] = 15.0
    sliderTermMutualDistanceIntensity: Annotated[float, WithinRange(0.0, 4.0)] = 0.75
    sliderDistanceKeepUniform: Annotated[float, WithinRange(0.0, 20.0)] = 6.0
    sliderTermCenter: Annotated[float, WithinRange(0.0, 20.0)] = 2.0
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
        :param inputVolume: volume to be thresholded
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
          
        # load the model
        in_channels = 1
        num_classes_seg = 2
        num_classes_hm = 3
        num_classes_cls = 4
        # net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 2]]
        net_num_pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1]]
        net_conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        # model_seg = STUNet(input_channels=in_channels, num_classes=num_classes_seg, depth=[1, 1, 1, 1, 1, 1], dims=[16, 32, 64, 128, 256, 256],
        #                pool_op_kernel_sizes=net_num_pool_op_kernel_sizes, conv_kernel_sizes=net_conv_kernel_sizes)
        # model_seg = STUNet_v2(input_channels=in_channels, num_classes=num_classes_seg, depth=[1, 1, 1, 1, 1, 1], dims=[16, 32, 64, 128, 256, 256],
        #                pool_op_kernel_sizes=net_num_pool_op_kernel_sizes, conv_kernel_sizes=net_conv_kernel_sizes, use_output_v2=True)
        # model = STUNet_hm_cls(input_channels=in_channels, num_classes_seg=num_classes_seg,
        #                       num_classes_hm=num_classes_hm, num_classes_cls=num_classes_cls,
        #                       depth=[1, 1, 1, 1, 1, 1], dims=[16, 32, 64, 128, 256, 256],
        #                pool_op_kernel_sizes=net_num_pool_op_kernel_sizes, conv_kernel_sizes=net_conv_kernel_sizes)
        model = STUNet_hm_cls_v2(input_channels=in_channels, num_classes_seg=num_classes_seg,
                              num_classes_hm=num_classes_hm, num_classes_cls=num_classes_cls,
                              depth=[1, 1, 1, 1, 1, 1], dims=[16, 32, 64, 128, 256, 256],
                              pool_op_kernel_sizes=net_num_pool_op_kernel_sizes,
                              conv_kernel_sizes=net_conv_kernel_sizes, use_output_v2=True)
        
        # model_seg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ModelWeights", "STU-NET-S-V2.pth")
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ModelWeights", "STU-NET-S-HM-CLS-V2_2026.pth")
        # model_seg.load_state_dict(torch.load(model_seg_path, map_location=torch.device('cpu'))['model_state_dict'])
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # model_seg.to(device)
        model.to(device)
        # model_seg.eval()
        model.eval()
          
        # preprocess the input volume
        uni_size = (96, 96, 160)       # 128, 128, 192 or 96, 96, 144
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
        # __, output_hm, output_cls = model(input_tensor)
        # output_seg = model_seg(input_tensor)
        output_seg, output_hm, output_cls = model(input_tensor)

        output_hm = torch.sigmoid(output_hm)
        
        # postprocess the output: cls
        pred_root_canal_num = torch.argmax(output_cls.squeeze()).item() + 1
        
        # postprocess the output: seg
        thres = 0.5
        output_seg = nn.Softmax(dim=1)(output_seg) 
        
        output_seg_m = output_seg.clone()
        output_seg_m = output_seg_m.squeeze()[1, :, :, :]
        output_seg_m = torch.where(output_seg_m >= thres, torch.tensor(1).cuda(), torch.tensor(0).cuda())
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
        # decode_points_list_world = decode_points_list_world.transpose(1, 0, 2)   # need change
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
        # print(segmentArray.shape)
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
        optimizer = torch.optim.Adam([points], lr=lr)
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
            loss = objective_function(data, points, points_initial, map_centroid, sum_area, 
                                    project_points, show=False,
                                    term_contour_coeff=sliderTermContour,
                                    term_regularization_1_coeff=sliderTermRegularization1,
                                    term_regularization_2_coeff=sliderTermRegularization2,
                                    term_distance_keepfar_coeff=sliderTermDistanceKeepFar,
                                    term_mutual_distance_intensity=sliderTermMutualDistanceIntensityIndex,
                                    term_distance_keepuniform_coeff=sliderDistanceKeepUniform,
                                    term_center_coeff=sliderTermCenter)
            
            loss.backward()
            optimizer.step()
            
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
                        term_contour_coeff=sliderTermContour,
                        term_regularization_1_coeff=sliderTermRegularization1,
                        term_regularization_2_coeff=sliderTermRegularization2,
                        term_distance_keepfar_coeff=sliderTermDistanceKeepFar,
                        term_mutual_distance_intensity=sliderTermMutualDistanceIntensityIndex,
                        term_distance_keepuniform_coeff=sliderDistanceKeepUniform,
                        term_center_coeff=sliderTermCenter)
        
        best_points = best_points.detach().numpy()
        json_points_processer.update_optimized_results(best_points, show=False)

        # print('Affine Mat: ', json_points_processer.affine_matrix)
        
        
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
        
        thickness = 1.5        # 原来是2.5 mm
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
        # print('segmentBottomGuideBeforeCropArray:', segmentBottomGuideBeforeCropArray.shape)   # z, y, x
        inputBottomGuideBeforeCropSegmentationDisplayNode = inputBottomGuideBeforeCropSegmentation.GetDisplayNode()
        inputBottomGuideBeforeCropSegmentationDisplayNode.SetVisibility3D(False)
        
        superVolumeShape = slicer.util.arrayFromVolume(inputSuperVolume).shape
        roiArray = np.zeros(superVolumeShape, dtype=np.uint8)   
        # print('superVolumeShape:', superVolumeShape)   # z, y, x
        
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
        # print('ijkMin:', ijkMin)
        # print('ijkMax:', ijkMax)
        ijkMinf = [min(ijkMin[0], ijkMax[0]), min(ijkMin[1], ijkMax[1]), min(ijkMin[2], ijkMax[2])]
        ijkMaxf = [max(ijkMin[0], ijkMax[0]), max(ijkMin[1], ijkMax[1]), max(ijkMin[2], ijkMax[2])]
        # print('ijkMinf:', ijkMinf)
        # print('ijkMaxf:', ijkMaxf)
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

        # 计算方向向量和平均方向
        direction_vectors = all_end_points - all_start_points  # (n, 3)
        all_norm_directions = direction_vectors / np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]
        # 计算所有终点的质心并偏移 offset 毫米
        offset = 1.5
        all_base_centers = all_end_points + offset * all_norm_directions
        
        # average_direction = np.mean(direction_vectors, axis=0)
        # average_direction_norm = np.linalg.norm(average_direction)
        # if average_direction_norm == 0:
        #     raise ValueError("平均方向向量的模为零，无法计算方向。")
        # average_direction_unit = average_direction / average_direction_norm  # 单位向量


        # base_center = np.mean(all_end_points, axis=0)
        # base_center = base_center + offset * average_direction_unit
        # print('base_center:', base_center)

        # 计算底面半径
        # distances = np.linalg.norm(all_end_points - base_center, axis=1)
        # max_distance = np.max(distances)
        # base_radius = 3.0 * max_distance          # 1.5
        base_radius = 2.2   # 原来是 3.0 mm

        # 主圆柱体的高度为 2 毫米
        cylinder_height = 2.5  # 毫米
        
        small_cylinder_radius = 0.75  #   (可以弄小一点：1.0 mm, 原来：1.2 mm)

        # # 主圆柱体的顶点坐标
        # top_center = base_center + cylinder_height * average_direction_unit
        
        ijkToRAS = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                ijkToRAS[i, j] = ijkToRASMatrix.GetElement(i, j)
        
        # main_geometry_array = create_cylinder(superVolumeShape, base_center, base_radius, average_direction_unit, 
        #                                       cylinder_height, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=True, 
        #                                       small_cylinder_radius=small_cylinder_radius, all_end_points=all_end_points, direction_vectors=direction_vectors)
        
        
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
        outputTopGuidePlateSegmentationDisplayNode.SetSegmentOverrideColor(currentSegmentOutputTopGuideID, 1.0, 0.8, 0.75)
        outputTopGuidePlateSegmentationDisplayNode.SetOpacity(0.75)
        outputTopGuidePlateSegmentationDisplayNode.SetVisibility(True)
        outputTopGuidePlateSegmentation.CreateClosedSurfaceRepresentation()
        
        
        
        
        
        
        # upward_substract_geometry_array = create_cylinder(superVolumeShape, base_center, base_radius, average_direction_unit, 
        #                                       10.0, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=False)
        # downward_substract_geometry_array = create_cylinder(superVolumeShape, base_center, base_radius*0.8, -average_direction_unit, 
        #                                       offset*4, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=False)
        # upward_add_geometry_array = create_cylinder(superVolumeShape, base_center, base_radius*1.2, average_direction_unit, 
        #                                       cylinder_height, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=False)
        
        resultBottomGuideArray = np.logical_or(upward_add_geometry_array, resultBottomGuideArray).astype(np.uint8)
        intersect_part_upward = np.logical_and(upward_substract_geometry_array, resultBottomGuideArray)
        intersect_part_downward = np.logical_and(downward_substract_geometry_array, resultBottomGuideArray)
        intersect_part = np.logical_or(intersect_part_upward, intersect_part_downward)
        intersect_part_from_add = np.logical_and(upward_add_geometry_array, teethSegmentArray)
        intersect_part = np.logical_or(intersect_part, intersect_part_from_add)
        resultBottomGuideArray_f = resultBottomGuideArray & (~intersect_part)
        # resultBottomGuideArray_f = resultBottomGuideArray - intersect_part
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

def batch_ijk_to_ras(ijk_coords, ijkToRAS):
    num_points = ijk_coords.shape[0]
    ijk_coords_hom = np.hstack((ijk_coords, np.ones((num_points, 1))))
    ras_coords_hom = ijk_coords_hom.dot(ijkToRAS.T)
    return ras_coords_hom[:, :3]

def ras_to_ijk(ras_point, rasToIJKMatrix):
    ras_point_hom = np.array([ras_point[0], ras_point[1], ras_point[2], 1.0])
    ijk_point_hom = rasToIJKMatrix.MultiplyPoint(ras_point_hom)
    ijk_point = [ijk_point_hom[i] for i in range(3)]
    return ijk_point


# def create_cylinder(superVolumeShape, base_center, base_radius, average_direction_unit,
#                     cylinder_height, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=True, small_cylinder_radius=1.0,
#                     all_end_points=None, direction_vectors=None):
#
#     # 获取主圆柱体的包围盒（世界坐标系）
#     cylinder_points = np.array([
#         base_center + average_direction_unit * t + np.array([dx, dy, dz])
#         for t in [0, cylinder_height]
#         for dx in [-8.0*base_radius, 8.0*base_radius]
#         for dy in [-8.0*base_radius, 8.0*base_radius]
#         for dz in [-8.0*base_radius, 8.0*base_radius]
#     ])
#
#
#     # 将包围盒顶点转换为 IJK 坐标
#     ijk_points = np.array([ras_to_ijk(point, rasToIJKMatrix) for point in cylinder_points])
#
#     # 获取包围盒的最小和最大索引
#     ijk_min = np.floor(ijk_points.min(axis=0)).astype(int)
#     ijk_max = np.ceil(ijk_points.max(axis=0)).astype(int)
#
#     # 确保索引在有效范围内
#     volume_shape = superVolumeShape  # (Z, Y, X)
#     ijk_min = np.maximum(ijk_min, [0, 0, 0])
#     ijk_max = np.minimum(ijk_max, np.array(volume_shape) - 1)
#
#     # 计算局部数组的形状
#     local_shape = ijk_max - ijk_min + 1  # (Z_size, Y_size, X_size)
#
#     # 创建局部数组
#     local_array = np.zeros(local_shape, dtype=np.uint8)
#
#     Z, Y, X = np.meshgrid(
#         np.arange(ijk_min[0], ijk_max[0]+1),
#         np.arange(ijk_min[1], ijk_max[1]+1),
#         np.arange(ijk_min[2], ijk_max[2]+1),
#         indexing='ij'
#     )
#
#     # 将 IJK 坐标转换为 RAS 坐标
#     ijk_coords = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
#     ras_coords = batch_ijk_to_ras(ijk_coords, ijkToRAS)
#
#     # 计算点到底面中心的向量
#     vec = ras_coords - base_center
#
#     # 计算每个点在轴线方向上的投影长度
#     projection_length = np.dot(vec, average_direction_unit)
#
#     # 判断是否在圆柱体高度范围内
#     in_height = np.logical_and(projection_length >= 0, projection_length <= cylinder_height)
#
#     # 计算投影点坐标
#     projection_point = base_center + np.outer(projection_length, average_direction_unit)
#
#     # 计算点到轴线的径向距离
#     radial_distance = np.linalg.norm(ras_coords - projection_point, axis=1)
#
#     # 判断是否在半径范围内
#     in_radius = radial_distance <= base_radius
#
#     # 最终判断点是否在圆柱体内
#     in_cylinder = np.logical_and(in_height, in_radius)
#
#     # 将结果映射回局部数组
#     local_array_flat = in_cylinder.astype(np.uint8)
#     local_array = local_array_flat.reshape(local_shape)
#
#
#     if substrateSmallCylinder:
#
#         all_norm_directions = direction_vectors / np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]
#         all_far_points = all_end_points + 5 * cylinder_height * all_norm_directions
#         all_near_points = all_end_points - 2 * cylinder_height * all_norm_directions
#
#         for start_point, end_point in zip(all_near_points, all_far_points):
#             # 计算轴线方向和长度
#             direction_vector = end_point - start_point
#             length = np.linalg.norm(direction_vector)
#             if length == 0:
#                 continue
#             axis_direction = direction_vector / length
#
#             # 计算小圆柱体的顶点坐标
#             cylinder_points = np.array([
#                 start_point + axis_direction * t + np.array([dx, dy, dz])
#                 for t in [0, length]
#                 for dx in [-15*small_cylinder_radius, 15*small_cylinder_radius]
#                 for dy in [-15*small_cylinder_radius, 15*small_cylinder_radius]
#                 for dz in [-15*small_cylinder_radius, 15*small_cylinder_radius]
#             ])
#
#             # 将包围盒顶点转换为 IJK 坐标
#             ijk_points = np.array([ras_to_ijk(point, rasToIJKMatrix) for point in cylinder_points])
#
#             # 获取包围盒的最小和最大索引
#             ijk_min_cyl = np.floor(ijk_points.min(axis=0)).astype(int)
#             ijk_max_cyl = np.ceil(ijk_points.max(axis=0)).astype(int)
#
#             # 确保索引在有效范围内
#             ijk_min_cyl = np.maximum(ijk_min_cyl, ijk_min)
#             ijk_max_cyl = np.minimum(ijk_max_cyl, ijk_max)
#
#             # 如果没有重叠，跳过
#             if np.any(ijk_min_cyl > ijk_max_cyl):
#                 continue
#
#             # 计算局部索引范围相对于主局部数组的偏移
#             offset = ijk_min_cyl - ijk_min
#             local_cyl_shape = ijk_max_cyl - ijk_min_cyl + 1
#
#             # 获取局部坐标
#             Z_cyl, Y_cyl, X_cyl = np.meshgrid(
#                 np.arange(ijk_min_cyl[0], ijk_max_cyl[0]+1),
#                 np.arange(ijk_min_cyl[1], ijk_max_cyl[1]+1),
#                 np.arange(ijk_min_cyl[2], ijk_max_cyl[2]+1),
#                 indexing='ij'
#             )
#             ijk_coords_cyl = np.column_stack((X_cyl.ravel(), Y_cyl.ravel(), Z_cyl.ravel()))
#             ras_coords_cyl = batch_ijk_to_ras(ijk_coords_cyl, ijkToRAS)
#
#             # 在局部区域内计算小圆柱体
#             vec = ras_coords_cyl - start_point
#             projection_length = np.dot(vec, axis_direction)
#             in_height = np.logical_and(projection_length >= 0, projection_length <= length)
#             projection_point = start_point + np.outer(projection_length, axis_direction)
#             radial_distance = np.linalg.norm(ras_coords_cyl - projection_point, axis=1)
#             in_radius = radial_distance <= small_cylinder_radius
#             in_cylinder = np.logical_and(in_height, in_radius)
#             small_cylinder_array_flat = in_cylinder.astype(np.uint8)
#             small_cylinder_array = small_cylinder_array_flat.reshape(local_cyl_shape)
#
#             # 从主局部数组中减去小圆柱体
#             z_slice = slice(offset[0], offset[0]+local_cyl_shape[0])
#             y_slice = slice(offset[1], offset[1]+local_cyl_shape[1])
#             x_slice = slice(offset[2], offset[2]+local_cyl_shape[2])
#             local_array[z_slice, y_slice, x_slice] = np.where(
#                 small_cylinder_array == 1, 0, local_array[z_slice, y_slice, x_slice]
#             )
#
#
#     # 创建全局数组
#     main_geometry_array = np.zeros(volume_shape, dtype=np.uint8)
#
#     # 将局部数组放回全局数组
#     main_geometry_array[ijk_min[0]:ijk_max[0]+1, ijk_min[1]:ijk_max[1]+1, ijk_min[2]:ijk_max[2]+1] = local_array
#
#     return main_geometry_array


@numba.njit(parallel=True)
def points_in_cylinder_optimized(ras_coords, base_center, axis_direction, cylinder_height, base_radius):
    """
    优化的点在圆柱体内判断函数
    使用numba加速，支持并行处理
    """
    n = ras_coords.shape[0]
    in_cyl = np.zeros(n, dtype=np.uint8)

    # 预计算常量
    base_radius_sq = base_radius * base_radius

    for i in numba.prange(n):
        # 计算点到基础中心的向量
        vec_x = ras_coords[i, 0] - base_center[0]
        vec_y = ras_coords[i, 1] - base_center[1]
        vec_z = ras_coords[i, 2] - base_center[2]

        # 计算沿轴向的投影长度
        projection_length = (vec_x * axis_direction[0] +
                             vec_y * axis_direction[1] +
                             vec_z * axis_direction[2])

        # 检查是否在高度范围内
        if 0 <= projection_length <= cylinder_height:
            # 计算到轴线的径向距离（平方）
            # 使用向量叉积的性质：|v - (v·n)n|² = |v|² - (v·n)²
            vec_length_sq = vec_x * vec_x + vec_y * vec_y + vec_z * vec_z
            radial_distance_sq = vec_length_sq - projection_length * projection_length

            # 检查是否在半径范围内
            if radial_distance_sq <= base_radius_sq:
                in_cyl[i] = 1

    return in_cyl



def create_cylinder_optimized(superVolumeShape, base_radius_range, base_center, base_radius,
                              average_direction_unit, cylinder_height, ijkToRAS, rasToIJKMatrix,
                              substrateSmallCylinder=False, small_cylinder_radius=1.0,
                              all_end_points=None, direction_vectors=None):
    """
    优化的圆柱体创建函数

    参数:
        superVolumeShape: 体积的形状 (Z, Y, X)
        base_radius_range: 基础半径范围（未使用，保留接口兼容性）
        base_center: 圆柱体基础中心的RAS坐标
        base_radius: 圆柱体半径
        average_direction_unit: 圆柱体轴向单位向量
        cylinder_height: 圆柱体高度
        ijkToRAS: IJK到RAS的转换矩阵
        rasToIJKMatrix: RAS到IJK的转换矩阵
        substrateSmallCylinder: 是否减去小圆柱体（已移除此功能）
        其他参数: 为接口兼容性保留

    返回:
        main_geometry_array: 包含圆柱体的二值体积
    """

    # 计算圆柱体的精确包围盒
    # 圆柱体的两个端面中心
    top_center = base_center + cylinder_height * average_direction_unit

    # 为了获得精确的包围盒，我们需要考虑圆柱体在各个方向上的极值点
    # 创建一个正交基，其中一个轴是圆柱体的轴向
    # 找到两个与轴向垂直的单位向量
    if abs(average_direction_unit[0]) < 0.9:
        perp1 = np.cross(average_direction_unit, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(average_direction_unit, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(average_direction_unit, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    # 生成圆柱体表面的关键点（用于精确包围盒计算）
    key_points = []
    # 添加两个端面的圆周上的点
    n_circle_points = 8  # 圆周上的采样点数
    for t in [0, cylinder_height]:
        center = base_center + t * average_direction_unit
        for angle in np.linspace(0, 2 * np.pi, n_circle_points, endpoint=False):
            point = center + base_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            key_points.append(point)

    key_points = np.array(key_points)
    print(key_points)

    # 将关键点转换为IJK坐标
    ijk_points = np.array([ras_to_ijk(point, rasToIJKMatrix) for point in key_points])

    # 获取包围盒的最小和最大索引，添加一些余量
    margin = 2  # 像素余量
    ijk_min = np.floor(ijk_points.min(axis=0) - margin).astype(int)
    ijk_max = np.ceil(ijk_points.max(axis=0) + margin).astype(int)

    print('min:', ijk_min, 'max:', ijk_max)

    # 确保索引在有效范围内
    volume_shape = superVolumeShape  # (Z, Y, X)
    ijk_min = np.maximum(ijk_min, [0, 0, 0])
    ijk_max = np.minimum(ijk_max, np.array(volume_shape) - 1)

    # 计算局部数组的形状
    local_shape = ijk_max - ijk_min + 1

    # 创建IJK坐标网格（注意顺序：Z, Y, X）
    # Z, Y, X = np.meshgrid(
    #     np.arange(ijk_min[0], ijk_max[0] + 1),
    #     np.arange(ijk_min[1], ijk_max[1] + 1),
    #     np.arange(ijk_min[2], ijk_max[2] + 1),
    #     indexing='ij'
    # )
    Z, Y, X = np.meshgrid(
        np.arange(ijk_min[2], ijk_max[2] + 1),
        np.arange(ijk_min[1], ijk_max[1] + 1),
        np.arange(ijk_min[0], ijk_max[0] + 1),
        indexing='ij'
    )

    # 将IJK坐标转换为RAS坐标
    ijk_coords = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    ras_coords = batch_ijk_to_ras(ijk_coords, ijkToRAS)
    print(np.floor(ras_coords.min(axis=0)), np.ceil(ras_coords.max(axis=0)))


    # 使用优化的points_in_cylinder函数
    in_cylinder = points_in_cylinder_optimized(
        ras_coords, base_center, average_direction_unit, cylinder_height, base_radius
    )

    # 将结果重塑为局部数组形状
    local_shape = np.flip(local_shape)
    local_array = in_cylinder.reshape(local_shape)

    # 创建全局数组
    main_geometry_array = np.zeros(volume_shape, dtype=np.uint8)

    # 将局部数组放回全局数组
    main_geometry_array[
    ijk_min[2]:ijk_max[2] + 1,
    ijk_min[1]:ijk_max[1] + 1,
    ijk_min[0]:ijk_max[0] + 1
    ] = local_array


    return main_geometry_array




@numba.njit(parallel=True)
def points_in_cylinder(ras_coords, base_center, average_direction_unit, cylinder_height, base_radius):
    n = ras_coords.shape[0]
    in_cyl = np.zeros(n, dtype=np.uint8)
    for i in numba.prange(n):
        vec_x = ras_coords[i, 0] - base_center[0]
        vec_y = ras_coords[i, 1] - base_center[1]
        vec_z = ras_coords[i, 2] - base_center[2]

        # projection_length = vec · average_direction_unit
        projection_length = vec_x * average_direction_unit[0] + vec_y * average_direction_unit[1] + vec_z * \
                            average_direction_unit[2]

        if 0 <= projection_length <= cylinder_height:
            # 计算投影点坐标
            proj_x = base_center[0] + projection_length * average_direction_unit[0]
            proj_y = base_center[1] + projection_length * average_direction_unit[1]
            proj_z = base_center[2] + projection_length * average_direction_unit[2]

            dx = ras_coords[i, 0] - proj_x
            dy = ras_coords[i, 1] - proj_y
            dz = ras_coords[i, 2] - proj_z
            radial_distance = np.sqrt(dx * dx + dy * dy + dz * dz)
            if radial_distance <= base_radius:
                in_cyl[i] = 1
    return in_cyl




def create_cylinder(superVolumeShape, base_radius_range, base_center, base_radius, average_direction_unit,
                    cylinder_height, ijkToRAS, rasToIJKMatrix, substrateSmallCylinder=True, small_cylinder_radius=1.0,
                    all_end_points=None, direction_vectors=None):
    # 获取主圆柱体的包围盒（世界坐标系）
    cylinder_points = np.array([
        base_center + average_direction_unit * t + np.array([dx, dy, dz])
        for t in [-cylinder_height, cylinder_height]
        for dx in [-25.0 * base_radius_range, 25.0 * base_radius_range]
        for dy in [-25.0 * base_radius_range, 25.0 * base_radius_range]
        for dz in [-25.0 * base_radius_range, 25.0 * base_radius_range]
    ])

    # 将包围盒顶点转换为 IJK 坐标
    ijk_points = np.array([ras_to_ijk(point, rasToIJKMatrix) for point in cylinder_points])

    # 获取包围盒的最小和最大索引
    ijk_min = np.floor(ijk_points.min(axis=0)).astype(int)
    ijk_max = np.ceil(ijk_points.max(axis=0)).astype(int)

    print('min:', ijk_min, 'max:', ijk_max)

    # 确保索引在有效范围内
    volume_shape = superVolumeShape  # (Z, Y, X)
    ijk_min = np.maximum(ijk_min, [0, 0, 0])
    ijk_max = np.minimum(ijk_max, np.array(volume_shape) - 1)

    # 计算局部数组的形状
    local_shape = ijk_max - ijk_min + 1  # (Z_size, Y_size, X_size)

    # 创建局部数组
    local_array = np.zeros(local_shape, dtype=np.uint8)

    Z, Y, X = np.meshgrid(
        np.arange(ijk_min[0], ijk_max[0] + 1),
        np.arange(ijk_min[1], ijk_max[1] + 1),
        np.arange(ijk_min[2], ijk_max[2] + 1),
        indexing='ij'
    )

    # 将 IJK 坐标转换为 RAS 坐标
    ijk_coords = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    ras_coords = batch_ijk_to_ras(ijk_coords, ijkToRAS)
    print(np.floor(ras_coords.min(axis=0)), np.ceil(ras_coords.max(axis=0)))

    # 使用加速过的points_in_cylinder函数
    in_cylinder = points_in_cylinder(ras_coords, base_center, average_direction_unit, cylinder_height, base_radius)

    # 将结果映射回局部数组
    local_array_flat = in_cylinder.astype(np.uint8)
    local_array = local_array_flat.reshape(local_shape)

    if substrateSmallCylinder:

        all_norm_directions = direction_vectors / np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]
        all_far_points = all_end_points + 5 * cylinder_height * all_norm_directions
        all_near_points = all_end_points - 2 * cylinder_height * all_norm_directions

        for start_point, end_point in zip(all_near_points, all_far_points):
            # 计算轴线方向和长度
            direction_vector = end_point - start_point
            length = np.linalg.norm(direction_vector)
            if length == 0:
                continue
            axis_direction = direction_vector / length

            # 计算小圆柱体的顶点坐标
            cylinder_points = np.array([
                start_point + axis_direction * t + np.array([dx, dy, dz])
                for t in [0, length]
                for dx in [-15 * small_cylinder_radius, 15 * small_cylinder_radius]
                for dy in [-15 * small_cylinder_radius, 15 * small_cylinder_radius]
                for dz in [-15 * small_cylinder_radius, 15 * small_cylinder_radius]
            ])

            # 将包围盒顶点转换为 IJK 坐标
            ijk_points = np.array([ras_to_ijk(point, rasToIJKMatrix) for point in cylinder_points])

            # 获取包围盒的最小和最大索引
            ijk_min_cyl = np.floor(ijk_points.min(axis=0)).astype(int)
            ijk_max_cyl = np.ceil(ijk_points.max(axis=0)).astype(int)

            # 确保索引在有效范围内
            ijk_min_cyl = np.maximum(ijk_min_cyl, ijk_min)
            ijk_max_cyl = np.minimum(ijk_max_cyl, ijk_max)

            # 如果没有重叠，跳过
            if np.any(ijk_min_cyl > ijk_max_cyl):
                continue

            # 计算局部索引范围相对于主局部数组的偏移
            offset = ijk_min_cyl - ijk_min
            local_cyl_shape = ijk_max_cyl - ijk_min_cyl + 1

            # 获取局部坐标
            Z_cyl, Y_cyl, X_cyl = np.meshgrid(
                np.arange(ijk_min_cyl[0], ijk_max_cyl[0] + 1),
                np.arange(ijk_min_cyl[1], ijk_max_cyl[1] + 1),
                np.arange(ijk_min_cyl[2], ijk_max_cyl[2] + 1),
                indexing='ij'
            )
            ijk_coords_cyl = np.column_stack((X_cyl.ravel(), Y_cyl.ravel(), Z_cyl.ravel()))
            ras_coords_cyl = batch_ijk_to_ras(ijk_coords_cyl, ijkToRAS)

            # 使用加速过的points_in_cylinder函数计算小圆柱
            in_cylinder = points_in_cylinder(ras_coords_cyl, start_point, axis_direction, length, small_cylinder_radius)
            small_cylinder_array_flat = in_cylinder.astype(np.uint8)
            small_cylinder_array = small_cylinder_array_flat.reshape(local_cyl_shape)

            # 从主局部数组中减去小圆柱体
            z_slice = slice(offset[0], offset[0] + local_cyl_shape[0])
            y_slice = slice(offset[1], offset[1] + local_cyl_shape[1])
            x_slice = slice(offset[2], offset[2] + local_cyl_shape[2])
            local_array[z_slice, y_slice, x_slice] = np.where(
                small_cylinder_array == 1, 0, local_array[z_slice, y_slice, x_slice]
            )

    # 创建全局数组
    main_geometry_array = np.zeros(volume_shape, dtype=np.uint8)

    # 将局部数组放回全局数组
    main_geometry_array[ijk_min[0]:ijk_max[0] + 1, ijk_min[1]:ijk_max[1] + 1, ijk_min[2]:ijk_max[2] + 1] = local_array

    return main_geometry_array








########################################################GuidePlateDesign########################################################







#####################################################Optimization#####################################################


def region_image_to_boundary_image(image, show=False):
    # structure_1 = np.ones((5, 5), dtype=int)
    structure_2 = np.ones((3, 3), dtype=int)
    structure_3 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])
    # eroded_image_1 = binary_erosion(image, structure=structure_1)
    # eroded_image_2 = binary_erosion(image, structure=structure_2)
    dilated_image = binary_dilation(image, structure=structure_3, iterations=2)
    boundary_image = dilated_image
    
    indices = np.argwhere(image == 1)
    centroid = np.mean(indices, axis=0)
    
    distance_map = distance_transform_edt(1 - boundary_image)
    # print(np.argwhere(distance_map == 0))
    distance_map = 3 * np.log(distance_map + 1)
    
    sum_area = np.sum(image)
    
    if show:
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 3, 1)
        plt.title('original region image at z_crown')
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.title('processed region image')
        plt.imshow(boundary_image, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title('distance map')
        plt.imshow(distance_map, cmap='gray')
        plt.show()
    
    return boundary_image, centroid, distance_map, sum_area


def bilinear_interpolate(image, point):
    """
    Bilinear interpolation at a point in an image.
    
    Args:
    image: (H, W) Tensor
    point: (2,) Tensor

    """
    height, width = image.shape
    x, y = point  

    x = min(max(x, 0), height - 1)
    y = min(max(y, 0), width - 1)

    x0, x1 = int(x), min(int(x) + 1, height - 1)
    y0, y1 = int(y), min(int(y) + 1, width - 1)

    xa, ya = x - x0, y - y0
    xb, yb = 1 - xa, 1 - ya

    Q11 = image[x0, y0]
    Q21 = image[x1, y0]
    Q12 = image[x0, y1]
    Q22 = image[x1, y1]

    return (Q11 * xb * yb + Q21 * xa * yb + Q12 * xb * ya + Q22 * xa * ya)


def objective_function(data, points, points_initial, map_centroid, sum_area, 
                       project_points, show=False,
                       term_contour_coeff=10.0,
                       term_regularization_1_coeff=0.5,
                       term_regularization_2_coeff=2.0,
                       term_distance_keepfar_coeff=15.0,
                       term_mutual_distance_intensity=2.0,
                       term_distance_keepuniform_coeff=6.0,
                       term_center_coeff=2.0):
    term_contour = 0
    term_regularization_1 = 0
    term_regularization_2 = 0
    # term_depth_consistent = 0
    term_distance_keepfar = 0
    term_distance_keepuniform = 0
    term_center = 0
    
    for x in points:
        # x_normalized = ((x / (torch.tensor(data.shape[2:]).float() - 1)) * 2.0 - 1.0)
        # x_normalized = x_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
        # interpolated_values = F.grid_sample(data, x_normalized, mode='bilinear', padding_mode='border', align_corners=True)
        interpolated_values = bilinear_interpolate(data, x)
        term_contour += interpolated_values.sum()
        if show:
            print('interpolated map values: ', interpolated_values.sum())
    term_contour = term_contour / points.shape[0]  
    
    # depth_tensor = torch.tensor([[0.0], [0.0], [1.0]], requires_grad=False)
    # position_tensor = torch.tensor([[1.0], [1.0], [0.0]], requires_grad=False)
    # print(torch.matmul(points - points_initial, depth_tensor), depth_tensor.shape)
    term_regularization_1 = (points - points_initial).pow(2).sum(dim=1).mean()     
    # tmp_points = points.clone().unsqueeze(1)
    # print('tmp_points:', tmp_points, tmp_points.shape, root_canal_near_points.shape)   
    # term_regularization_2 = (tmp_points - root_canal_near_points).pow(2).sum(dim=2).mean()    
    # print((tmp_points - root_canal_near_points).pow(2).sum(dim=2).shape)
    
    # z_crown_tensor = torch.tensor([z_crown], requires_grad=False, dtype=torch.float32).unsqueeze(0)
    # z_crown_tensor = torch.concat([z_crown_tensor] * points.shape[0], dim=0)
    # tmp_points_3d = torch.concat([points, z_crown_tensor], dim=1)
    # path_direction = root_canal_near_points - tmp_points_3d
    # path_direction = path_direction / torch.norm(path_direction, dim=1, keepdim=True, p=2)
    # for d1, d2 in zip(path_direction, root_canal_central_direction):
    #     # term_regularization_2 += 2 - (torch.dot(d1, d2) + 1)
    #     term_regularization_2 += (d1 - d2).pow(2).sum()
    #     if show:
    #         print('d1:', d1, d1.shape, '\nd2:', d2, d2.shape, '\ndiffs:', (d1 - d2).pow(2).sum())
    # term_regularization_2 = term_regularization_2 / points.shape[0]
    term_regularization_2 = (points - project_points).pow(2).sum(dim=1).mean()
    
    # term_depth_consistent = torch.matmul(points - points_initial, depth_tensor).pow(2).sum(dim=1).mean()  
    
    if points.shape[0] > 1:
        points_pair_distance = torch.pdist(points, p=2)
        # term_distance_keepfar =  - points_pair_distance.mean()    # p=2 表示欧氏距离
        sum_area = torch.tensor(float(sum_area), requires_grad=False)
        sum_area_coeff = term_mutual_distance_intensity * term_mutual_distance_intensity
        sum_area_coeff = torch.tensor(float(sum_area_coeff), requires_grad=False)

        f_points_pair_distance = points_pair_distance + sum_area_coeff / points_pair_distance - (2 * torch.sqrt(sum_area_coeff) - 2)
        term_distance_keepfar = f_points_pair_distance.sum() / (f_points_pair_distance.shape[0] + 1e-6)                # min: 2
        # print('f_points_pair_distance: ', f_points_pair_distance, f_points_pair_distance.shape)
        points_pair_distance_pair_distance = torch.pdist(points_pair_distance.unsqueeze(1), p=2)
        term_distance_keepuniform = points_pair_distance_pair_distance.sum() / (points_pair_distance_pair_distance.shape[0] + 1e-6)
        # print('points_pair_distance_pair_distance: ', points_pair_distance_pair_distance, points_pair_distance_pair_distance.shape)
        if show:
            print('points_pair_distance: ', points_pair_distance)
            print('points_pair_distance_pair_distance: ', torch.pdist(points_pair_distance.unsqueeze(1), p=2))
    elif points.shape[0] == 1:
        term_distance_keepfar = 0
        term_distance_keepuniform = 0
    else:
        raise ValueError('points shape error!')
    
    
    term_center = (torch.mean(points, dim=0) - map_centroid).pow(2).sum()
    if show:
        print('term_contour:', term_contour, 
                '\nterm_regularization_1(initial_points):', term_regularization_1, 
                '\nterm_regularization_2(project_points):', term_regularization_2,
                '\nterm_distance_keepfar:', term_distance_keepfar, 
                '\nterm_distance_keepuniform:', term_distance_keepuniform, 
                '\nterm_center:', term_center)
    
    total_loss = term_contour_coeff * term_contour + term_regularization_1_coeff * term_regularization_1 + term_regularization_2_coeff * term_regularization_2 + \
                term_distance_keepfar_coeff * term_distance_keepfar + term_distance_keepuniform_coeff * term_distance_keepuniform + term_center_coeff * term_center
    
    return total_loss



def find_single_connected_component_slices(volume):
    indices_with_single_component = []
    max_area = 0
    
    for z in range(volume.shape[2]):
        slice_2d = volume[:, :, z]
        if np.sum(slice_2d) > max_area:
            max_area = np.sum(slice_2d)
        sitk_image = sitk.GetImageFromArray(slice_2d.astype(np.int16))
        connected_components = sitk.ConnectedComponent(sitk_image)
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(connected_components)
        
        # print(label_stats.GetNumberOfLabels())
        if label_stats.GetNumberOfLabels() == 1:
            indices_with_single_component.append(z)
    
    indices_with_single_component = np.array(indices_with_single_component, dtype=int)
    
    return indices_with_single_component, max_area


def find_groups_and_average_indices(indices):
    groups = []
    current_group = []
    last_index = None
    
    for index in indices:
        if last_index is None or index == last_index + 1:
            current_group.append(index)
        else:
            current_group = np.array(current_group, dtype=int)
            groups.append(current_group)
            current_group = [index]
        last_index = index
    
    if current_group:
        current_group = np.array(current_group, dtype=int)
        groups.append(current_group)
    
    
    average_indices = [np.mean(group) for group in groups]
    average_indices = np.array(average_indices, dtype=float)
    
    return groups, average_indices



class JsonPointsProcesser:
    def __init__(self, folder_path, start_at_root=False, z_crown_index=0, z_pulp_index=None):
        self.folder_path = folder_path
        self.all_data_list = os.listdir(self.folder_path)
        self.volume_list = [v for v in self.all_data_list if v.endswith('.nii.gz')]
        assert(len(self.volume_list) == 2)
        for volume_name in self.volume_list:
            if 'Seg' in volume_name or 'seg' in volume_name:
                self.label_name = volume_name
            else:
                self.ctimg_name = volume_name
        assert(self.ctimg_name is not None and self.label_name is not None and self.ctimg_name != self.label_name)
        self.json_list = [j for j in self.all_data_list if j.endswith('.json')]
        self.root_canal_num = len(self.json_list)
        
        self.start_at_root = start_at_root
        
        self.affine_matrix = nib.load(os.path.join(self.folder_path, self.ctimg_name)).affine
        self.affine_matrix_inv = np.linalg.inv(self.affine_matrix)
               
        self.all_points_world = []    # n * 3 * 3
        self.all_points_index = []    # n * 3 * 3
        self.datas = []              # len = n
        self.load_data()
        
        self.matrix_orientation = np.array(self.datas[0]["markups"][0]["controlPoints"][0]["orientation"], dtype=float).reshape(3, 3)
        for data in self.datas:
            for points_dict in data["markups"][0]["controlPoints"]:
                assert np.allclose(np.array(points_dict["orientation"], dtype=float).reshape(3, 3), self.matrix_orientation)
        
        self.all_2nd_order_control_points_world = []     # n * 3 * 3
        self.all_2nd_order_control_points_index = []     # n * 3 * 3
        self.all_initial_tangent_world = []        # n * 3
        self.all_initial_tangent_index = []        # n * 3
        self.all_central_tangent_world = []        # n * 3
        self.all_central_tangent_index = []        # n * 3      
        
        self.z_crown_index = float(z_crown_index)
        self.z_crown_world = index_coordinate_to_world_coordinate(self.affine_matrix, 
                                                                  self.matrix_orientation, 
                                                                  np.array([0, 0, self.z_crown_index], dtype=float))[2]
        # print(self.z_crown_world)
        
        self.crown_points_index = []        # n * 3
        self.crown_points_world = []        # n * 3     
        
        self.all_line_paths_origin_world = []    # n * 2 * 3
        self.all_line_paths_origin_index = []    # n * 2 * 3
        self.mean_root_direction_world = None         # (3, )
        self.mean_root_direction_index = None         # (3, )
        self.calc_bezier_control_points_tangent_and_crown_points()   
        self.datas_line_paths_origin = self.save_data(self.all_line_paths_origin_world, 
                                                      'line_paths_origin', 
                                                      color=[1.0, 0.5, 0.5],
                                                      return_json_dict=True)
        
        # print("self.crown_points_world: \n", self.crown_points_world)
        # print("self.crown_points_index: \n", self.crown_points_index)
       
        self.region_image_origin = None
        self.region_image_z_index = None
        self.region_image_at_z_crown = None
        self.volume_height = None
        
        tmp_z_pulp_index = self.get_z_pulp_index()
        if z_pulp_index is not None:
            self.region_image_z_index = float(z_pulp_index)
        else:
            self.region_image_z_index = float(tmp_z_pulp_index)
        print('calculated z_pulp_index: ', tmp_z_pulp_index)
        self.segment_volume_new = None
        self.get_region_image_at_z_crown(z_pulp_index=self.region_image_z_index)
        
            
        self.crown_points_index_optimized = []       # n * 3
        self.crown_points_world_optimized = []      # n * 3
        self.all_line_paths_optimized_world = []        # n * 2 * 3
        self.all_line_paths_optimized_index = []        # n * 2 * 3
        self.datas_line_paths_optimized = None
        self.optimized = False
        
        
    
    def load_data(self):
        for json_name in self.json_list:
            with open(os.path.join(self.folder_path, json_name), 'r') as f:
                data = json.load(f)
                self.datas.append(data)
                
                points_world = []
                points_index = []
                
                if not self.start_at_root:
                    for points_dict in data["markups"][0]["controlPoints"]:
                        points_world.append(np.array(points_dict["position"], dtype=float))
                        matrix_orientation = np.array(points_dict["orientation"], dtype=float).reshape(3, 3)
                        points_index.append(world_coordinate_to_index_coordinate(self.affine_matrix, 
                                                                                 matrix_orientation, 
                                                                                 np.array(points_dict["position"], dtype=float)))
                        
                else:
                    for points_dict in data["markups"][0]["controlPoints"][::-1]:
                        points_world.append(np.array(points_dict["position"], dtype=float))
                        matrix_orientation = np.array(points_dict["orientation"], dtype=float).reshape(3, 3)
                        points_index.append(world_coordinate_to_index_coordinate(self.affine_matrix, 
                                                                                 matrix_orientation, 
                                                                                 np.array(points_dict["position"], dtype=float)))
                    
                points_world = np.array(points_world, dtype=float)
                points_index = np.array(points_index, dtype=float)
                
                self.all_points_world.append(points_world)
                self.all_points_index.append(points_index)
        
        self.all_points_world = np.array(self.all_points_world, dtype=float)
        self.all_points_index = np.array(self.all_points_index, dtype=float)
        # print(self.all_points_world)
        # print(len(self.all_points_world))
        # print(self.all_points_index)
        # print(self.all_points_index.shape)
        # print(len(self.all_points_index))   
    
    
    def calc_bezier_control_points_tangent_and_crown_points(self):
        for point_list in self.all_points_world:
            assert len(point_list) == 3
            q0 = point_list[0, :]
            q1 = point_list[1, :]
            q2 = point_list[2, :]
            # print(q0, q1, q2)
            
            p0 = q0
            p2 = q2
            p1 = 2 * q1 - 0.5 * (q0 + q2)
            control_points_world = np.array([p0, p1, p2], dtype=float)
            control_points_index = np.array([world_coordinate_to_index_coordinate(self.affine_matrix, self.matrix_orientation, p0),
                                             world_coordinate_to_index_coordinate(self.affine_matrix, self.matrix_orientation, p1),
                                             world_coordinate_to_index_coordinate(self.affine_matrix, self.matrix_orientation, p2)], dtype=float)
            self.all_2nd_order_control_points_world.append(control_points_world)
            self.all_2nd_order_control_points_index.append(control_points_index)
            # print(control_points_world)
            # print(control_points_index)
            
            tangent_at_start_world = 2 * (p1 - p0)   # world coordinate
            tangent_at_start_index = 2 * (world_coordinate_to_index_coordinate(self.affine_matrix, self.matrix_orientation, p1) \
                - world_coordinate_to_index_coordinate(self.affine_matrix, self.matrix_orientation, p0))
            tangent_at_start_world = tangent_at_start_world / np.linalg.norm(tangent_at_start_world, ord=2)
            self.all_initial_tangent_world.append(tangent_at_start_world)
            tangent_at_start_index = tangent_at_start_index / np.linalg.norm(tangent_at_start_index, ord=2)
            self.all_initial_tangent_index.append(tangent_at_start_index)
            # print(tangent_at_start_world)
            # print(tangent_at_start_index)
 
            tangent_at_mid_world = p2 - p0    # world coordinate
            tangent_at_mid_index = world_coordinate_to_index_coordinate(self.affine_matrix, self.matrix_orientation, p2) \
                - world_coordinate_to_index_coordinate(self.affine_matrix, self.matrix_orientation, p0)
            tangent_at_mid_world = tangent_at_mid_world / np.linalg.norm(tangent_at_mid_world, ord=2)
            self.all_central_tangent_world.append(tangent_at_mid_world)
            tangent_at_mid_index = tangent_at_mid_index / np.linalg.norm(tangent_at_mid_index, ord=2)
            self.all_central_tangent_index.append(tangent_at_mid_index)
            # print(tangent_at_mid_world)
            # print(tangent_at_mid_index)
            
            x = (self.z_crown_world - q0[2]) / tangent_at_start_world[2]
            crown_point_world = q0 + x * tangent_at_start_world
            crown_point_index = world_coordinate_to_index_coordinate(self.affine_matrix, 
                                                                     self.matrix_orientation, 
                                                                     crown_point_world)
            # print(crown_point_world)
            # print(crown_point_index)
            self.crown_points_world.append(crown_point_world)
            self.crown_points_index.append(crown_point_index)
            
            self.all_line_paths_origin_world.append(np.array([q0, crown_point_world], dtype=float))
            self.all_line_paths_origin_index.append(np.array([world_coordinate_to_index_coordinate(self.affine_matrix, self.matrix_orientation, q0), 
                                                       crown_point_index], dtype=float))
        
        self.all_2nd_order_control_points_world = np.array(self.all_2nd_order_control_points_world, dtype=float)
        self.all_2nd_order_control_points_index = np.array(self.all_2nd_order_control_points_index, dtype=float)
        self.all_initial_tangent_world = np.array(self.all_initial_tangent_world, dtype=float)
        self.all_initial_tangent_index = np.array(self.all_initial_tangent_index, dtype=float)
        self.all_central_tangent_world = np.array(self.all_central_tangent_world, dtype=float)
        self.all_central_tangent_index = np.array(self.all_central_tangent_index, dtype=float)
        self.crown_points_world = np.array(self.crown_points_world, dtype=float)
        self.crown_points_index = np.array(self.crown_points_index, dtype=float)
        self.all_line_paths_origin_world = np.array(self.all_line_paths_origin_world, dtype=float)
        self.all_line_paths_origin_index = np.array(self.all_line_paths_origin_index, dtype=float)
        # print(self.all_line_paths_origin_world)
        # print(self.all_line_paths_origin_index)
        
        mean_root_direction_world = np.mean(self.all_central_tangent_world, axis=0)
        mean_root_direction_world = mean_root_direction_world / np.linalg.norm(mean_root_direction_world, ord=2)
        self.mean_root_direction_world = mean_root_direction_world
        mean_root_direction_index = np.mean(self.all_central_tangent_index, axis=0)
        mean_root_direction_index = mean_root_direction_index / np.linalg.norm(mean_root_direction_index, ord=2)
        self.mean_root_direction_index = mean_root_direction_index
        # print(mean_root_direction_world)
        # print(mean_root_direction_index)
        
            
    
    def save_data(self, points_world, json_name, color=[1.0, 0.4, 1.0], return_json_dict=False):
        assert(len(points_world.shape) <= 3)
        if len(points_world.shape) == 2:
            points_world_tmp = points_world[np.newaxis, :, :]
        elif len(points_world.shape) == 1:
            points_world_tmp = points_world[np.newaxis, np.newaxis, :]
        else:
            points_world_tmp = points_world
        assert(len(points_world_tmp.shape) == 3)
        
        data = copy.deepcopy(self.datas)
        
        for i, line_world in enumerate(points_world_tmp):
            # print(i)
            control_point_template = data[i]["markups"][0]["controlPoints"][0]
            data[i]["markups"][0]["controlPoints"] = []
            data[i]["markups"][0]["display"]["color"] = list(1.0 - np.array(color))
            data[i]["markups"][0]["display"]["activeColor"] = list(1.0 - np.array(color))
            data[i]["markups"][0]["display"]["selectedColor"] = color
            # print(control_point_template)
            
            for j, point_world in enumerate(line_world):
                control_point_dict = copy.deepcopy(control_point_template)
                control_point_dict["position"] = list(point_world)
                control_point_dict["id"] = str(j + 1)
                control_point_dict["label"] = control_point_template["label"].split('-')[0] + '-' + json_name + '-' + str(j + 1)
                # print(list(point_world))
                # print(control_point_dict)
                
                data[i]["markups"][0]["controlPoints"].append(control_point_dict)
            data[i]["markups"][0]["lastUsedControlPointNumber"] = len(data[i]["markups"][0]["controlPoints"])
            
            os.makedirs(self.folder_path + ' - Out Json', exist_ok=True)
            json.dump(data[i], open(os.path.join(self.folder_path + ' - Out Json', json_name + "_" + str(i+1) + '.json'), 'w'), indent=4)
            
        if return_json_dict:
            return data
    
    
    def get_region_image_at_z_crown(self, z_pulp_index=None):
        if z_pulp_index is not None:
            z_pulp_index_tmp = round(z_pulp_index)
        else:
            assert(z_pulp_index is not None)
            pass
        
        segment_volume = nib.load(os.path.join(self.folder_path, self.label_name)).get_fdata()
        self.volume_height = segment_volume.shape[2]
        # print(self.volume_height)
        self.region_image_origin = segment_volume[:, :, z_pulp_index_tmp]
        
        indexes = np.argwhere(self.region_image_origin == 1)
        # print(indexes)
        
        x = (self.z_crown_index - z_pulp_index_tmp) / self.mean_root_direction_index[2]
        new_indexes = indexes + x * np.array([self.mean_root_direction_index[:2]])
        # print(np.array([self.mean_root_direction_index[:2]]).shape)
        new_indexes = np.rint(new_indexes).astype(int)
        # print(new_indexes)
        
        self.region_image_at_z_crown = np.zeros_like(self.region_image_origin)
        for new_index in new_indexes:
            if new_index[0] >= 0 and new_index[0] < self.region_image_at_z_crown.shape[0] and \
                new_index[1] >= 0 and new_index[1] < self.region_image_at_z_crown.shape[1]:
                    self.region_image_at_z_crown[new_index[0], new_index[1]] = 1
        # self.region_image_at_z_crown[new_indexes[:, 0], new_indexes[:, 1]] = 1
        
        self.segment_volume_new = np.zeros_like(segment_volume)
        self.segment_volume_new[:, :, round(self.z_crown_index)] = self.region_image_at_z_crown
        self.segment_volume_new[:, :, z_pulp_index_tmp] = self.region_image_origin
        nib.save(nib.Nifti1Image(self.segment_volume_new, self.affine_matrix), 
                 os.path.join(self.folder_path + ' - Out Json', 'pulp_region_at_z_crown.nii.gz'))
    
    
    def update_optimized_results(self, crown_points_2d_index_optimized, show=False):
        
        for i, crown_point_2d_index_optimized in enumerate(crown_points_2d_index_optimized):
            tmp_index_optimized = np.array([crown_point_2d_index_optimized[0], crown_point_2d_index_optimized[1], self.z_crown_index], dtype=float)
            tmp_world_optimized = index_coordinate_to_world_coordinate(self.affine_matrix, self.matrix_orientation, tmp_index_optimized)
            
            self.crown_points_index_optimized.append(tmp_index_optimized)
            self.crown_points_world_optimized.append(tmp_world_optimized)
            
            self.all_line_paths_optimized_world.append(np.array([self.all_points_world[i][0], tmp_world_optimized], dtype=float))
            self.all_line_paths_optimized_index.append(np.array([self.all_points_index[i][0], tmp_index_optimized], dtype=float))
        
        self.crown_points_index_optimized = np.array(self.crown_points_index_optimized, dtype=float)
        self.crown_points_world_optimized = np.array(self.crown_points_world_optimized, dtype=float)
        self.all_line_paths_optimized_world = np.array(self.all_line_paths_optimized_world, dtype=float)
        self.all_line_paths_optimized_index = np.array(self.all_line_paths_optimized_index, dtype=float)
        
        self.datas_line_paths_optimized = self.save_data(self.all_line_paths_optimized_world, 
                                                                        'line_paths_optimized', 
                                                                        color=[0.5, 1.0, 0.5],
                                                                        return_json_dict=True)
        self.optimized = True
        
        if show:
            
            print('crown_points_index: \n', self.crown_points_index)
            print('crown_points_index_optimized: \n', self.crown_points_index_optimized)
            print('crown_points_world: \n', self.crown_points_world)
            print('crown_points_world_optimized: \n', self.crown_points_world_optimized)
            
            plt.figure(figsize=(8, 8))
            plt.subplot(2, 2, 1)
            plt.title('original region image at z_pulp')
            plt.imshow(self.region_image_origin, cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title('calculated region image at z_crown')
            plt.imshow(self.region_image_at_z_crown, cmap='gray')
            
            tmp_rgb_img_origin = np.repeat(self.region_image_at_z_crown[:, :, np.newaxis], 3, axis=2)
            tmp_rgb_img_optimized = np.repeat(self.region_image_at_z_crown[:, :, np.newaxis], 3, axis=2)
            
            for crown_point_index_origin in self.crown_points_index:
                try: 
                    tmp_rgb_img_origin[int(crown_point_index_origin[0]), int(crown_point_index_origin[1]), 0] = 1.0
                    tmp_rgb_img_origin[int(crown_point_index_origin[0]), int(crown_point_index_origin[1]), 1] = 0.0
                    tmp_rgb_img_origin[int(crown_point_index_origin[0]), int(crown_point_index_origin[1]), 2] = 0.0
                except:
                    pass
            
            for crwon_point_index_optimized in self.crown_points_index_optimized:
                tmp_rgb_img_optimized[int(crwon_point_index_optimized[0]), int(crwon_point_index_optimized[1]), 0] = 0.0
                tmp_rgb_img_optimized[int(crwon_point_index_optimized[0]), int(crwon_point_index_optimized[1]), 1] = 1.0
                tmp_rgb_img_optimized[int(crwon_point_index_optimized[0]), int(crwon_point_index_optimized[1]), 2] = 0.0
            
            plt.subplot(2, 2, 3)
            plt.title('original crown points')
            plt.imshow(tmp_rgb_img_origin)
            plt.subplot(2, 2, 4)
            plt.title('optimized crown points')
            plt.imshow(tmp_rgb_img_optimized)
            plt.show()
    
    
    def get_z_pulp_index(self):
        
        segment_volume = nib.load(os.path.join(self.folder_path, self.label_name)).get_fdata()
        ct_volume = nib.load(os.path.join(self.folder_path, self.ctimg_name)).get_fdata()
        assert segment_volume.shape == ct_volume.shape
        
        indices_with_single_component, max_area = find_single_connected_component_slices(segment_volume)
        groups, average_indices = find_groups_and_average_indices(indices_with_single_component)
        
        # plt.figure(figsize=(8, 8))
        # plt.imshow(segment_volume[:, :, 0], cmap='gray')
        # plt.show()
        
        # print('groups:', groups)
        # print('average_indices:', average_indices)

        assert(np.all(average_indices[:-1] <= average_indices[1:]))
        # print('max_area:', max_area)
        z_pulp_index = None
        area_threshold = 0.7 * max_area
        # max_area_of_average_indices = np.max([np.sum(segment_volume[:, :, round(average_index)]) for average_index in average_indices])
        arg_max_area_of_average_indices = np.argmax([np.sum(segment_volume[:, :, round(average_index)]) for average_index in average_indices])
        # print('max_area_of_average_indices:', max_area_of_average_indices)
        # print('arg_max_area_of_average_indices:', arg_max_area_of_average_indices)
        len_max_group = len(groups[arg_max_area_of_average_indices])
        offset_max_group = len_max_group // 3
        # print('len_max_group:', len_max_group)
        
        if self.mean_root_direction_index[2] > 0:    # root index > crown index
            for group, average_index in zip(groups, average_indices):
                if np.sum(segment_volume[:, :, round(average_index)]) > area_threshold and len(group) >= 1:
                    z_pulp_index = round(average_index)
                    # print('selected area: ', np.sum(segment_volume[:, :, z_pulp_index]))
                    break
            if z_pulp_index is None:
                avg_idx_of_max_area = round(average_indices[arg_max_area_of_average_indices])
                for i in range(1, offset_max_group):
                    current_index = avg_idx_of_max_area + i
                    # print('area at current_index:', np.sum(segment_volume[:, :, current_index]), 'current_index:', current_index)
                    if np.sum(segment_volume[:, :, current_index]) > area_threshold:
                        z_pulp_index = current_index
                        break
                    current_index = avg_idx_of_max_area - i
                    # print('area at current_index:', np.sum(segment_volume[:, :, current_index]), 'current_index:', current_index)
                    if np.sum(segment_volume[:, :, current_index]) > area_threshold:
                        z_pulp_index = current_index
                        break
                else:
                    # z_pulp_index = round(average_indices[0])
                    for group, average_index in zip(groups, average_indices):
                        if len(group) > 1:
                            z_pulp_index = round(average_index)
                            break
                # print('selected area: ', np.sum(segment_volume[:, :, z_pulp_index]))
        else:   # root index < crown index
            for group, average_index in zip(groups[::-1], average_indices[::-1]):
                if np.sum(segment_volume[:, :, round(average_index)]) > area_threshold and len(group) >= 1:
                    z_pulp_index = round(average_index)
                    # print('selected area: ', np.sum(segment_volume[:, :, z_pulp_index]))
                    break
            if z_pulp_index is None:
                avg_idx_of_max_area = round(average_indices[arg_max_area_of_average_indices])
                for i in range(1, offset_max_group):
                    current_index = avg_idx_of_max_area + i
                    # print('area at current_index:', np.sum(segment_volume[:, :, current_index]), 'current_index:', current_index)
                    if np.sum(segment_volume[:, :, current_index]) > area_threshold:
                        z_pulp_index = current_index
                        break
                    current_index = avg_idx_of_max_area - i
                    # print('area at current_index:', np.sum(segment_volume[:, :, current_index]), 'current_index:', current_index)
                    if np.sum(segment_volume[:, :, current_index]) > area_threshold:
                        z_pulp_index = current_index
                        break
                else:
                    # z_pulp_index = round(average_indices[-1])
                    for group, average_index in zip(groups[::-1], average_indices[::-1]):
                        if len(group) > 1:
                            z_pulp_index = round(average_index)
                            break
                # print('selected area: ', np.sum(segment_volume[:, :, z_pulp_index]))
        
        # for average_index in average_indices:
        #     print('area at average_index:', np.sum(segment_volume[:, :, round(average_index)]))    
        # print(area_threshold)
        
        return z_pulp_index




#####################################################Optimization#####################################################






#####################################################decode#####################################################

dilation_structure = np.ones((2, 2, 2), dtype=bool)

def compute_unit_vector(start_point, end_point):
    """
    Compute the unit vector from start_point to end_point.

    Parameters:
    - start_point: numpy ndarray, the starting point coordinates.
    - end_point: numpy ndarray, the ending point coordinates.

    Returns:
    - unit_vector: numpy ndarray, the unit vector from start to end point.
    """
    # Ensure inputs are numpy arrays
    if not isinstance(start_point, np.ndarray):
        start_point = np.array(start_point)
    if not isinstance(end_point, np.ndarray):
        end_point = np.array(end_point)
    
    # Compute the difference vector (end_point - start_point)
    diff_vector = end_point - start_point
    
    # Compute the norm (length) of the difference vector
    norm = np.linalg.norm(diff_vector, ord=2)
    
    if norm == 0:
        raise ValueError("The start point and end point cannot be the same.")
    
    # Compute the unit vector
    unit_vector = diff_vector / norm
    
    return unit_vector


def find_top_k_local_maxima(heatmap, k=5, threshold=0.3, direction=None, current_point=None):
    
    """
    Decode heatmap to get top k local maxima points and values
    Args:
        output_hm: (H, W, D) tensor, 3D heatmap
        k: int, number of local maxima to return
        threshold: float, threshold to filter low values
        direction: str or None, optional, 'up' for finding local maxima in positive z direction, 'down' for negative z direction, or None for any direction
        current_point: (3,) ndarray, the current point coordinates, only used when direction is not None
    Returns:
        top_k_points: (k, 3) ndarray, coordinates of these points
        top_k_values: (k,) ndarray, values of these points
    """
    
    # Step 1: Non-Maximum Suppression using max pooling
    heatmap_unsqueezed = heatmap.unsqueeze(0).unsqueeze(0)
    # print(heatmap_unsqueezed.shape)
    max_pool_unsqueezed = F.max_pool3d(heatmap_unsqueezed, kernel_size=5, stride=1, padding=2)
    # print(max_pool_unsqueezed.shape)
    max_pool = max_pool_unsqueezed.squeeze(0).squeeze(0)
    local_max = (heatmap == max_pool)   
    # print(local_max.shape)

    # Step 2: Threshold to filter low values
    threshold = threshold * torch.max(heatmap)
    thresholded = (heatmap > threshold)  
    local_max = local_max & thresholded
    # print(local_max.shape)
    
    # Step 3: Filter local maxima in a specific direction
    if direction is not None:
        assert current_point is not None
        if direction == 'up':
            local_max = local_max & (current_point[2] < torch.arange(heatmap.shape[2]).unsqueeze(0).unsqueeze(0).to(heatmap.device))
        elif direction == 'down':
            local_max = local_max & (current_point[2] > torch.arange(heatmap.shape[2]).unsqueeze(0).unsqueeze(0).to(heatmap.device))
        else:
            raise ValueError("Invalid direction. Must be 'up' or 'down'.")

    # Step 4: Get coordinates and values of local maxima
    indices = torch.nonzero(local_max)
    # values = heatmap[local_max]   
    # print(indices[:, 0].shape)
    values = heatmap[indices[:, 0], indices[:, 1], indices[:, 2]]
    # print(indices.shape, values.shape)

    # Step 5: Sort by value and get top k
    sorted_indices = torch.argsort(values, descending=True)
    top_k_indices = sorted_indices[:k]

    top_k_points = indices[top_k_indices]
    top_k_values = values[top_k_indices]
    
    top_k_points = top_k_points.detach().cpu().numpy().astype(float)
    top_k_values = top_k_values.detach().cpu().numpy().astype(float)

    return top_k_points, top_k_values


def find_closest_skeleton_points(skeleton, coord, n, direction=None):
    """
    Find the closest foreground points in a 3D skeleton image.

    Parameters:
        skeleton: 3D numpy array, a binary image representing the skeleton (True for foreground).
        coord: tuple of floats (x, y, z), the target coordinate.
        n: int, the number of closest points to find.
        direction: str or None, optional, 'up' for positive z direction, 'down' for negative z direction,
                 or None for any direction.

    Returns:
        closest_points: Numpy array of shape (n, 3), the coordinates of the n closest points.
        closest_distances: Numpy array of shape (n,), the Euclidean distances from the target coordinate to the n closest points.
        
    """
    # Convert the coordinate to numpy array for distance calculation
    if isinstance(skeleton, torch.Tensor):
        skeleton = skeleton.detach().cpu().numpy()
    if not isinstance(coord, np.ndarray):
        coord = np.array(coord)

    # Find the indices of all foreground points in the skeleton
    foreground_indices = np.argwhere(skeleton)
    # print(foreground_indices, np.array([coord], dtype=float))

    if direction == 'up':
        # Filter for points above the given coordinate in the z-direction
        foreground_indices = foreground_indices[foreground_indices[:, 2] > coord[2]]
    elif direction == 'down':
        # Filter for points below the given coordinate in the z-direction
        foreground_indices = foreground_indices[foreground_indices[:, 2] < coord[2]]

    # Calculate the Euclidean distance from the target coordinate to each foreground point
    distances = distance.cdist(np.array([coord], dtype=float), foreground_indices, 'euclidean')[0]
    # print(distance.cdist(np.array([coord], dtype=float), foreground_indices, 'euclidean'), distances)

    # Get the indices of the n closest points
    closest_indices = np.argsort(distances)[:n]

    # Get the coordinates and distances of the n closest points
    # closest_points = [(tuple(foreground_indices[i]), distances[i]) for i in closest_indices]
    closest_points = np.array([np.array(foreground_indices[i], dtype=float) for i in closest_indices], dtype=float)
    closest_distances = np.array([distances[i] for i in closest_indices], dtype=float)

    return closest_points, closest_distances


def reverse_direction(direction):
    if direction == 'up':
        return 'down'
    elif direction == 'down':
        return 'up'
    else:
        raise ValueError("Invalid direction. Must be 'up' or 'down'.")


def calculate_edge_cost(candidate_middle, candidate_non_middle, candidate_middle_value, candidate_non_middle_value, 
                        n, direction, skeleton_mask_np, c1=0.4, c2=0.3, c3=0.3):
    
    skeleton_mask_np_dilated = binary_dilation(skeleton_mask_np, structure=dilation_structure, iterations=1)
    
    closest_points_middle, closest_distances_middle = find_closest_skeleton_points(skeleton_mask_np, candidate_middle, n)
    closest_point_middle = closest_points_middle[0]
    
    closest_points_to_one_dir_forward, closest_distances_to_one_dir_forward = find_closest_skeleton_points(skeleton_mask_np, closest_point_middle, n, direction=direction)
    # Remove points with distances greater than 8 (corresponding points in closest_distances_to_one_dir_forward)
    closest_points_to_one_dir_forward = closest_points_to_one_dir_forward[closest_distances_to_one_dir_forward < 8.0]
    if closest_points_to_one_dir_forward.shape[0] == 0:
        suited_unit_vector_forward = 0.0
    else:
        closest_point_to_one_dir_forward_mean = np.mean(closest_points_to_one_dir_forward, axis=0)
        suited_unit_vector_forward = compute_unit_vector(closest_point_middle, closest_point_to_one_dir_forward_mean)
    closest_points_to_one_dir_backward, closest_distances_to_one_dir_backward = find_closest_skeleton_points(skeleton_mask_np, closest_point_middle, n, direction=reverse_direction(direction))
    # Remove points with distances greater than 8
    closest_points_to_one_dir_backward = closest_points_to_one_dir_backward[closest_distances_to_one_dir_backward < 8.0]
    if closest_points_to_one_dir_backward.shape[0] == 0:
        suited_unit_vector_backward = 0.0
    else:
        closest_point_to_one_dir_backward_mean = np.mean(closest_points_to_one_dir_backward, axis=0)
        suited_unit_vector_backward = compute_unit_vector(closest_point_middle, closest_point_to_one_dir_backward_mean)
    assert(isinstance(suited_unit_vector_forward, np.ndarray) or isinstance(suited_unit_vector_backward, np.ndarray))
    suited_unit_vector = 0.6 * suited_unit_vector_forward - 0.4 * suited_unit_vector_backward
    suited_unit_vector = suited_unit_vector / np.linalg.norm(suited_unit_vector, ord=2)
    
    
    # criterion 1: dot product
    if np.linalg.norm(candidate_middle - candidate_non_middle) == 0:
        candidate_dot_product = 0.0
    else:
        candidate_unit_vector = compute_unit_vector(candidate_middle, candidate_non_middle)
        candidate_dot_product = np.dot(candidate_unit_vector, suited_unit_vector)      # normalize to [-1, 1], larger is better
        candidate_dot_product = (candidate_dot_product + 1) / 2           # normalize to [0, 1], larger is better
    
    # criterion 2: distance
    c_p_nm, c_dis_nm = find_closest_skeleton_points(skeleton_mask_np_dilated, candidate_non_middle, n=1)
    c_dis_nm = c_dis_nm[0]
    c_p_m, c_dis_m = find_closest_skeleton_points(skeleton_mask_np_dilated, candidate_middle, n=1)
    c_dis_m = c_dis_m[0]
    dis_score_nm = np.exp(- 0.01 * c_dis_nm)   # normalize to [0, 1], larger is better
    dis_score_m = np.exp(- 0.01 * c_dis_m)   # normalize to [0, 1], larger is better
    dis_score = dis_score_nm + 0.5 * dis_score_m
    dis_score = dis_score / 1.5   # normalize to [0, 1], larger is better
    
    # criterion 3: heatmap value
    candidate_value = candidate_middle_value * 0.5 + candidate_non_middle_value
    candidate_value = candidate_value / 1.5              # normalize to [0, 1], larger is better
    
    
    final_score = c1 * candidate_dot_product + c2 * dis_score + c3 * candidate_value   # normalize to [0, 1], larger is better
    final_cost = 1 - final_score              # normalize to [0, 1], smaller is better   
    
    return final_cost


def construct_min_cost_flow(candidates_crown, candidates_middle, candidates_root, 
                            candidate_crown_values, candidate_middle_values, candidate_root_values,
                            root_canal_num, n, crown_direction, skeleton_mask_np, c1=0.4, c2=0.3, c3=0.3):
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    
    # 节点数
    num_nodes = 1 + len(candidates_crown) + 2 * len(candidates_middle) + len(candidates_root) + 1
    source = 0
    sink = num_nodes - 1
    
    # 源节点到a组节点的边
    for i in range(len(candidates_crown)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(source, i + 1, 1, 0)
    
    # a组节点到b组节点的边
    for i in range(len(candidates_crown)):
        for j in range(len(candidates_middle)):
            cost_f = calculate_edge_cost(candidates_middle[j], candidates_crown[i],
                                         candidate_middle_values[j], candidate_crown_values[i], 
                                         n, crown_direction, skeleton_mask_np, c1=c1, c2=c2, c3=c3)
            cost = int(cost_f * 1e15)  # 放大成本，避免小数问题
            min_cost_flow.AddArcWithCapacityAndUnitCost(i + 1, j + 1 + len(candidates_crown), 1, cost)
    
    # b组节点到中间节点的边
    for i in range(len(candidates_middle)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(i + 1 + len(candidates_crown), i + 1 + len(candidates_crown) + len(candidates_middle), 1, 0)
    
    # 中间节点到c组节点的边
    for i in range(len(candidates_middle)):
        for j in range(len(candidates_root)):
            cost_f = calculate_edge_cost(candidates_middle[i], candidates_root[j],
                                         candidate_middle_values[i], candidate_root_values[j], 
                                         n, reverse_direction(crown_direction), skeleton_mask_np, c1=c1, c2=c2, c3=c3)
            cost = int(cost_f * 1e15)  # 放大成本，避免小数问题
            min_cost_flow.AddArcWithCapacityAndUnitCost(i + 1 + len(candidates_crown) + len(candidates_middle), j + 1 + len(candidates_crown) + 2 * len(candidates_middle), 1, cost)
    
    # c组节点到汇节点的边
    for i in range(len(candidates_root)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(i + 1 + len(candidates_crown) + 2 * len(candidates_middle), sink, 1, 0)
    
    # 设置每个节点的需求
    min_cost_flow.SetNodeSupply(source, root_canal_num)
    min_cost_flow.SetNodeSupply(sink, -root_canal_num)
    
    return min_cost_flow


def mcf_optimization(candidates_crown, candidates_middle, candidates_root, 
                    candidate_crown_values, candidate_middle_values, candidate_root_values,
                    root_canal_num, n, crown_direction, skeleton_mask_np, max_attempts=5, c1=0.4, c2=0.3, c3=0.3):
    attempts = 0
    
    points_list_np = np.zeros((root_canal_num, 3, 3), dtype=float)
    
    while attempts < max_attempts:
        attempts += 1

        min_cost_flow = construct_min_cost_flow(candidates_crown, candidates_middle, candidates_root, 
                                                candidate_crown_values, candidate_middle_values, candidate_root_values,
                                                root_canal_num, n, crown_direction, skeleton_mask_np, c1=c1, c2=c2, c3=c3)
        
        if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
            selected_crown = set()
            selected_middle = set()
            selected_root = set()
            connections_ab = []
            connections_bc = []
            
            for i in range(min_cost_flow.NumArcs()):
                if min_cost_flow.Flow(i) > 0:
                    assert min_cost_flow.Flow(i) == 1
                    tail = min_cost_flow.Tail(i)   # start
                    head = min_cost_flow.Head(i)   # end
                    
                    if tail == 0:
                        selected_crown.add('crown_' + str(head - 1))
                    elif head == min_cost_flow.NumNodes() - 1:
                        selected_root.add('root_' + str(tail - 1 - len(candidates_crown) - 2 * len(candidates_middle)))
                    elif 1 <= tail <= len(candidates_crown) and len(candidates_crown) < head <= len(candidates_crown) + len(candidates_middle):
                        selected_crown.add('crown_' + str(tail - 1))
                        selected_middle.add('middle_' + str(head - 1 - len(candidates_crown)))
                        connections_ab.append(('crown_' + str(tail - 1), 'middle_' + str(head - 1 - len(candidates_crown))))
                    elif len(candidates_crown) + len(candidates_middle) < tail <= len(candidates_crown) + 2 * len(candidates_middle) and head > len(candidates_crown) + 2 * len(candidates_middle):
                        selected_middle.add('middle_' + str(tail - 1 - len(candidates_crown) - len(candidates_middle)))
                        selected_root.add('root_' + str(head - 1 - len(candidates_crown) - 2 * len(candidates_middle)))
                        connections_bc.append(('middle_' + str(tail - 1 - len(candidates_crown) - len(candidates_middle)), 'root_' + str(head - 1 - len(candidates_crown) - 2 * len(candidates_middle))))
                    elif len(candidates_crown) < tail <= len(candidates_crown) + len(candidates_middle) and len(candidates_crown) + len(candidates_middle) < head <= len(candidates_crown) + 2 * len(candidates_middle):
                        assert tail - len(candidates_crown) - 1 == head - len(candidates_crown) - len(candidates_middle) - 1
            
            selected_crown = list(selected_crown)
            selected_middle = list(selected_middle)
            selected_root = list(selected_root)
            
            filtered_connections = []
            b_to_c = {b: c for b, c in connections_bc if b in selected_middle}
            
            for a, b in connections_ab:
                if a in selected_crown and b in selected_middle:
                    c = b_to_c.get(b)
                    if c in selected_root:
                        filtered_connections.append((a, b, c))
                    
            if len(selected_crown) == root_canal_num and len(selected_middle) == root_canal_num and len(selected_root) == root_canal_num and len(filtered_connections) == root_canal_num:
                
                for i, (c, m, r) in enumerate(filtered_connections):
                    points_list_np[i, 0, :] = candidates_crown[int(c.split('_')[1])]
                    points_list_np[i, 1, :] = candidates_middle[int(m.split('_')[1])]
                    points_list_np[i, 2, :] = candidates_root[int(r.split('_')[1])]
                
                print(f"Attempt {attempts}: Valid solution.")
                print('selected_crown:', selected_crown)
                print('selected_middle:', selected_middle)
                print('selected_root:', selected_root)
                print('filtered_connections:', filtered_connections)
                
                return points_list_np
            else:
                print(f"Attempt {attempts}: Invalid solution.")
    
    
    return points_list_np


def decode_heatmap_v2(output_hm, root_canal_num, seg_mask, k=6, threshold=0.3, n=3, c1=0.4, c2=0.3, c3=0.3):
    '''
    Decode the heatmap to get the root canal points based on global optimization using min cost flow algorithm.
    
    Parameters:
        output_hm: 5D tensor, the output heatmap from the model.
        root_canal_num: int, the number of root canals to find.
        seg_mask: 3D tensor, the segmentation mask of the input volume.
        k: int, the number of local maxima to find in the crown and root heatmaps.
        threshold: float, the threshold to filter low values in the heatmap.
        n: int, the number of closest points to find in the skeleton.
    
    Returns:
        points_list: numpy array, the coordinates of the root canal points. dimensions: (root_canal_num, 3, 3).
    '''
    
    # output_hm (N, C, D, H, W)
    
    output_hm = output_hm.squeeze(0)  # (C, D, H, W)
    C, D, H, W = output_hm.shape
    assert(C == 3)
    points_list_np = np.zeros((root_canal_num, 3, 3), dtype=float)
    # points_list = []
    
    seg_mask = seg_mask.squeeze()
    assert(seg_mask.dim() == 3)
    seg_mask_np = seg_mask.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    skeleton_mask_np = skeletonize_3d(seg_mask_np).squeeze().astype(bool)
    
    heatmap_middle = output_hm[1, :, :, :].permute(1, 2, 0)
    heatmap_crown = output_hm[0, :, :, :].permute(1, 2, 0)
    heatmap_root = output_hm[2, :, :, :].permute(1, 2, 0)
      
    top_rcn_points_middle, top_rcn_values_middle = find_top_k_local_maxima(heatmap_middle, root_canal_num, threshold)
    # print(top_rcn_points_middle, top_rcn_values_middle)
    top_rcn_points_crown, top_rcn_values_crown = find_top_k_local_maxima(heatmap_crown, root_canal_num, threshold)
    # print(top_rcn_points_crown, top_rcn_values_crown)
   
    top_point_crown_HWD_z = top_rcn_points_crown[0][2]
    top_point_middle_HWD_z = top_rcn_points_middle[0][2]
    
    # sort the points in the middle heatmap based on z coordinate:
    sorted_rcn_points_middle = top_rcn_points_middle[top_rcn_points_middle[:, 2].argsort()]  # default: ascending order
    min_z_point_middle_HWD = sorted_rcn_points_middle[0]
    max_z_point_middle_HWD = sorted_rcn_points_middle[-1]
    
    assert(top_point_crown_HWD_z != top_point_middle_HWD_z)
    if top_point_crown_HWD_z > top_point_middle_HWD_z:
        crown_direction = 'up'
        root_direction = 'down'
        crown_bounary_point_middle_HWD = min_z_point_middle_HWD
        root_bounary_point_middle_HWD = max_z_point_middle_HWD
    else:
        crown_direction = 'down'
        root_direction = 'up'
        crown_bounary_point_middle_HWD = max_z_point_middle_HWD
        root_bounary_point_middle_HWD = min_z_point_middle_HWD
        
    
    # k_middle = root_canal_num + 2 if root_canal_num > 2 else root_canal_num * 2        
    # k_middle = root_canal_num
    k_middle = root_canal_num * 2
    top_k_points_middle, top_k_values_middle = find_top_k_local_maxima(heatmap_middle, k_middle, threshold)   
    top_k_points_crown, top_k_values_crown = find_top_k_local_maxima(heatmap_crown, k, threshold, 
                                                                     direction=crown_direction, 
                                                                     current_point=crown_bounary_point_middle_HWD)
    top_k_points_root, top_k_values_root = find_top_k_local_maxima(heatmap_root, k, threshold, 
                                                                   direction=root_direction, 
                                                                   current_point=root_bounary_point_middle_HWD)
    
    
    tt = time.time()
    points_list_np = mcf_optimization(top_k_points_crown, top_k_points_middle, top_k_points_root, 
                                      top_k_values_crown, top_k_values_middle, top_k_values_root,
                                      root_canal_num, n, crown_direction, skeleton_mask_np, c1=c1, c2=c2, c3=c3)
    print('Time of graph construction and optimization:', time.time() - tt)
        
    
    return points_list_np


#####################################################decode#####################################################








####################################################utils####################################################


def fromCurrentOffsettoCurrentSlice(currentOffset, LowerBound, UpperBound, numberOfChannels):
    minOffset = LowerBound + ((UpperBound - LowerBound) / (2 * numberOfChannels))
    gapValue = (UpperBound - LowerBound) / numberOfChannels
    
    return round((currentOffset - minOffset) / gapValue)


def save_json_data_ldm(datas, folder_path, points_world, json_name, color=[1.0, 0.4, 1.0], return_json_dict=False):
    
    assert(len(points_world.shape) <= 3)
    if len(points_world.shape) == 2:
        points_world_tmp = points_world[np.newaxis, :, :]
    elif len(points_world.shape) == 1:
        points_world_tmp = points_world[np.newaxis, np.newaxis, :]
    else:
        points_world_tmp = points_world
    assert(len(points_world_tmp.shape) == 3)
    
    data = copy.deepcopy(datas)
    if len(data) == len(points_world_tmp):
        print('data length is equal to points_world length:', len(data))
    elif len(data) > len(points_world_tmp):
        data = data[:len(points_world_tmp)]
        print('data length:', len(data))
    elif len(data) < len(points_world_tmp):
        diff = len(points_world_tmp) - len(data)
        for i in range(diff):
            data.append(copy.deepcopy(data[0]))
        print('data length:', len(data))
    
    for i, line_world in enumerate(points_world_tmp):
        # print(i)
        control_point_template = data[i]["markups"][0]["controlPoints"][0]
        data[i]["markups"][0]["controlPoints"] = []
        data[i]["markups"][0]["display"]["color"] = list(1.0 - np.array(color))
        data[i]["markups"][0]["display"]["activeColor"] = list(1.0 - np.array(color))
        data[i]["markups"][0]["display"]["selectedColor"] = color
        # print(control_point_template)
        
        for j, point_world in enumerate(line_world):
            control_point_dict = copy.deepcopy(control_point_template)
            control_point_dict["position"] = list(point_world)
            control_point_dict["id"] = str(j + 1)
            try:
                control_point_dict["label"] = control_point_template["label"].split('-')[0] + '-' + json_name + '-' + str(j + 1)
            except:
                control_point_dict["label"] = control_point_template["label"][0].split('-')[0] + '-' + json_name + '-' + str(j + 1)
            # print(list(point_world))
            # print(control_point_dict)
            
            data[i]["markups"][0]["controlPoints"].append(control_point_dict)
        data[i]["markups"][0]["lastUsedControlPointNumber"] = len(data[i]["markups"][0]["controlPoints"])
        
        os.makedirs(folder_path + ' - Out Json', exist_ok=True)
        json.dump(data[i], open(os.path.join(folder_path + ' - Out Json', json_name + "_" + str(i+1) + '.json'), 'w'), indent=4)
        
    if return_json_dict:
        return data


def make_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.makedirs(path)


def load_medical_image_normalize_np(path, type=None, resample=None,
                       viz3d=False, to_canonical=False, rescale=None, rescale_order=None, 
                       clip_intenisty=True, window_center='none', window_width='none'):
    img_nii = nib.load(path)
    
    if to_canonical:
        img_nii = nib.as_closest_canonical(img_nii)

    if resample is not None:
        img_nii = processing.resample_to_output(img_nii, voxel_sizes=resample)
    
    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))

    if viz3d:
        return torch.from_numpy(img_np)

    # 1. Intensity outlier clipping
    if clip_intenisty and type != "label":
        img_np = percentile_clip(img_np)

    # 2. Rescale to specified output shape
    if rescale is not None:
        img_np = rescale_data_volume(img_np, rescale, rescale_order, consistent_scale=True)

    # 3. intensity normalization
    if type != "label":
        if window_center == 'none' and window_width == 'none':
            min_window = img_np.min()
            max_window = img_np.max()
        elif window_center != 'none' and window_width != 'none':
            min_window = window_center - window_width / 2
            max_window = window_center + window_width / 2
        else:
            raise ValueError('Both window_center and window_width should be set or neither should be set')
        
        ct_windowed = np.clip(img_np, min_window, max_window)

        img_np = (ct_windowed - min_window) / (max_window - min_window)


    return img_np

def load_medical_image_affine_and_size(path):
    img_nii = nib.load(path)
    affine = img_nii.affine
    
    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))
    size = img_np.shape

    return affine, size

def pad_volume_to_center(original_volume, target_size):
    padded_volume = np.zeros(target_size, dtype=original_volume.dtype)
    
    original_shape = original_volume.shape
    padding = [(target_dim - original_dim) // 2 for target_dim, original_dim in zip(target_size, original_shape)]
    
    start_idx = [pad for pad in padding]
    end_idx = [original_dim + pad for pad, original_dim in zip(padding, original_shape)]

    padded_volume[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]] = original_volume
    
    return padded_volume, start_idx, end_idx

def percentile_clip(img_numpy, min_val=0.1, max_val=99.8):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    :param min_val: should be in the range [0,100]
    :param max_val: should be in the range [0,100]
    :return: intesity normalized image
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)

    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy

def rescale_data_volume(img_numpy, out_dim, order=0, mode='nearest', consistent_scale=True):
    """
    Resize the 3d numpy array to the dim size
    :param out_dim is the new 3d tuple
    in fact d h w is h w d, The order is consistent so it does not affect
    """
    depth, height, width = img_numpy.shape
    if not consistent_scale:
        scale = [out_dim[0] * 1.0 / depth, out_dim[1] * 1.0 / height, out_dim[2] * 1.0 / width]
    else:
        scale_per_dim = min([out_dim[0] * 1.0 / depth, out_dim[1] * 1.0 / height, out_dim[2] * 1.0 / width])
        scale = [scale_per_dim, scale_per_dim, scale_per_dim]
    
    return ndimage.interpolation.zoom(img_numpy, scale, order=order, mode=mode)


def matrix4x4_multiply_vector3(affine, coords):
    coords_homogeneous = np.append(coords, 1)
    transformed_coords = np.dot(affine, coords_homogeneous)
    return transformed_coords[:3]

def matrix4x4_multiply_multidim_vector3(affine_matrix, coords):
    N, M, _ = coords.shape
    coords_homogeneous = np.ones((N, M, 4))
    coords_homogeneous[..., :3] = coords
    
    transformed_coords_homogeneous = np.einsum('ij,nmj->nmi', affine_matrix, coords_homogeneous)
    
    transformed_coords = transformed_coords_homogeneous[..., :3]
    return transformed_coords

def world_coordinate_to_index_coordinate(affine_matrix, matrix_orientation, point_world):
    affine_matrix_inv = np.linalg.inv(affine_matrix)
    point_index = matrix4x4_multiply_vector3(affine_matrix_inv, matrix_orientation @ point_world)
    return point_index

def index_coordinate_to_world_coordinate(affine_matrix, matrix_orientation, point_index):
    point_world = matrix4x4_multiply_vector3(affine_matrix, point_index)
    point_world = np.dot(np.linalg.inv(matrix_orientation), point_world)
    return point_world


def world_distance_to_index_scalar(affine_matrix, matrix_orientation, d_world):
    """
    将世界坐标系的标量距离转换为索引坐标系下的距离（方向未知时的几何平均解）

    参数:
        affine_matrix (np.ndarray): 4x4仿射矩阵（世界坐标到索引坐标的变换矩阵的逆）
        matrix_orientation (np.ndarray): 3x3方向调整矩阵
        d_world (float): 世界坐标系中的标量距离

    返回:
        float: 索引坐标系下的估算距离
    """
    # 计算仿射矩阵的逆并提取线性部分（3x3）
    affine_inv = np.linalg.inv(affine_matrix)
    affine_linear_inv = affine_inv[:3, :3]

    # 组合线性变换矩阵：affine_inv_linear @ matrix_orientation
    combined_linear = np.dot(affine_linear_inv, matrix_orientation)

    # 计算行列式绝对值（体积缩放比例）
    det = np.abs(np.linalg.det(combined_linear))
    if np.isclose(det, 0):
        raise ValueError("线性变换矩阵不可逆，无法计算缩放因子")

    # 几何平均缩放因子：det^(1/3)
    avg_scaling = det ** (1 / 3)

    # 转换距离
    d_index = d_world * avg_scaling
    return d_index



def largest_connected_component(binary_image):
    """
    Find the largest connected component in a 3D binary image and retain it.

    Parameters:
    - binary_image: 3D numpy ndarray or PyTorch tensor, the input binary image (0-1 values).

    Returns:
    - output_image: 3D numpy ndarray or PyTorch tensor, the output binary image with only the largest component retained.
    """
    if isinstance(binary_image, np.ndarray):
        input_type = 'numpy'
    else:
        if isinstance(binary_image, torch.Tensor):
            input_type = 'tensor'
            device = binary_image.device
            binary_image = binary_image.detach().cpu().numpy()  # Convert to numpy for processing
        else:
            raise TypeError("Input must be a numpy ndarray or a PyTorch tensor.")
    
    # Label connected components
    labeled_image, num_features = ndimage.label(binary_image)
    
    # Find the largest connected component
    if num_features == 0:
        raise ValueError("No foreground components found in the binary image.")
    
    component_sizes = np.bincount(labeled_image.ravel())
    largest_component_label = component_sizes[1:].argmax() + 1  # Exclude the background label 0
    
    # Create an output binary image with only the largest component
    output_image = (labeled_image == largest_component_label).astype(np.int32)
    
    if input_type == 'tensor':
        output_image = torch.from_numpy(output_image).to(device)  # Convert back to tensor if necessary
    
    return output_image


###################################################utils####################################################







#####################################################network#####################################################


class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        
        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)
    
class OutputBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(output_channels, affine=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size=1)
        self.act1 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out

class OutputBlockV2(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(input_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(input_channels, input_channels, kernel_size=3, padding=1)
        self.bn2 = nn.InstanceNorm3d(input_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        # self.conv3 = nn.Conv3d(input_channels, input_channels, kernel_size=1)

        self.conv_out = nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.bn1(y))

        y = self.bn2(self.conv2(y))

        # x1 = self.conv3(x)
        # y += x1
        y = self.act2(y)

        out = self.conv_out(y)
        return out

class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='trilinear'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x
    
class STUNet(nn.Module):

    def __init__(self, input_channels, num_classes, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        self.final_nonlin = lambda x:x 
        self._deep_supervision = False
        self.do_ds = False
        seg_output_use_bias = False
        self.upscale_logits = False

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])
       
        num_pool  = len(pool_op_kernel_sizes)
        
        assert num_pool == len(dims) - 1
        
        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True), 
                              *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0]-1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool+1):
            stage = nn.Sequential(BasicResBlock(dims[d-1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d], stride=self.pool_op_kernel_sizes[d-1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d]) for _ in range(depth[d]-1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1-u], dims[-2-u], pool_op_kernel_sizes[-1-u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2-u] * 2, dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u], use_1x1conv=True),
                *[BasicResBlock(dims[-2-u], dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u]) for _ in range(depth[-2-u]-1)])
            self.conv_blocks_localization.append(stage)
            
        # outputs    
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            # self.seg_outputs.append(nn.Conv3d(dims[-2-ds], num_classes, kernel_size=1))
            self.seg_outputs.append(OutputBlock(dims[-2-ds], num_classes))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)
        
        # self.apply(self.weightInitializer)

    def forward(self, x):
        skips = []
        seg_outputs = []
        
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1) 
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
    
    def test(self, device='cpu'):
        
        input_tensor = torch.rand(1, self.input_channels, 192, 128, 128).to(device)
        ideal_out = torch.rand(1, self.num_classes, 192, 128, 128).to(device)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape

        print("STUNet test is complete")


class STUNet_v2(nn.Module):

    def __init__(self, input_channels, num_classes, depth=[1, 1, 1, 1, 1, 1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, use_output_v2=False):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        # self.weightInitializer = InitWeights_He(1e-2)

        self.final_nonlin = lambda x: x
        self._deep_supervision = False
        self.do_ds = False
        seg_output_use_bias = False
        self.upscale_logits = False

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in
              range(depth[0] - 1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                                                stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                                  *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                                    for _ in range(depth[d] - 1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2 - u] * 2, dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                self.conv_pad_sizes[-2 - u], use_1x1conv=True),
                                  *[BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                  self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)])
            self.conv_blocks_localization.append(stage)

        # outputs
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            # self.seg_outputs.append(nn.Conv3d(dims[-2-ds], num_classes, kernel_size=1))
            if use_output_v2:
                self.seg_outputs.append(OutputBlockV2(dims[-2 - ds], num_classes))
            else:
                self.seg_outputs.append(OutputBlock(dims[-2 - ds], num_classes))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

        # self.apply(self.weightInitializer)

    def forward(self, x):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    def test(self, device='cpu'):

        input_tensor = torch.rand(1, self.input_channels, 192, 128, 128)
        ideal_out = torch.rand(1, self.num_classes, 192, 128, 128)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape

        print("STUNet test is complete")


class BasicLinearBlock(nn.Module):
    
    def __init__(self, in_neuron, out_neuron, dropoutrate=0.0):
        super(BasicLinearBlock, self).__init__()
        self.fc = nn.Linear(in_neuron, out_neuron)
        # self.bn1 = nn.BatchNorm1d(out_neuron)
        self.in1 = nn.InstanceNorm1d(out_neuron)
        self.activate = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout(dropoutrate)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.in1(x)
        x = self.activate(x)
        x = self.drop(x)
        
        return x

def normalize_and_concat_features(x1, x2, x3):
    '''
    Normalize the input features and concatenate them
    
    Parameters:
    x1: torch.tensor of shape (batch_size, num_features1)
    x2: torch.tensor of shape (batch_size, num_features2)
    x3: torch.tensor of shape (batch_size, num_features3)
    
    Returns:
    x_n: torch.tensor of shape (batch_size, num_features1 + num_features2 + num_features3)
    '''
    
    x1_n = x1 / torch.norm(x1, p=2, dim=1, keepdim=True)
    x2_n = x2 / torch.norm(x2, p=2, dim=1, keepdim=True)
    x3_n = x3 / torch.norm(x3, p=2, dim=1, keepdim=True)
    
    x_n = torch.cat((x1_n, x2_n, x3_n), dim=1)
    # print(x_n.shape)
    return x_n

class STUNet_hm_cls(nn.Module):

    def __init__(self, input_channels, num_classes_seg, num_classes_hm, num_classes_cls, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes_seg = num_classes_seg
        self.num_classes_hm = num_classes_hm
        self.num_classes_cls = num_classes_cls
        
        self.final_nonlin_seg = lambda x:x 
        self.final_nonlin_hm = nn.Sigmoid()
        
        self._deep_supervision = False
        self.do_ds = False
        seg_output_use_bias = False
        self.upscale_logits = False

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        
        num_pool  = len(pool_op_kernel_sizes)
        
        assert num_pool == len(dims) - 1
        
        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True), 
                              *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0]-1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool+1):
            stage = nn.Sequential(BasicResBlock(dims[d-1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d], stride=self.pool_op_kernel_sizes[d-1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d]) for _ in range(depth[d]-1)])
            self.conv_blocks_context.append(stage)
        
        
        self.cls_conv = BasicResBlock(dims[-1], dims[-1], self.conv_kernel_sizes[-1], self.conv_pad_sizes[-1], use_1x1conv=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(BasicLinearBlock(dims[-1], dims[-1], 0.3), nn.Linear(dims[-1], num_classes_cls))


        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1-u], dims[-2-u], pool_op_kernel_sizes[-1-u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2-u] * 2, dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u], use_1x1conv=True),
                *[BasicResBlock(dims[-2-u], dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u]) for _ in range(depth[-2-u]-1)])
            self.conv_blocks_localization.append(stage)
            
        # outputs    
        self.seg_outputs = nn.ModuleList()
        self.hm_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            # self.seg_outputs.append(nn.Conv3d(dims[-2-ds], num_classes_seg, kernel_size=1))
            # self.hm_outputs.append(nn.Conv3d(dims[-2-ds], num_classes_hm, kernel_size=1))
            self.seg_outputs.append(OutputBlock(dims[-2-ds], num_classes_seg))
            self.hm_outputs.append(OutputBlock(dims[-2-ds], num_classes_hm))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)
        
        # self.apply(self.weightInitializer)

    def forward(self, x):
        skips = []
        seg_outputs = []
        hm_outputs = []
        
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            # print(x.shape)

        x = self.conv_blocks_context[-1](x)
        # print(x.shape)
        
        x_cls = self.cls_conv(x)
        # print(x_cls.shape)
        x_cls = self.avgpool(x_cls)
        x_cls = x_cls.view(x_cls.size(0), -1)
        cls_outputs = self.fc(x_cls)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1) 
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin_seg(self.seg_outputs[u](x)))
            hm_outputs.append(self.final_nonlin_hm(self.hm_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]), tuple([hm_outputs[-1]] + [i(j) for i, j in 
                                            zip(list(self.upscale_logits_ops)[::-1], hm_outputs[:-1][::-1])]), cls_outputs
        else:
            return seg_outputs[-1], hm_outputs[-1], cls_outputs
       
    def test(self, device='cpu'):
        
        input_tensor = torch.rand(3, self.input_channels, 192, 128, 128).to(device)
        ideal_out_seg = torch.rand(3, self.num_classes_seg, 192, 128, 128).to(device)
        ideal_out_hm = torch.rand(3, self.num_classes_hm, 192, 128, 128).to(device)
        ideal_out_cls = torch.rand(3, self.num_classes_cls).to(device)
        out_seg, out_hm, out_cls = self.forward(input_tensor)
        assert ideal_out_seg.shape == out_seg.shape
        assert ideal_out_hm.shape == out_hm.shape
        assert ideal_out_cls.shape == out_cls.shape

        print("STUNet-hm-cls test is complete")


class STUNet_hm_cls_v2(nn.Module):

    def __init__(self, input_channels, num_classes_seg, num_classes_hm, num_classes_cls, depth=[1, 1, 1, 1, 1, 1],
                 dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, use_output_v2=False):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes_seg = num_classes_seg
        self.num_classes_hm = num_classes_hm
        self.num_classes_cls = num_classes_cls

        # self.weightInitializer = InitWeights_He(1e-2)

        self.final_nonlin_seg = lambda x: x
        # self.final_nonlin_hm = nn.Sigmoid()
        self.final_nonlin_hm = lambda x: x

        self._deep_supervision = False
        self.do_ds = False
        seg_output_use_bias = False
        self.upscale_logits = False

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)

        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in
              range(depth[0] - 1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(BasicResBlock(dims[d - 1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d],
                                                stride=self.pool_op_kernel_sizes[d - 1], use_1x1conv=True),
                                  *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                                    for _ in range(depth[d] - 1)])
            self.conv_blocks_context.append(stage)

        self.cls_conv = BasicResBlock(dims[-1], dims[-1], self.conv_kernel_sizes[-1], self.conv_pad_sizes[-1],
                                      use_1x1conv=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if use_output_v2:
            self.fc = nn.Sequential(BasicLinearBlock(dims[-1], dims[-1], 0.3), nn.Linear(dims[-1], num_classes_cls))
        else:
            self.fc = nn.Sequential(BasicLinearBlock(dims[-1], dims[-1], 0.3), nn.Linear(dims[-1], num_classes_cls))

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2 - u] * 2, dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                self.conv_pad_sizes[-2 - u], use_1x1conv=True),
                                  *[BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u],
                                                  self.conv_pad_sizes[-2 - u]) for _ in range(depth[-2 - u] - 1)])
            self.conv_blocks_localization.append(stage)

        # outputs
        self.seg_outputs = nn.ModuleList()
        self.hm_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            # self.seg_outputs.append(nn.Conv3d(dims[-2-ds], num_classes_seg, kernel_size=1))
            # self.hm_outputs.append(nn.Conv3d(dims[-2-ds], num_classes_hm, kernel_size=1))
            if use_output_v2:
                self.seg_outputs.append(OutputBlockV2(dims[-2 - ds], num_classes_seg))
                self.hm_outputs.append(OutputBlockV2(dims[-2 - ds], num_classes_hm))
            else:
                self.seg_outputs.append(OutputBlock(dims[-2 - ds], num_classes_seg))
                self.hm_outputs.append(OutputBlock(dims[-2 - ds], num_classes_hm))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

        # self.apply(self.weightInitializer)

    def forward(self, x):
        skips = []
        seg_outputs = []
        hm_outputs = []

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            # print(x.shape)

        x = self.conv_blocks_context[-1](x)
        # print(x.shape)

        x_cls = self.cls_conv(x)
        # print(x_cls.shape)
        x_cls = self.avgpool(x_cls)
        x_cls = x_cls.view(x_cls.size(0), -1)
        cls_outputs = self.fc(x_cls)

        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin_seg(self.seg_outputs[u](x)))
            hm_outputs.append(self.final_nonlin_hm(self.hm_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]), tuple(
                [hm_outputs[-1]] + [i(j) for i, j in
                                    zip(list(self.upscale_logits_ops)[::-1], hm_outputs[:-1][::-1])]), cls_outputs
        else:
            return seg_outputs[-1], hm_outputs[-1], cls_outputs

    def test(self, device='cpu'):

        input_tensor = torch.rand(3, self.input_channels, 160, 96, 96)
        ideal_out_seg = torch.rand(3, self.num_classes_seg, 160, 96, 96)
        ideal_out_hm = torch.rand(3, self.num_classes_hm, 160, 96, 96)
        ideal_out_cls = torch.rand(3, self.num_classes_cls)
        out_seg, out_hm, out_cls = self.forward(input_tensor)
        # print(out_cls.shape)
        # print(len(out_seg), len(out_hm))
        # print(out_seg[0].shape, out_hm[0].shape)
        # print(out_seg[1].shape, out_hm[1].shape)
        # print(out_seg[2].shape, out_hm[2].shape)
        # print(out_seg[3].shape, out_hm[3].shape)
        # print(out_seg[4].shape, out_hm[4].shape)
        assert ideal_out_seg.shape == out_seg.shape
        assert ideal_out_hm.shape == out_hm.shape
        assert ideal_out_cls.shape == out_cls.shape

        print("STUNet-hm-cls test is complete")


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
