cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerPulpChamberOpenPlanning)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/ZhyBrian/SlicerEndoPlanner")
set(EXTENSION_CONTRIBUTORS "Yi Zhang (SJTU), Xiaojun Chen (SJTU)")
set(EXTENSION_DESCRIPTION "EndoPlanner: an adaptive preoperative planning framework for Guided Endodontics, integrating automated root canal landmark detection, minimally invasive access cavity planning, and surgical template generation from dental CBCT.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/ZhyBrian/SlicerEndoPlanner/main/SlicerPulpChamberOpenPlanning.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/ZhyBrian/SlicerEndoPlanner/main/extension_fig.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(PulpChamberOpenPlanning)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
