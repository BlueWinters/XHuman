
from .base import *
from .application import *
from .thirdparty import *


"""
module dict
"""
GlobalModuleDictClass = dict()  # use for devlop mode
GlobalModuleDictSource = dict()  # use for deployment mode

# human base
GlobalModuleDictClass['face_detection'] = LibFaceDetection
GlobalModuleDictClass['face_landmark'] = LibFaceLandmark
GlobalModuleDictClass['head_pose'] = LibHeadPose
GlobalModuleDictClass['face_attribute'] = LibFaceAttribute
GlobalModuleDictClass['portrait_parsing'] = LibPortraitParsing
GlobalModuleDictClass['human_matting'] = LibHumanMatting_DevKit
GlobalModuleDictClass['human_matting_sghm'] = LibHumanMatting_SGHM
GlobalModuleDictClass['human_fine_matting'] = LibHumanFineMatting
GlobalModuleDictClass['function'] = LibFunction

# face-based application
GlobalModuleDictClass['portrait_light'] = LibPortraitLight
GlobalModuleDictClass['portrait_beauty'] = LibPortraitBeauty

# third party
GlobalModuleDictClass['insightface'] = LibInsightFace
GlobalModuleDictClass['face_restoration_gfpgan'] = LibFaceRestorationGFPGAN  # 1.4
GlobalModuleDictClass['human_detection_yolox'] = LibYoloX
GlobalModuleDictClass['rtmpose'] = LibRTMPose
GlobalModuleDictClass['sapiens'] = LibSapiens
