
import numpy as np


class AngleHelper:
    @staticmethod
    def getAngle(v):
        if v[1] > 0:
            r = abs(v[0]) / abs(v[1])
            if v[0] < 0:
                if r < 1:
                    return 0
                else:
                    return 270
            elif v[0] > 0:
                if r < 1:
                    return 0
                else:
                    return 90
            else:
                return 0
        elif v[1] < 0:
            r = abs(v[0]) / abs(v[1])
            if v[0] < 0:
                if r < 1:
                    return 180
                else:
                    return 270
            elif v[0] > 0:
                if r < 1:
                    return 180
                else:
                    return 90
            else:
                return 180
        else:
            if v[0] < 0:
                return 270
            else:
                return 90

    @staticmethod
    def getAngleRollByLandmark(landmark):
        vector = landmark[30, :] - np.mean(landmark[17:27, :], axis=0)
        return AngleHelper.getAngle(vector)
