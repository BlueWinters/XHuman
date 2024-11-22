


class XPortraitException(Exception):
    def __init__(self, content):
        self.code = content[0]
        self.info = content[1]

    def __str__(self):
        return '"code: {}, info: {}"'.format(self.code, self.info)



class XPortraitExceptionAssert:
    """
    """
    XPortraitException_NoFace = (1, 'no-face')
    XPortraitException_MultiFace = (2, 'multi-face')

    @staticmethod
    def assertNoFace(num_faces:int):
        if num_faces > 0:
            return  # satisfy, just return
        if num_faces == 0:
            raise XPortraitException(XPortraitExceptionAssert.XPortraitException_NoFace)
        raise ValueError('number of faces: {}'.format(num_faces))

    @staticmethod
    def assertSingle(num_faces:int):
        if num_faces == 1:
            return  # satisfy, just return
        if num_faces == 0:
            raise XPortraitException(XPortraitExceptionAssert.XPortraitException_NoFace)
        if num_faces > 1:
            raise XPortraitException(XPortraitExceptionAssert.XPortraitException_MultiFace)
        raise ValueError('number of faces: {}'.format(num_faces))

    @staticmethod
    def assertKwArgs(name, value, arg_list:list):
        for arg in arg_list:
            if arg == value:
                return  # satisfy, just return
        raise ValueError('unexpected input args "{}": {}, not in {}'.format(name, value, arg_list))

