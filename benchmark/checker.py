
import os


def configLogging():
    import logging
    format = '%(asctime)s - Lv%(levelno)s - %(filename)s:%(lineno)d - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format, datefmt='%Y-%m-%d %H:%M:%S')

def checkFiles():
    def getFolder(path):
        return os.path.split(path)[0]

    from core.checkpoints import checkModelFiles
    path = getFolder(getFolder(__file__))
    checkModelFiles()


if __name__ == '__main__':
    configLogging()
    checkFiles()