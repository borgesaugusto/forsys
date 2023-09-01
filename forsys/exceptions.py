class DifferentTissueException(Exception):
    def __init__(self, message='Tissues are too different'):
        super(DifferentTissueException, self).__init__(message)

class SegmentationArtifactException(Exception):
    def __init__(self, message='There might be a segmentation artifact from the TIFF'):
        super(SegmentationArtifactException, self).__init__(message)

class BigEdgesBadlyCreated(Exception):
    def __init__(self, message='Big edge was not created correctly'):
        super(BigEdgesBadlyCreated, self).__init__(message)