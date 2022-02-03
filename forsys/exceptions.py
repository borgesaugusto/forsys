class DifferentTissueException(Exception):
    def __init__(self, message='Tissues are too different'):
        super(DifferentTissueException, self).__init__(message)