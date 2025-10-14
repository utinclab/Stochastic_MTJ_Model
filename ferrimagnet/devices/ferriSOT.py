from devices.base import DeviceBase
from parameters.ferri_params import FerriParameters

class FerrimagnetMTJ(DeviceBase):
    """Ferrimagnetic MTJ device model."""

    def __init__(self, params: FerriParameters, device_id=0):
        super().__init__(params, device_id)
