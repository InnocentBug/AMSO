import uuid

import numpy as np

from ._amso import (
    MemArray1DDouble,
    MemArray1DFloat,
    MemArray1DInt32,
    MemArray1DInt64,
    MemArray2DDouble,
    MemArray2DFloat,
    MemArray2DInt32,
    MemArray2DInt64,
    MemArray3DDouble,
    MemArray3DFloat,
    MemArray3DInt32,
    MemArray3DInt64,
)
from ._amso import dlpack as amso_dlpack


class MemArray:
    class DLPackWrapper:
        def __init__(self, lock_id: int, cpp_memarray):
            lock_id = lock_id % (2**63 - 1)  # Int64 max to avoid bad C++ values
            if lock_id < 0:
                lock_id *= -1

            self._lock_id: int = lock_id
            self._cpp_memarray = cpp_memarray

        def __enter__(self):
            self._cpp_memarray._enter(self._lock_id)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._cpp_memarray._exit()

        def __dlpack__(
            self,
            stream: int | None = None,
            max_version: tuple[int, int] | None = None,
            dl_device: tuple | None = None,
            copy: bool | None = None,
        ):  # We are trying to implement https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html as close as possible.

            if dl_device is not None:
                current_device_type, current_device_id = (
                    self._cpp_memarray._dlpack_device()
                )
                requested_device_type, requested_device_id = dl_device
                if current_device_id != requested_device_id:
                    raise BufferError(
                        f"Memarray DLPack Request to a different device. Currently not supported requested {requested_device_id} current {current_device_id}"
                    )
                if not int(current_device_type) == int(
                    requested_device_type
                ):  # We have to compare as int, since the request might be a different Enum definition.
                    if requested_device_type == amso_dlpack.kDLCPU:
                        self._cpp_memarray._to_host(self._lock_id)
                    elif requested_device_type == amso_dlpack.kDLCUDA:
                        self._cpp_memarray._to_device(self._lock_id)
                    else:
                        raise BufferError(
                            f"Unable to convert Array to Device type {requested_device_type}. Only supported are {amso_dlpack.kDLCPU} and {amso_dlpack.kDLCUDA}."
                        )

            if max_version is None:
                dlpack = self._cpp_memarray._dlpack(False, self._lock_id)
            else:
                if max_version[0] > amso_dlpack.DLPACK_MAJOR_VERSION:
                    dlpack = self._cpp_memarray._dlpack(True, self._lock_id)
                else:
                    dlpack = self._cpp_memarray._dlpack(False, self._lock_id)

            return dlpack

        def __dlpack_device__(self):
            return self._cpp_memarray._dlpack_device()

    # On the C++ side we instanciate only certain types from the template.
    # This map maps the user arguments for types and ndim to the correct instance
    _TYPE_INSTANCE_MAP = {
        (np.dtype(np.int32), 1): MemArray1DInt32,
        (np.dtype(np.int32), 2): MemArray2DInt32,
        (np.dtype(np.int32), 3): MemArray3DInt32,
        (np.dtype(np.int64), 1): MemArray1DInt64,
        (np.dtype(np.int64), 2): MemArray2DInt64,
        (np.dtype(np.int64), 3): MemArray3DInt64,
        (np.dtype(np.float32), 1): MemArray1DFloat,
        (np.dtype(np.float32), 2): MemArray2DFloat,
        (np.dtype(np.float32), 3): MemArray3DFloat,
        (np.dtype(np.float64), 1): MemArray1DDouble,
        (np.dtype(np.float64), 2): MemArray2DDouble,
        (np.dtype(np.float64), 3): MemArray3DDouble,
    }

    def __init__(
        self,
        shape: list[int] | np.ndarray,
        dtype: np.dtype | None = None,
        device_id: int = 0,
    ):
        if dtype is None:
            dtype = np.dtype(float)

        self._dtype = np.dtype(dtype)
        self._shape = [int(element) for element in shape]
        self._ndim = len(self._shape)

        try:
            self._cpp_type = self._TYPE_INSTANCE_MAP[(self._dtype, self._ndim)]
        except KeyError as exc:
            raise RuntimeError(
                f"The compiled C++ AMSO only supports a number of predefined types and ndim. You requested {self._dtype} and {self._ndim} ({self._shape}), but AMSO only knows {self._TYPE_INSTANCE_MAP.keys()}. If you need your types, consider adding more instanciaces of the C++-template, and add this type combination to the map."
            ) from exc

        self._cpp_obj = self._cpp_type(self._shape, device_id)

    def get_dlpack(self):
        """
        To be used only in a context manager.

        ```
        my_array = amso.MemArray([2,3], int)
        with my_array.get_dlpack() as dlpack:
           np_array = np.from_dlpack(dlpack)
        ```
        """
        return self.DLPackWrapper(int(uuid.uuid4()), self._cpp_obj)

    def read_numpy_array(self, array):
        self._cpp_obj._read_numpy_array(array)

    def to_device(self, device_type: amso_dlpack):
        if device_type == amso_dlpack.kDLCPU:
            self._cpp_obj._to_host(-1)
        elif device_type == amso_dlpack.kDLCUDA:
            self._cpp_obj._to_device(-1)
        else:
            raise RuntimeError(f"Requested device {device_type} not supported.")
