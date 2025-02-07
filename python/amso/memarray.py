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


class MemArray:
    class DLPackWrapper:
        def __init__(self, cpp_memarray):
            self._cpp_memarray = cpp_memarray

        def __enter__(self):
            self._cpp_memarray._enter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._cpp_memarray._exit()

        def __dlpack__(
            self,
            stream: int | None = None,
            max_version: tuple[int, int] | None = None,
            dl_device: tuple | None = None,
            copy: bool | None = None,
        ):
            if max_version[0] > 1:
                dlpack = self._cpp_memarray._dlpack(True)
            else:
                dlpack = self._cpp_memarray._dlpack(False)

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

    def get_dlpack(self, device: str = "host"):
        """
        To be used only in a context manager.

        ```
        my_array = amso.MemArray([2,3], int)
        with my_array.get_dlpack("host") as dlpack:
           np_array = np.from_dlpack(dlpack)
        ```
        """
        if device == "host":
            self._cpp_obj._to_host()
        elif device == "device":
            self._cpp_obj._to_device()

        return self.DLPackWrapper(self._cpp_obj)

    def read_numpy_array(self, array):
        self._cpp_obj._read_numpy_array(array)
