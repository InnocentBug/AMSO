import amso
import numpy as np
import pytest
from amso import dlpack as amso_dlpack


def get_init_array(shape, dtype):
    total_size = 1
    for length in shape:
        total_size *= length

    np_arr = np.zeros(total_size, dtype=dtype)
    for i in range(total_size):
        np_arr[i] = i
    np_arr = np_arr.reshape(shape)
    return np_arr


def get_numpy_ref_array(shape, dtype):
    np_arr = get_init_array(shape, dtype)

    if len(shape) == 1:
        for x in range(shape[0]):
            np_arr[x] += x
    elif len(shape) == 2:
        for x in range(shape[0]):
            for y in range(shape[1]):
                np_arr[x, y] += x + y
    elif len(shape) == 3:
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    np_arr[x, y, z] += x + y + z
    else:
        raise RuntimeError("Test only valid for dimension, 1, 2, 3")

    return np_arr


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (11,),
        (1, 1),
        (12, 13),
        (
            1,
            1,
            1,
        ),
        (4, 5, 6),
        (134, 126, 180),
    ],
)
@pytest.mark.parametrize(
    "dtype", [int, float, np.int32, np.int64, np.float32, np.float64]
)
def test_memarray(shape, dtype):
    mem_array = amso.MemArray(shape, dtype)

    with mem_array.get_dlpack() as dlpack:
        local_array = np.from_dlpack(dlpack)
        assert local_array.shape == shape
        assert local_array.dtype == dtype

    mem_array.move_memory(amso_dlpack.kDLCUDA)

    with mem_array.get_dlpack(amso_dlpack.kDLCPU) as dlpack:
        dlpack_array = np.from_dlpack(dlpack)
        assert dlpack_array.shape == shape

    mem_array.move_memory(amso_dlpack.kDLCUDA)
    init_array = get_init_array(shape, dtype)

    mem_array.read_numpy_array(init_array)
    mem_array.move_memory(amso_dlpack.kDLCUDA)

    # Run the CUDA kernel
    mem_array.add_index_tuple()
    mem_array.move_memory(amso_dlpack.kDLCPU)

    # Access the result and transfer it to CPU
    with mem_array.get_dlpack(amso_dlpack.kDLCPU) as dlpack:
        dlpack_array = np.from_dlpack(dlpack)

        # Do the same, what you expect the CUDA kernel to in numpy
        ref_array = get_numpy_ref_array(shape, dtype)
        assert np.allclose(dlpack_array, ref_array)
