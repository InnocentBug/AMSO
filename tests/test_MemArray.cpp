#include "memarray.h"
#include <gtest/gtest.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace amso;

template <typename T, int ndim> struct TestParams {
  using Type = T;
  static constexpr int Dimensions = ndim;
};

template <typename TestParam> class MemArrayTest : public ::testing::Test {
protected:
  using T = typename TestParam::Type;
  static constexpr int ndim = TestParam::Dimensions;
  std::array<int, ndim> shape;
  std::unique_ptr<MemArray<T, ndim>> mem_array;

  void SetUp() override {
    shape = generateRandomShape();
    mem_array = std::make_unique<MemArray<T, ndim>>(shape, 0);
  }

  void TearDown() override {}

  std::array<int, ndim> generateRandomShape() {
    std::array<int, ndim> shape;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    for (int i = 0; i < ndim; ++i) {
      shape[i] = dis(gen);
    }
    return shape;
  }

  int64_t calculateTotalSize() {
    return std::accumulate(shape.begin(), shape.end(), 1LL,
                           std::multiplies<int64_t>());
  }
};

using TestTypes =
    ::testing::Types<TestParams<int64_t, 1>, TestParams<int32_t, 2>,
                     TestParams<int32_t, 3>, TestParams<float, 2>,
                     TestParams<double, 3>>;

TYPED_TEST_SUITE(MemArrayTest, TestTypes);

TYPED_TEST(MemArrayTest, ConstructorTest) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;

  EXPECT_EQ(this->mem_array->get_device_id(), 0);
  EXPECT_EQ(this->mem_array->get_size(), this->calculateTotalSize());
}

TYPED_TEST(MemArrayTest, IndexerTest) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;

  auto indexer = this->mem_array->get_indexer();

  // Test zero index
  std::array<int, ndim> zero_index{};
  EXPECT_EQ(indexer(zero_index), 0);

  // Test last index
  std::array<int, ndim> last_index;
  for (int i = 0; i < ndim; ++i) {
    last_index[i] = this->shape[i] - 1;
  }
  EXPECT_EQ(indexer(last_index), this->calculateTotalSize() - 1);

  // Test random valid indices
  for (int test = 0; test < 100; ++test) {
    std::array<int, ndim> random_index;
    int64_t expected_index = 0;
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      random_index[i] = rand() % this->shape[i];
      expected_index += random_index[i] * stride;
      stride *= this->shape[i];
    }
    EXPECT_EQ(indexer(random_index), expected_index);
  }
}

TYPED_TEST(MemArrayTest, PointerAccessTest) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;

  T *ptr_host = this->mem_array->ptr();
  const T *ptr_const_host = this->mem_array->ptr();

  EXPECT_NE(ptr_host, nullptr);
  EXPECT_NE(ptr_const_host, nullptr);

  if (this->mem_array->get_on_device()) {
    thrust::device_vector<T> device_vec(ptr_host,
                                        ptr_host + this->calculateTotalSize());
    EXPECT_EQ(device_vec.size(), this->calculateTotalSize());
  }
}

TYPED_TEST(MemArrayTest, ContextManagerTest) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;

  // Lock the context manager
  this->mem_array->enter(12345);
  EXPECT_EQ(this->mem_array->get_lock_id(), 12345);

  // Unlock the context manager
  this->mem_array->exit();
  EXPECT_EQ(this->mem_array->get_lock_id(), -1);
}

TYPED_TEST(MemArrayTest, MemoryMovementTest) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;

  // Move memory to device
  this->mem_array->move_memory(kDLCUDA, 0, 0);
  EXPECT_TRUE(this->mem_array->get_on_device());
  this->mem_array->move_memory(kDLCPU, 0, 0);
  EXPECT_FALSE(this->mem_array->get_on_device());
}

TYPED_TEST(MemArrayTest, WriteAndReadHostMemory) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;

  this->mem_array->move_memory(kDLCPU, 0, 0, -1, false);
  // Write data to host memory
  int64_t total_size = this->calculateTotalSize();
  auto indexer = this->mem_array->get_indexer();
  for (int64_t i = 0; i < total_size; ++i) {
    // Use the indexer to calculate multi-dimensional indices
    auto indices = indexer.get_indices(i);
    this->mem_array->write(indices, i);
  }

  // this->mem_array->move_memory(kDLCUDA, 0, 0, -1, false);
  // this->mem_array->move_memory(kDLCPU, 0, 0, -1, false);

  // Read back and verify data from host memory
  for (int64_t i = 0; i < total_size; ++i) {
    auto indices = indexer.get_indices(i);
    EXPECT_EQ((*this->mem_array)(indices), static_cast<T>(i));
  }
}

TYPED_TEST(MemArrayTest, WriteAndReadWithMemoryTransfer) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;

  int64_t total_size = this->calculateTotalSize();
  this->mem_array->move_memory(kDLCPU, 0, 0, -1, false);

  // Step 1: Write data to host memory
  for (int64_t i = 0; i < total_size; ++i) {
    auto indexer = this->mem_array->get_indexer();
    auto indices = indexer.get_indices(i);
    this->mem_array->write(indices, static_cast<T>(i));
  }

  // Step 2: Move memory to the device
  this->mem_array->move_memory(kDLCUDA, 0, 0, -1, false);

  EXPECT_TRUE(
      this->mem_array->get_on_device()); // Verify memory is on the device

  // Step 3: Move memory back to the host
  this->mem_array->move_memory(kDLCPU, 0, 0, -1, false);

  EXPECT_FALSE(
      this->mem_array->get_on_device()); // Verify memory is back on the host

  // Step 4: Read back and verify data from host memory after transfer
  auto indexer = this->mem_array->get_indexer();
  for (int64_t i = 0; i < total_size; ++i) {
    auto indices = indexer.get_indices(i);
    EXPECT_EQ((*this->mem_array)(indices), static_cast<T>(i));
  }
}

TYPED_TEST(MemArrayTest, OutOfBoundsAccess) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;

  auto indexer = this->mem_array->get_indexer();

  // Generate an out-of-bounds index
  std::array<int, ndim> out_of_bounds_index;
  for (int i = 0; i < ndim; ++i) {
    out_of_bounds_index[i] = this->shape[i]; // One past the valid range
  }
  EXPECT_THROW((*this->mem_array)(out_of_bounds_index), std::runtime_error);
  EXPECT_THROW(this->mem_array->write(out_of_bounds_index, 0),
               std::runtime_error);
}

#include "memarray_cuda.cuh"
TYPED_TEST(MemArrayTest, IndexTuple) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;

  this->mem_array->move_memory(kDLCPU, 0, 0, -1, false);
  // Write data to host memory
  int64_t total_size = this->calculateTotalSize();
  auto indexer = this->mem_array->get_indexer();
  for (int64_t i = 0; i < total_size; ++i) {
    // Use the indexer to calculate multi-dimensional indices
    auto indices = indexer.get_indices(i);
    this->mem_array->write(indices, 2);
  }

  add_index_tuple(*(this->mem_array));
  EXPECT_TRUE(this->mem_array->get_on_device());

  // Access without copy to CPU
  for (int64_t i = 0; i < total_size; ++i) {
    // Use the indexer to calculate multi-dimensional indices
    auto indices = indexer.get_indices(i);
    int sum =
        std::accumulate(indices.begin(), indices.end(), 0, std::plus<int>());
    EXPECT_EQ((*this->mem_array)(indices), static_cast<T>(2 + sum));
  }
  EXPECT_TRUE(this->mem_array->get_on_device());

  this->mem_array->move_memory(kDLCPU, 0, 0, -1, false);
  EXPECT_FALSE(this->mem_array->get_on_device());
  // Access after copy to CPU
  for (int64_t i = 0; i < total_size; ++i) {
    // Use the indexer to calculate multi-dimensional indices
    auto indices = indexer.get_indices(i);
    int sum =
        std::accumulate(indices.begin(), indices.end(), 0, std::plus<int>());
    EXPECT_EQ((*this->mem_array)(indices), static_cast<T>(2 + sum));
  }
  EXPECT_FALSE(this->mem_array->get_on_device());
}
