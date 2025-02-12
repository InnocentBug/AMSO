#include "memarray.h"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace amso;

template <typename T, int ndim> struct TestParams {
  using Type = T;
  static constexpr int Dimensions = ndim;
};

template <typename TestParam> class ArrayIndexerTest : public ::testing::Test {
protected:
  using T = typename TestParam::Type;
  static constexpr int ndim = TestParam::Dimensions;

  std::array<int, ndim> shape;
  std::unique_ptr<MemArray<T, ndim>> arr;
  std::unique_ptr<typename MemArray<T, ndim>::ArrayIndexer> indexer;

  void SetUp() override {
    shape = generateRandomShape();
    arr = std::make_unique<MemArray<T, ndim>>(shape, 0);
    indexer = std::make_unique<typename MemArray<T, ndim>::ArrayIndexer>(
        arr->get_indexer());
  }

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
    ::testing::Types<TestParams<int, 1>, TestParams<int, 2>, TestParams<int, 3>,
                     TestParams<float, 2>, TestParams<double, 3>>;

TYPED_TEST_SUITE(ArrayIndexerTest, TestTypes);

TYPED_TEST(ArrayIndexerTest, TestIndexerOperations) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;
  const int64_t total_size = this->calculateTotalSize();

  // Test zero index
  std::array<int, ndim> zero_index;
  std::fill(zero_index.begin(), zero_index.end(), 0);
  EXPECT_EQ(this->indexer->operator()(zero_index), 0);

  // Test last index
  std::array<int, ndim> last_index;
  for (int i = 0; i < ndim; ++i) {
    last_index[i] = this->shape[i] - 1;
  }
  EXPECT_EQ(this->indexer->operator()(last_index), total_size - 1);

  // Test random valid indices
  std::random_device rd;
  std::mt19937 gen(rd());
  for (int test = 0; test < 100; ++test) {
    std::array<int, ndim> random_index;
    int64_t expected_index = 0;
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      std::uniform_int_distribution<> dis(0, this->shape[i] - 1);
      random_index[i] = dis(gen);
      expected_index += random_index[i] * stride;
      stride *= this->shape[i];
    }
    EXPECT_EQ(this->indexer->operator()(random_index), expected_index);
  }

  // Test get_indices
  for (int64_t idx = 0; idx < total_size; idx += total_size / 10 + 1) {
    auto indices = this->indexer->get_indices(idx);
    EXPECT_EQ(this->indexer->operator()(indices), idx);
  }
}

TYPED_TEST(ArrayIndexerTest, TestIndexerSize) {
  int64_t expected_size = this->calculateTotalSize();
  EXPECT_EQ(this->indexer->size(), expected_size);
}

TYPED_TEST(ArrayIndexerTest, TestIndexerBoundaries) {
  using T = typename TypeParam::Type;
  constexpr int ndim = TypeParam::Dimensions;

  std::array<int, ndim> index;

  // Test lower boundary
  std::fill(index.begin(), index.end(), 0);
  EXPECT_NO_THROW(this->indexer->operator()(index));

  // Test upper boundary
  for (int i = 0; i < ndim; ++i) {
    index[i] = this->shape[i] - 1;
  }
  EXPECT_NO_THROW(this->indexer->operator()(index));

  // We are not testing for this here. The Indexer does not provide error
  // checking for out of bounds for performance reasons.
  // // Test out of bounds (should assert)
  // index[0] = this->shape[0];
  // EXPECT_DEATH(this->indexer->operator()(index), "");
}
