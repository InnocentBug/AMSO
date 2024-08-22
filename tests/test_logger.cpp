#include "logger.h"
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>

/**
 * @brief Test fixture for Logger tests.
 */
class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Redirect standard error to a string stream for testing
        oldCerrBuf = std::cerr.rdbuf();
        std::cerr.rdbuf(cerrStream.rdbuf());
    }

    void TearDown() override {
        // Restore standard error
        std::cerr.rdbuf(oldCerrBuf);
    }

    std::stringstream cerrStream;
    std::streambuf* oldCerrBuf;
};

/**
 * @brief Test logging to standard error.
 */
TEST_F(LoggerTest, LogToStandardError) {
    Logger logger("");
    logger.log(Logger::INFO, "Test message");

    std::string output = cerrStream.str();
    EXPECT_NE(output.find("INFO"), std::string::npos);
    EXPECT_NE(output.find("Test message"), std::string::npos);
}

/**
 * @brief Test logging to a file.
 */
TEST_F(LoggerTest, LogToFile) {
    const std::string filename = "test_log.txt";
    Logger logger(filename);
    const std::string msg = "File test message";
    logger.log(Logger::ERROR, msg);

    std::ifstream logFile(filename);
    ASSERT_TRUE(logFile.is_open());

    std::string line;
    std::getline(logFile, line);
    EXPECT_NE(line.find("INFO"), std::string::npos);
    EXPECT_NE(line.find("SOMA starts logging into file "+filename), std::string::npos);

    std::getline(logFile, line);
    std::cout<<"asdf"<<line<<std::endl;
    EXPECT_NE(line.find("ERROR"), std::string::npos);
    EXPECT_NE(line.find(msg), std::string::npos);

    logFile.close();
    std::remove(filename.c_str());  // Clean up the test file
}

/**
 * @brief Test singleton behavior.
 */
TEST_F(LoggerTest, SingletonInstance) {
    Logger& logger1 = Logger::getInstance();
    Logger& logger2 = Logger::getInstance();
    EXPECT_EQ(&logger1, &logger2);
}
