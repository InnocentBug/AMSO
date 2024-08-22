// SOMA version 2, accelerated Monte-Carlo for many particles in interacting
// fields Copyright (C) 2024 Ludwig Schneider

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
// USA

#pragma once

#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

/**
 * @class Logger
 * @brief A simple logging class that writes log messages to a file or standard
 * error.
 */
class Logger {
public:
  /**
   * @enum LogLevel
   * @brief Defines log levels for the Logger.
   */
  enum LogLevel {
    INFO,    ///< Informational messages
    WARNING, ///< Warning messages
    ERROR    ///< Error messages
  };

  /**
   * @brief Constructs a Logger object.
   * @param filename The name of the file to log messages to. If empty, logs to
   * standard error.
   */
  Logger(const std::string &filename);

  /**
   * @brief Destructor for the Logger.
   */
  ~Logger();

  /**
   * @brief Logs a message with the specified log level.
   * @param level The log level of the message.
   * @param message The message to log.
   */
  void log(LogLevel level, const std::string &message);

  /**
   * @brief Gets the singleton instance of the Logger.
   * @return A reference to the Logger instance.
   */
  static Logger &getInstance(void);

  /**
   * @brief Updates the log file.
   * @param filename The new file to log messages to. If empty, logs to standard
   * error.
   */
  void updateFile(const std::string &filename);

  // Delete copy constructor and assignment operator
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

private:
  std::ofstream logFile; ///< The file stream for logging
  bool useFile;          ///< Flag indicating whether to use a file for logging

  /**
   * @brief Converts a log level to its string representation.
   * @param level The log level to convert.
   * @return The string representation of the log level.
   */
  std::string logLevelToString(LogLevel level);

  /**
   * @brief Gets the current timestamp as a string.
   * @return The current timestamp.
   */
  std::string getTimestamp();
};
