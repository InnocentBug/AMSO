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

#include "logger.h"

Logger::Logger(const std::string &filename = "") : useFile(false) {
  updateFile(filename);
}

void Logger::updateFile(const std::string &filename) {
  std::string end_msg = "SOMA stops logging in this stream and will log into ";
  if (filename.empty()) {
    end_msg += "standard error stream now.";
  } else {
    end_msg += "the file " + filename + " now.";
  }
  log(Logger::INFO, end_msg);

  if (useFile && logFile.is_open()) {
    logFile.close();
  }
  useFile = !filename.empty();

  if (useFile) {
    logFile.open(filename, std::ios_base::app);
    if (!logFile.is_open()) {
      throw std::runtime_error("Unable to open log file");
    }
    log(Logger::INFO, "SOMA starts logging into file " + filename);
  } else {
    log(Logger::INFO, "SOMA starts logging into standard error stream.");
  }
}

Logger &Logger::getInstance(void) {
  static Logger instance;
  return instance;
}

Logger::~Logger() {
  if (logFile.is_open()) {
    logFile.close();
  }
}

void Logger::log(LogLevel level, const std::string &message) {
  std::string logLevelStr = logLevelToString(level);
  std::string timestamp = getTimestamp();
  std::string logEntry = "[" + timestamp + "] [" + logLevelStr + "] " + message;

  if (useFile) {
    logFile << logEntry << std::endl;
  } else {
    std::cerr << logEntry << std::endl;
  }
}

std::string Logger::logLevelToString(LogLevel level) {
  switch (level) {
  case INFO:
    return "INFO";
  case WARNING:
    return "WARNING";
  case ERROR:
    return "ERROR";
  default:
    return "UNKNOWN";
  }
}

std::string Logger::getTimestamp() {
  std::time_t now = std::time(nullptr);
  std::tm *localTime = std::localtime(&now);
  std::stringstream ss;
  ss << (localTime->tm_year + 1900) << '-' << (localTime->tm_mon + 1) << '-'
     << localTime->tm_mday << ' ' << localTime->tm_hour << ':'
     << localTime->tm_min << ':' << localTime->tm_sec;
  return ss.str();
}
