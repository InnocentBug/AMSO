// SOMA version 2, accelerated Monte-Carlo for many particles in interacting fields
// Copyright (C) 2024 Ludwig Schneider

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
#include <thread>
#include "platform_status.h"
#include "platform_status.cuh"

#include "logger.h"

void printPlatformInfo() {
  Logger& logger = Logger::getInstance();
#ifdef ENABLE_CUDA
  cudaPrintPlatformInfo();
#else //ENABLE_CUDA
  const unsigned int cores = std::thread::hardware_concurrency();
  logger.log(Logger::INFO, "CPU platform. "+std::to_string(cores)+" parallel core available.");
#endif//ENABLE_CUDA
}
