#ifndef POCL_THREADS_H
#define POCL_THREADS_H

#include "config.h"

#ifdef ENABLE_PLATFORM_CPP

#include "pocl_threads_cpp.hh"

#else

#include "pocl_threads_c.h"

#endif

#endif // POCL_THREADS_H
