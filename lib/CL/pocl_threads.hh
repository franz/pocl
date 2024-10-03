/* pocl_threads.h - various helper macros & functions for multithreading.

   Copyright (c) 2023 Jan Solanti / Tampere University
                 2024 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

/** \file pocl_threads.h
 *
 * PoCL core should use only the abstractions in this file for threading and
 * synchronization. Thus, this file acts as a portability layer for various
 * threading libraries needed in the runtime.
 */

#ifndef POCL_THREADS_H
#define POCL_THREADS_H

#include "pocl_export.h"

#include <cstdint>

typedef struct _pocl_lock_t *pocl_lock_t;
typedef struct _pocl_cond_t *pocl_cond_t;
typedef struct _pocl_thread_t *pocl_thread_t;

/* These return the new value. */
/* See:
 * https://gcc.gnu.org/onlinedocs/gcc-4.7.4/gcc/_005f_005fatomic-Builtins.html
 */
// TODO

#if defined(__GNUC__) || defined(__clang__)

#define POCL_ATOMIC_ADD(x, val) __atomic_add_fetch (&x, val, __ATOMIC_SEQ_CST);
#define POCL_ATOMIC_INC(x) __atomic_add_fetch (&x, 1, __ATOMIC_SEQ_CST)
#define POCL_ATOMIC_DEC(x) __atomic_sub_fetch (&x, 1, __ATOMIC_SEQ_CST)
#define POCL_ATOMIC_LOAD(x) __atomic_load_n (&x, __ATOMIC_SEQ_CST)
#define POCL_ATOMIC_STORE(x, val) __atomic_store_n (&x, val, __ATOMIC_SEQ_CST)
#define POCL_ATOMIC_CAS(ptr, oldval, newval)                                  \
  __sync_val_compare_and_swap (ptr, oldval, newval)

#elif defined(_WIN32)
#define POCL_ATOMIC_ADD(x, val) InterlockedAdd64 (&x, val);
#define POCL_ATOMIC_INC(x) InterlockedIncrement64 (&x)
#define POCL_ATOMIC_DEC(x) InterlockedDecrement64 (&x)
#define POCL_ATOMIC_LOAD(x) InterlockedOr64 (&x, 0)
#define POCL_ATOMIC_STORE(x, val) InterlockedExchange64 (&x, val)
#define POCL_ATOMIC_CAS(ptr, oldval, newval)                                  \
  InterlockedCompareExchange64 (ptr, newval, oldval)
#else
#error Need atomic_inc() builtin for this compiler
#endif


/* Some pthread_*() calls may return '0' or a specific non-zero value on
 * success.
 */
#define PTHREAD_CHECK2(_status_ok, _code)                                     \
  do                                                                          \
    {                                                                         \
      int _pthread_status = (_code);                                          \
      if (_pthread_status != 0 && _pthread_status != (_status_ok))            \
        pocl_abort_on_pthread_error (_pthread_status, __LINE__,               \
                                     __FUNCTION__);                           \
    }                                                                         \
  while (0)

#define PTHREAD_CHECK(code) PTHREAD_CHECK2 (0, code)

/* Generic functionality for handling different types of
   OpenCL (host) objects. */


#ifdef __cplusplus
extern "C"
{
#endif

void pocl_mutex_init(pocl_lock_t *L);
void pocl_mutex_destroy(pocl_lock_t *L);
void pocl_mutex_lock(pocl_lock_t L);
void pocl_mutex_unlock(pocl_lock_t L);

void pocl_cond_init(pocl_cond_t *C);
void pocl_cond_destroy(pocl_cond_t *C);
void pocl_cond_signal(pocl_cond_t C);
void pocl_cond_broadcast(pocl_cond_t C);
void pocl_cond_wait(pocl_cond_t C, pocl_lock_t L);
void pocl_cond_timedwait(pocl_cond_t C, pocl_lock_t L, uint64_t Timeout);

void pocl_thread_create (pocl_thread_t *T, void* (*F)(void*), void *Arg);
void pocl_thread_destroy (pocl_thread_t *T);
void pocl_thread_join (pocl_thread_t T);

#ifdef __cplusplus
}
#endif

#define POCL_LOCK(__LOCK__) pocl_mutex_lock (__LOCK__)
#define POCL_UNLOCK(__LOCK__) pocl_mutex_unlock (__LOCK__)
#define POCL_INIT_LOCK(__LOCK__) pocl_mutex_init (&__LOCK__)
#define POCL_DESTROY_LOCK(__LOCK__) pocl_mutex_destroy (&__LOCK__)

#define POCL_FAST_LOCK_T POCL_LOCK_T
#define POCL_FAST_LOCK(l) POCL_LOCK(l)
#define POCL_FAST_UNLOCK(l) POCL_UNLOCK(l)
#define POCL_FAST_INIT(l) POCL_INIT_LOCK(l)
#define POCL_FAST_DESTROY(l) POCL_DESTROY_LOCK(l)

#define POCL_INIT_COND(c) pocl_cond_init(&c)
#define POCL_DESTROY_COND(c) pocl_cond_destroy(&c)
#define POCL_SIGNAL_COND(c) pocl_cond_signal(c)
#define POCL_BROADCAST_COND(c) pocl_cond_broadcast(c)
#define POCL_WAIT_COND(c, m) pocl_cond_wait(c, m)
// TODO: should ignore ETIMEDOUT
#define POCL_TIMEDWAIT_COND(c, m, t) pocl_cond_timedwait(c, m, t)

#define POCL_CREATE_THREAD(thr, func, arg) pocl_thread_create(&thr, func, arg)
#define POCL_JOIN_THREAD(thr) pocl_thread_join(thr)
#define POCL_DESTROY_THREAD(thr) pocl_thread_destroy(&thr)

#endif
