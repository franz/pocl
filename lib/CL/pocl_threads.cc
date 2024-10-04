/* pocl_threads.c - helper functions for thread operations

   Copyright (c) 2023 Jan Solanti / Tampere University

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

#include "pocl_threads_cpp.hh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

//#include "pocl_debug.h"

struct _pocl_barrier_t {
public:
  _pocl_barrier_t(unsigned long ctr) : counter(ctr) {};
  ~_pocl_barrier_t() = default;
  void wait();
private:
   unsigned long counter;
   std::mutex lock;
   std::condition_variable cond;
};

struct _pocl_lock_t {
  std::mutex lock;
};

struct _pocl_cond_t {
  std::condition_variable cond;
};

struct _pocl_thread_t {
  std::thread T;
};

static _pocl_lock_t pocl_init_lock_m;
pocl_lock_t pocl_init_lock = &pocl_init_lock_m;

void pocl_mutex_lock(pocl_lock_t L) {
  L->lock.lock();
}

void pocl_mutex_unlock(pocl_lock_t L) {
  L->lock.unlock();
}

void pocl_mutex_init(pocl_lock_t *L) {
  *L = new _pocl_lock_t;
}

void pocl_mutex_destroy(pocl_lock_t *L) {
  if (*L != nullptr) {
    delete *L;
  }
}

void pocl_cond_init(pocl_cond_t *C) {
  *C = new _pocl_cond_t;
//  POCL_MSG_ERR("@@@@@@@@@@@@ CREATED COND VAR: %p\n", *C);
}

void pocl_cond_destroy(pocl_cond_t *C) {
//  POCL_MSG_ERR("@@@@@@@@@@@@ DESTROY COND VAR: %p\n", *C);
  if (*C != nullptr) {
    delete *C;
  }
  *C = nullptr;
}

void pocl_cond_signal(pocl_cond_t C) {
  C->cond.notify_one();
}

void pocl_cond_broadcast(pocl_cond_t C) {
  C->cond.notify_all();
}

void pocl_cond_wait(pocl_cond_t C, pocl_lock_t L) {
  // the lock is expected to be locked by the user outside this call
  auto UL = std::unique_lock<std::mutex>(L->lock, std::defer_lock);
  C->cond.wait(UL);
}

void pocl_cond_timedwait(pocl_cond_t C, pocl_lock_t L, unsigned long msec) {
  // the lock is expected to be locked by the user outside this call
  auto UL = std::unique_lock<std::mutex>(L->lock, std::defer_lock);
  auto TP = std::chrono::milliseconds(msec);
  C->cond.wait_for(UL, TP);
}

void pocl_thread_create (pocl_thread_t *T, void* (*F)(void*), void *Arg) {
  pocl_thread_t NT = new _pocl_thread_t;
  if (NT)
    NT->T = std::thread(F, Arg);
  *T = NT;
}

void pocl_thread_join (pocl_thread_t T) {
//  assert(T);
  T->T.join();
}

void pocl_thread_destroy (pocl_thread_t *T) {
  if (*T != nullptr) {
    delete *T;
  }
  *T = nullptr;
}

void _pocl_barrier_t::wait() {
  std::unique_lock<std::mutex> L(lock);
  --counter;
  if (counter == 0)
    cond.notify_all();
  while (counter > 0) {
    cond.wait(L);
  }
}

void pocl_barrier_init(pocl_barrier_t *B, unsigned long N) {
  _pocl_barrier_t *L = new _pocl_barrier_t(N);
  *B = L;
}

void pocl_barrier_wait(pocl_barrier_t B) {
//  assert(B);
  B->wait();
}

void pocl_barrier_destroy(pocl_barrier_t *B) {
  if (*B != nullptr) {
    delete *B;
  }
  *B = nullptr;
}
