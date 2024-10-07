
#include <filesystem>
#include <iostream>
#include <fstream>

#include <cassert>

#ifdef _WIN32
#include "vccompat.hpp"
#endif

#include "pocl.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"

/*****************************************************************************/

using namespace std::filesystem;

int
pocl_rm_rf(const char* dir_path)
{
  std::error_code EC;
  std::filesystem::remove_all(path(dir_path), EC);
  return EC ? -1 : 0;
}

int
pocl_mkdir_p (const char* dir_path)
{
  std::error_code EC;
  std::filesystem::create_directories(path(dir_path), EC);
  return EC ? -1 : 0;
}

int
pocl_remove(const char* rm_path)
{
  std::error_code EC;
  std::filesystem::remove(path(rm_path), EC);
  return EC ? -1 : 0;
}

int
pocl_exists(const char* ex_path)
{
  std::error_code EC;
  bool exists = std::filesystem::exists(path(ex_path), EC);
  return exists ? 0 : (EC ? -1 : -2);
}

int 
pocl_touch_file(const char* f_path)
{
  std::error_code EC;
  path P(f_path);
  std::ofstream ofs(P);
  ofs.close();
  bool exists = std::filesystem::exists(P, EC);
  return exists ? 0 : (EC ? -1 : -2);
}

int
pocl_rename(const char *oldpath, const char *newpath) {
  std::error_code EC;
  std::filesystem::rename(path(oldpath), path(newpath), EC);
  return EC ? -1 : 0;
}

/****************************************************************************/

#define CHUNK_SIZE (2 * 1024 * 1024)

int
pocl_read_file(const char* f_path, char** content, uint64_t *filesize)
{
  assert(f_path);
  *content = nullptr;
  *filesize = 0;

  /* files in /proc return zero size, while
     files in /sys return size larger than actual actual content size;
     this reads the content sequentially. */
  size_t total_size = 0;
  char *ptr = (char *)malloc (CHUNK_SIZE + 1);
  if (ptr == nullptr)
    return -1;

  std::error_code EC;
  path P(f_path);
  std::ifstream ifs(P, std::ios::binary);
  if (!ifs) {
    POCL_MSG_ERR ("fopen( %s ) failed\n", f_path);
    goto ERROR;
  }

  do
    {
      char *reallocated = (char *)realloc (ptr, (total_size + CHUNK_SIZE + 1));
      if (reallocated == nullptr)
        goto ERROR;
      ptr = reallocated;

      ifs.read(ptr + total_size, CHUNK_SIZE);
      total_size += ifs.gcount();
    }
  while (ifs.good());

  if (ifs.bad())
    goto ERROR;

  /* add an extra NULL character for strings */
  ptr[total_size] = 0;
  *content = ptr;
  *filesize = (uint64_t)total_size;
  return 0;

ERROR:
  free (ptr);
  return -1;
}



/* Atomic write - with rename()
 * TODO there is no portable fsync/fdatasync in the C++ lib */
int
pocl_write_file (const char *f_path, const char *content, uint64_t count,
                 int append)
{
  assert(f_path);
  assert(content);
  // assert(count); TODO
  int err = 0;
  char f_path2[POCL_MAX_PATHNAME_LENGTH];

  std::error_code EC;
  path P(f_path);
  if (!append) {
    err = pocl_mk_tempname (f_path2, f_path, ".temp", NULL);
    if (err)
      {
        POCL_MSG_ERR ("mktempname(%s) failed\n", f_path);
        return -1;
      }
    P.assign(f_path2);
  }
  std::ofstream ofs(P,
                    append ? (std::ios::binary | std::ios::app)
                           : (std::ios::binary | std::ios::trunc));
  if (!ofs) {
    POCL_MSG_ERR ("fopen( %s ) failed\n", f_path);
    return -1;
  }

  ofs.write(content, count);
  ofs.flush();
  ofs.close();
  // TODO there is no portable fsync/fdatasync
  if (ofs.bad())
    {
      POCL_MSG_ERR ("write(%s) failed\n", f_path);
      return -1;
    }

  if (append)
    return 0;
  else
    return pocl_rename (f_path2, f_path);
}

/****************************************************************************/

/* TODO there is no portable mktemp in the C++ lib */
int
pocl_mk_tempname (char *output, const char *prefix, const char *suffix,
                  int *ret_fd)
{
#if defined(_WIN32)
  char buf[256];
  int ok = GetTempFileName(getenv("TEMP"), prefix, 0, buf);
  return ok ? 0 : 1;
#elif defined(HAVE_MKOSTEMPS) || defined(HAVE_MKSTEMPS) || defined(__ANDROID__)
  /* using mkstemp() instead of tmpnam() has no real benefit
   * here, as we have to pass the filename to llvm,
   * but tmpnam() generates an annoying warning... */
  int fd;

  strncpy (output, prefix, POCL_MAX_PATHNAME_LENGTH);
  size_t len = strlen (prefix);
  strncpy (output + len, "_XXXXXX", (POCL_MAX_PATHNAME_LENGTH - len));

#ifdef __ANDROID__
  fd = pocl_mkstemp (output);
#else
  if (suffix)
    {
      len += 7;
      strncpy (output + len, suffix, (POCL_MAX_PATHNAME_LENGTH - len));
#ifdef HAVE_MKOSTEMPS
      fd = mkostemps (output, strlen (suffix), O_CLOEXEC);
#else
      fd = mkstemps (output, strlen (suffix));
#endif
    }
  else
#ifdef HAVE_MKOSTEMPS
    fd = mkostemp (output, O_CLOEXEC);
#else
    fd = mkstemp (output);
#endif
#endif

  if (fd < 0)
    {
      POCL_MSG_ERR ("mkstemp() failed\n");
      return errno;
    }

  int err = 0;
  if (ret_fd)
    *ret_fd = fd;
  else
    err = close (fd);

  return err ? errno : 0;

#else
#error mkostemps() / mkstemps() both unavailable
#endif
}

int
pocl_mk_tempdir (char *output, const char *prefix)
{
#if defined(_WIN32)
  assert (0);
#elif defined(HAVE_MKDTEMP)
  /* TODO mkdtemp() might not be portable outside Linux */
  strncpy (output, prefix, POCL_MAX_PATHNAME_LENGTH);
  size_t len = strlen (prefix);
  strncpy (output + len, "_XXXXXX", (POCL_MAX_PATHNAME_LENGTH - len));
  return (mkdtemp (output) == NULL);
#else
#error mkdtemp() not available
#endif
}

/* write content[count] into a temporary file, and return the tempfile name in
 * output_path */
int
pocl_write_tempfile (char *output_path, const char *prefix, const char *suffix,
                     const char *content, unsigned long count)
{
  assert (output_path);
  assert (prefix);
  assert (suffix);
  assert (content);

  int fd = -1, err = 0;

  err = pocl_mk_tempname (output_path, prefix, suffix, &fd);
  if (err)
    {
      POCL_MSG_ERR ("pocl_mk_tempname() failed\n");
      return err;
    }

  size_t bytes = count;
  ssize_t res;
  while (bytes > 0)
    {
      res = write (fd, content, bytes);
      if (res < 0)
        {
          POCL_MSG_ERR ("write(%s) failed\n", output_path);
          return errno;
        }
      else
        {
          bytes -= res;
          content += res;
        }
    }

#ifdef HAVE_FDATASYNC
  if (fdatasync (fd))
    {
      POCL_MSG_ERR ("fdatasync() failed\n");
      return errno;
    }
#elif defined(HAVE_FSYNC)
  if (fsync (fd))
    return errno;
#elif defined(_WIN32)
  // TODO get native handle from FD
  // FlushFileBuffers(fh);
#endif

  err = close (fd);

  return err ? -2 : 0;
}
