/* test_printf_vectors.cl - printf tests for cl_khr_fp16 data types

   Copyright (c) 2012-2024 PoCL developers

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

#ifndef cl_khr_fp16
#error cl_khr_fp16 not supported
#endif

kernel void test_printf_vectors_halfn()
{
  printf ("\nhalf VECTORS\n\n");
  ushort2 u2 = (ushort2)(0x7C00, 0x7E00);
  printf("half2  %v2hf\n", (as_half2(u2)));
  printf("half2  %v2hf\n", (half2)(10.112, 20.222));
  printf("half3  %v3hf\n", (half3)(10.113, 20.223, 30.333));
  printf("half4  %v4hf\n", (half4)(10.114, 20.224, 30.334, 40.444));
  printf("half8  %v8hf\n", (half8)(10.118, 20.228, 30.338, 40.448, 50.558, 60.668, 70.778, 80.888));
  printf("half16 %v16hf\n", (half16)(10.11, 20.22, 30.33, 40.44, 50.55, 60.66, 70.77, 80.88, 90.99, 100.1, 110.2, 120.3, 130.4, 140.5, 150.6, 160.7));

  printf ("\nPARAMETER PASSING\n\n");

  printf("\n%c %v2hf |  %v2hf %c\n", '2',
         (half2)(21.1, 21.2),
         (half2)(22.3, 22.4), '.');
  printf("%c %v3hf | %v3hf %c\n", '3',
         (half3)(31.1, 31.2, 31.3),
         (half3)(32.4, 32.5, 32.6), '.');
  printf("%c %v4hf | %v4hf %c\n", '4',
         (half4)(41.1, 41.2, 41.3, 41.4),
         (half4)(42.5, 42.6, 42.7, 42.8), '.');
  printf("%c %v8hf | %v8hf %c\n", '8',
         (half8)(81.01, 81.02, 81.03, 81.04, 81.05, 81.06, 81.07, 81.08),
         (half8)(82.09, 82.10, 82.11, 82.12, 82.13, 82.14, 82.15, 82.16), '.');
  printf("%c %v16hf | %v16hf %c\n", 'F',
         (half16)(1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16),
         (half16)(2.17, 2.18, 2.19, 2.20, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.27, 2.28, 2.29, 2.30, 2.31, 2.32), '.');

  printf("%c %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %c\n", 'd',
         1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8, 16.8,
         17.8, 18.8, 19.8, 20.8, 21.8, 22.8, 23.8, 24.8, 25.8, 26.8, 27.8, 28.8, 29.8, 30.8, 31.8, 32.8, '.');
}
