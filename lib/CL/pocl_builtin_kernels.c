
#include "pocl_builtin_kernels.h"

#include <string.h>

/* initializers for kernel arguments */
#define BI_ARG_FULL(TYPENAME, NAME, TYPE, ADQ, ACQ, TQ, SIZE) \
(const pocl_argument_info){ .type_name = TYPENAME, .name = NAME,  \
  .address_qualifier = ADQ, .access_qualifier = ACQ, .type_qualifier = TQ, \
  .type = TYPE, .type_size = SIZE }

#define BI_ARG(TYPENAME, NAME, TYPE) \
(const pocl_argument_info){ .name = NAME, .type_name = TYPENAME, .type = TYPE, \
  .address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL, \
  .access_qualifier = CL_KERNEL_ARG_ACCESS_NONE, \
  .type_qualifier = CL_KERNEL_ARG_TYPE_NONE, \
  .type_size = 0 }

/* initializers for builtin kernel */
#define BIKD_FULL(ID, NAME, IS_DBK, LOCAL_SIZE, ARGUMENTS...) \
(const pocl_kernel_metadata_t){ \
  .num_args = (sizeof((const pocl_argument_info[]){ARGUMENTS}) / sizeof(pocl_argument_info)), \
  .num_locals = ((LOCAL_SIZE > 0) ? 1 : 0), \
  .local_sizes = (size_t[]){LOCAL_SIZE}, \
  .name = NAME, .attributes = NULL, \
  .arg_info = (pocl_argument_info[]){ARGUMENTS}, \
  .has_arg_metadata = POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER | POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER | POCL_HAS_KERNEL_ARG_TYPE_NAME | POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER | POCL_HAS_KERNEL_ARG_NAME,\
  .reqd_wg_size = {0, 0, 0}, \
  .wg_size_hint = {0, 0, 0}, \
  .vectypehint = {0}, \
  .total_argument_storage_size = 0, \
  .max_subgroups = 0, .compile_subgroups = 0,\
  .max_workgroup_size = NULL, .preferred_wg_multiple = NULL, \
  .local_mem_size = NULL, .private_mem_size = NULL, .spill_mem_size = NULL, \
  .build_hash = NULL, \
  .builtin_kernel = (IS_DBK ? POCL_DBK : POCL_BIK), \
  .builtin_kernel_id = ID, \
  .builtin_max_global_work = {0, 0, 0}, \
  .data = NULL \
}

// BIKD for non-DBK
#define BIKD(ID, NAME, ...) \
  BIKD_FULL(ID, NAME, 0, 0, __VA_ARGS__)

// BIKD for DBK
#define BIKD_DBK(ID, NAME, ...) \
  BIKD_FULL(ID, NAME, 1, 0, __VA_ARGS__)

// BIKD with nonzero local size
#define BIKD_LOCAL(ID, NAME, LOCAL_SIZE, ...) \
  BIKD_FULL(ID, NAME, 0, LOCAL_SIZE, __VA_ARGS__)

// Shortcut handles to make the descriptor list more compact.
#define BI_ARG_READ_BUF(TYPENAME, NAME)                                       \
  BI_ARG_FULL(TYPENAME, NAME, POCL_ARG_TYPE_POINTER,                          \
       CL_KERNEL_ARG_ADDRESS_GLOBAL,  CL_KERNEL_ARG_ACCESS_NONE,              \
      (CL_KERNEL_ARG_TYPE_CONST | CL_KERNEL_ARG_TYPE_RESTRICT), 0)

#define BI_ARG_WRITE_BUF(TYPENAME, NAME)                                      \
  BI_ARG_FULL(TYPENAME, NAME, POCL_ARG_TYPE_POINTER,                          \
       CL_KERNEL_ARG_ADDRESS_GLOBAL,                                          \
      CL_KERNEL_ARG_ACCESS_NONE, CL_KERNEL_ARG_TYPE_RESTRICT, 0)

#define BI_ARG_POD_WITH_ATTRS(TYPENAME, NAME, SIZE)                                  \
  BI_ARG_FULL(TYPENAME, NAME, POCL_ARG_TYPE_NONE,  \
    CL_KERNEL_ARG_ADDRESS_PRIVATE,                           \
      CL_KERNEL_ARG_ACCESS_NONE, CL_KERNEL_ARG_TYPE_NONE, SIZE)

#define BI_ARG_POD(TYPENAME, NAME, NUM_BITS) \
  BI_ARG_POD_WITH_ATTRS(TYPENAME, NAME, ((NUM_BITS + 7u) / 8u))

#define BI_ARG_POD_32b(TYPENAME, NAME) BI_ARG_POD(TYPENAME, NAME, 32)

#define BI_ARG_READ_PIPE(TYPENAME, NAME)                                    \
  BI_ARG_FULL(TYPENAME, NAME, POCL_ARG_TYPE_PIPE, \
    CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE, \
    CL_KERNEL_ARG_TYPE_NONE, 4)

#define BI_ARG_WRITE_PIPE(TYPENAME, NAME)                                      \
  BI_ARG_FULL(TYPENAME, NAME, POCL_ARG_TYPE_PIPE,                   \
    CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE, \
    CL_KERNEL_ARG_TYPE_NONE, 4)



pocl_kernel_metadata_t pocl_BIDescriptors[BIKERNELS];

/* C standards older than C17 refuse to initialize module-scope variables
 * even with const values as they don't consider them "constant".
 * The same restrictions don't apply to initializing stack variables.
 * This function is a workaround to make initialization work with C11/C99 */
void pocl_init_builtin_kernel_metadata() {
  pocl_kernel_metadata_t temporary_BIDescriptors[BIKERNELS] = {
  BIKD(POCL_CDBI_ADD_I32, "pocl.add.i32",
             BI_ARG_READ_BUF("int*", "input1"),
             BI_ARG_READ_BUF("int*", "input2"),
             BI_ARG_WRITE_BUF("int*", "output"),
          ),
    BIKD(POCL_CDBI_MUL_I32, "pocl.mul.i32",
           BI_ARG_READ_BUF("int*", "input1"),
           BI_ARG_READ_BUF("int*", "input2"),
           BI_ARG_WRITE_BUF("int*", "output"),
          ),
    BIKD(POCL_CDBI_LEDBLINK, "pocl.ledblink",
          BI_ARG_READ_BUF("int*", "input1"),
          BI_ARG_READ_BUF("int*", "input2"),
        ),
    BIKD(POCL_CDBI_COUNTRED, "pocl.countred",
          BI_ARG_READ_BUF("int*", "input"),
          BI_ARG_WRITE_BUF("int*", "output"),
        ),
    BIKD(POCL_CDBI_DNN_CONV2D_RELU_I8, "pocl.dnn.conv2d.relu.i8",
             BI_ARG_READ_BUF("char*", "input"),
             BI_ARG_READ_BUF("char*", "weights"),
             BI_ARG_WRITE_BUF("char*", "output"),
             BI_ARG_READ_BUF("int*", "bias"),
             BI_ARG_READ_BUF("int*", "scale"),
             BI_ARG_READ_BUF("int*", "shift"),
             BI_ARG_READ_BUF("char*", "zero_point"),
             BI_ARG_POD_32b("unsigned", "window_size_x"),
             BI_ARG_POD_32b("unsigned", "window_size_y"),
             BI_ARG_POD_32b("unsigned", "stride_x"),
             BI_ARG_POD_32b("unsigned", "stride_y"),
             BI_ARG_POD_32b("unsigned", "input_depth"),
         ),
    BIKD_LOCAL(POCL_CDBI_SGEMM_LOCAL_F32, "pocl.sgemm.local.f32",
         (2 * 16 * 16 * 4), // local mem size, 2 float arrays 16x16
             BI_ARG_READ_BUF("float*", "A"),
             BI_ARG_READ_BUF("float*", "B"),
             BI_ARG_WRITE_BUF("float*", "C"),
             BI_ARG_POD_32b("unsigned", "M"),
             BI_ARG_POD_32b("unsigned", "N"),
             BI_ARG_POD_32b("unsigned", "K"),
         ),
    BIKD(POCL_CDBI_SGEMM_TENSOR_F16F16F32_SCALE,
         "pocl.sgemm.scale.tensor.f16f16f32",
             BI_ARG_READ_BUF("half*", "A"),
             BI_ARG_READ_BUF("half*", "B"),
             BI_ARG_WRITE_BUF("float*", "C"),
             BI_ARG_POD_32b("unsigned", "M"),
             BI_ARG_POD_32b("unsigned", "N"),
             BI_ARG_POD_32b("unsigned", "K"),
             BI_ARG_POD_32b("float", "alpha"),
             BI_ARG_POD_32b("float", "beta"),
         ),
    BIKD(POCL_CDBI_SGEMM_TENSOR_F16F16F32, "pocl.sgemm.tensor.f16f16f32",
             BI_ARG_READ_BUF("half*", "A"),
             BI_ARG_READ_BUF("half*", "B"),
             BI_ARG_WRITE_BUF("float*", "C"),
             BI_ARG_POD_32b("unsigned", "M"),
             BI_ARG_POD_32b("unsigned", "N"),
             BI_ARG_POD_32b("unsigned", "K"),
         ),
    BIKD(POCL_CDBI_ABS_F32, "pocl.abs.f32",

             BI_ARG_READ_BUF("float*", "input"),
             BI_ARG_WRITE_BUF("float*", "output"),
          ),
    BIKD(POCL_CDBI_DNN_DENSE_RELU_I8, "pocl.dnn.dense.relu.i8",

             BI_ARG_READ_BUF("char*", "input"),
             BI_ARG_READ_BUF("char*", "weights"),
             BI_ARG_WRITE_BUF("char*", "output"),
             BI_ARG_READ_BUF("int*", "bias"),
             BI_ARG_POD_32b("unsigned", "scale"),
             BI_ARG_POD_32b("unsigned", "shift"),
             BI_ARG_POD_32b("unsigned", "zero_point"),
             BI_ARG_POD_32b("unsigned", "output_minus"),
             BI_ARG_POD_32b("unsigned", "input_size"),
         ),
    BIKD(POCL_CDBI_MAXPOOL_I8, "pocl.maxpool.i8",

             BI_ARG_READ_BUF("char*", "input"),
             BI_ARG_WRITE_BUF("char*", "output"),
             BI_ARG_POD_32b("unsigned", "window_size_x"),
             BI_ARG_POD_32b("unsigned", "window_size_y"),
             BI_ARG_POD_32b("unsigned", "stride_x"),
             BI_ARG_POD_32b("unsigned", "stride_y"),
         ),
    BIKD(POCL_CDBI_ADD_I8, "pocl.add.i8",

             BI_ARG_READ_BUF("char*", "input1"),
             BI_ARG_READ_BUF("char*", "input2"),
             BI_ARG_WRITE_BUF("char*", "output"),
         ),
    BIKD(POCL_CDBI_MUL_I8, "pocl.mul.i8",

             BI_ARG_READ_BUF("char*", "input1"),
             BI_ARG_READ_BUF("char*", "input2"),
             BI_ARG_WRITE_BUF("char*", "output"),
         ),
    BIKD(POCL_CDBI_ADD_I16, "pocl.add.i16",

             BI_ARG_READ_BUF("short*", "input1"),
             BI_ARG_READ_BUF("short*", "input2"),
             BI_ARG_WRITE_BUF("short*", "output"),
         ),
    BIKD(POCL_CDBI_MUL_I16, "pocl.mul.i16",

             BI_ARG_READ_BUF("short*", "input1"),
             BI_ARG_READ_BUF("short*", "input2"),
             BI_ARG_WRITE_BUF("short*", "output"),
         ),
    BIKD(POCL_CDBI_STREAMOUT_I32, "pocl.streamout.i32",

             BI_ARG_READ_BUF("int*", "output"),
         ),
    BIKD(POCL_CDBI_STREAMIN_I32, "pocl.streamin.i32",

             BI_ARG_WRITE_BUF("int*", "output"),
         ),
    BIKD(POCL_CDBI_VOTE_U32, "pocl.vote.u32",

           BI_ARG_READ_BUF("int*", "output"),
           BI_ARG_POD_32b("unsigned", "num_inputs"),
           BI_ARG_READ_BUF("int*", "input0"),
           BI_ARG_READ_BUF("int*", "input1"),
           BI_ARG_READ_BUF("int*", "input2"),
           BI_ARG_READ_BUF("int*", "input3"),
           BI_ARG_READ_BUF("int*", "input4"),
           BI_ARG_READ_BUF("int*", "input5"),
           BI_ARG_READ_BUF("int*", "input6"),
           BI_ARG_READ_BUF("int*", "input7"),
         ),
    BIKD(POCL_CDBI_VOTE_U8, "pocl.vote.u8",

           BI_ARG_READ_BUF("char*", "output"),
           BI_ARG_POD_32b("unsigned", "num_inputs"),
           BI_ARG_READ_BUF("char*", "input0"),
           BI_ARG_READ_BUF("char*", "input1"),
           BI_ARG_READ_BUF("char*", "input2"),
           BI_ARG_READ_BUF("char*", "input3"),
           BI_ARG_READ_BUF("char*", "input4"),
           BI_ARG_READ_BUF("char*", "input5"),
           BI_ARG_READ_BUF("char*", "input6"),
           BI_ARG_READ_BUF("char*", "input7"),
          ),
   BIKD(POCL_CDBI_DNN_CONV2D_NCHW_F32, "pocl.dnn.conv2d.nchw.f32",

             BI_ARG_READ_BUF("float*", "input"),
             BI_ARG_READ_BUF("float*", "weights"),
             BI_ARG_WRITE_BUF("float*", "output"),
             BI_ARG_POD_32b("int", "input_n"),
             BI_ARG_POD_32b("int", "input_c"),
             BI_ARG_POD_32b("int", "input_h"),
             BI_ARG_POD_32b("int", "input_w"),
             BI_ARG_POD_32b("int", "filt_k"),
             BI_ARG_POD_32b("int", "filt_c"),
             BI_ARG_POD_32b("int", "filt_h"),
             BI_ARG_POD_32b("int", "filt_w"),
             BI_ARG_POD_32b("int", "stride_h"),
             BI_ARG_POD_32b("int", "stride_w"),
             BI_ARG_POD_32b("int", "dilation_h"),
             BI_ARG_POD_32b("int", "dilation_w"),
             BI_ARG_POD_32b("int", "padding_h"),
             BI_ARG_POD_32b("int", "padding_w"),
             BI_ARG_POD_32b("int", "groups"),
             BI_ARG_POD_32b("float", "alpha"),
             BI_ARG_POD_32b("float", "beta"),
         ),
    BIKD(POCL_CDBI_OPENVX_SCALEIMAGE_NN_U8,
         "org.khronos.openvx.scale_image.nn.u8",

             BI_ARG_READ_BUF("unsigned char*", "input"),
             BI_ARG_WRITE_BUF("unsigned char*", "output"),
             BI_ARG_POD_32b("float", "width_scale"),
             BI_ARG_POD_32b("float", "height_scale"),
             BI_ARG_POD_32b("int", "input_width"),
             BI_ARG_POD_32b("int", "input_height"),
         ),
    BIKD(POCL_CDBI_OPENVX_SCALEIMAGE_BL_U8,
         "org.khronos.openvx.scale_image.bl.u8",

             BI_ARG_READ_BUF("unsigned char*", "input"),
             BI_ARG_WRITE_BUF("unsigned char*", "output"),
             BI_ARG_POD_32b("float", "width_scale"),
             BI_ARG_POD_32b("float", "height_scale"),
             BI_ARG_POD_32b("int", "input_width"),
             BI_ARG_POD_32b("int", "input_height"),
         ),
    BIKD(POCL_CDBI_OPENVX_TENSORCONVERTDEPTH_WRAP_U8_F32,
         "org.khronos.openvx.tensor_convert_depth.wrap.u8.f32",

             BI_ARG_READ_BUF("unsigned char*", "input"),
             BI_ARG_WRITE_BUF("float*", "output"),
             BI_ARG_POD_32b("float", "norm"),
             BI_ARG_POD_32b("float", "offset"),
         ),
    BIKD(POCL_CDBI_OPENVX_MINMAXLOC_R1_U8,
         "org.khronos.openvx.minmaxloc.r1.u8",

             BI_ARG_READ_BUF("unsigned char*", "input"),
             BI_ARG_WRITE_BUF("unsigned char*", "min"),
             BI_ARG_WRITE_BUF("unsigned char*", "max"),
             BI_ARG_WRITE_BUF("unsigned int*", "minloc"),
             BI_ARG_WRITE_BUF("unsigned int*", "maxloc"),
         ),
    BIKD(POCL_CDBI_SOBEL3X3_U8,
         "pocl.sobel3x3.u8",

             BI_ARG_READ_BUF("unsigned char*", "input"),
             BI_ARG_WRITE_BUF("unsigned short*", "sobel_x"),
             BI_ARG_WRITE_BUF("unsigned short*", "sobel_y"),
         ),
    BIKD(POCL_CDBI_PHASE_U8,
         "pocl.phase.u8",

             BI_ARG_READ_BUF("unsigned short*", "in_x"),
             BI_ARG_READ_BUF("unsigned short*", "in_y"),
             BI_ARG_WRITE_BUF("unsigned char*", "output"),
         ),
    BIKD(POCL_CDBI_MAGNITUDE_U16,
         "pocl.magnitude.u16",

             BI_ARG_READ_BUF("unsigned short*", "in_x"),
             BI_ARG_READ_BUF("unsigned short*", "in_y"),
             BI_ARG_WRITE_BUF("unsigned short*", "output"),
         ),
    BIKD(POCL_CDBI_ORIENTED_NONMAX_U16,
         "pocl.oriented.nonmaxsuppression.u16",

             BI_ARG_READ_BUF("unsigned short*", "magnitude"),
             BI_ARG_READ_BUF("unsigned char*", "phase"),
             BI_ARG_WRITE_BUF("unsigned char*", "output"),
             BI_ARG_POD_32b("unsigned short", "threshold_lower"),
             BI_ARG_POD_32b("unsigned short", "threshold_upper"),
         ),
    BIKD(POCL_CDBI_CANNY_U8,
         "pocl.canny.u8",

             BI_ARG_READ_BUF("unsigned char*", "input"),
             BI_ARG_WRITE_BUF("unsigned char*", "output"),
             BI_ARG_POD_32b("unsigned short", "threshold_lower"),
             BI_ARG_POD_32b("unsigned short", "threshold_upper"),
         ),
    BIKD(POCL_CDBI_STREAM_MM2S_P512,
         "pocl.stream.mm2s.p512",

             BI_ARG_READ_BUF("char*", "in"),
             BI_ARG_WRITE_PIPE("uchar64", "out"),
         ),
    BIKD(POCL_CDBI_STREAM_S2MM_P512,
         "pocl.stream.s2mm.p512",

             BI_ARG_READ_PIPE("uchar64", "in"),
             BI_ARG_WRITE_BUF("char*", "out"),
         ),
    BIKD(POCL_CDBI_BROADCAST_1TO2_P512,
         "pocl.broadcast.1to2.p512",

             BI_ARG_READ_PIPE("uchar64", "in"),
             BI_ARG_WRITE_PIPE("uchar64", "out0"),
             BI_ARG_WRITE_PIPE("uchar64", "out1"),
         ),
    BIKD(POCL_CDBI_SOBEL3X3_P512,
         "pocl.sobel3x3.p512",

             BI_ARG_READ_PIPE("uchar64", "in"),
             BI_ARG_WRITE_PIPE("short32", "sobel_x"),
             BI_ARG_WRITE_PIPE("short32", "sobel_y"),
         ),
    BIKD(POCL_CDBI_PHASE_P512,
         "pocl.phase.p512",

             BI_ARG_READ_PIPE("short32", "in_x"),
             BI_ARG_READ_PIPE("short32", "in_y"),
             BI_ARG_WRITE_PIPE("uchar64", "output"),
         ),
    BIKD(POCL_CDBI_MAGNITUDE_P512,
         "pocl.magnitude.p512",

             BI_ARG_READ_PIPE("short32", "in_x"),
             BI_ARG_READ_PIPE("short32", "in_y"),
             BI_ARG_WRITE_PIPE("ushort32", "output"),
         ),
    BIKD(POCL_CDBI_ORIENTED_NONMAX_P512,
         "pocl.oriented.nonmaxsuppression.p512",

             BI_ARG_READ_PIPE("ushort32", "magnitude"),
             BI_ARG_READ_PIPE("uchar64", "phase"),
             BI_ARG_WRITE_PIPE("uchar64", "output"),
             BI_ARG_POD_32b("unsigned short", "threshold_lower"),
             BI_ARG_POD_32b("unsigned short", "threshold_upper"),
         ),
    BIKD(POCL_CDBI_GAUSSIAN3X3_P512,
           "pocl.gaussian3x3.p512",
           BI_ARG_READ_PIPE("uchar64", "in"),
           BI_ARG_WRITE_PIPE("uchar64", "out"),
         ),
    BIKD_DBK(POCL_CDBI_DBK_KHR_GEMM,
           "khr_gemm",
           // The types are placeholders
           BI_ARG_READ_BUF("unsigned char*", "a"),
           BI_ARG_READ_BUF("unsigned char*", "b"),
           BI_ARG_READ_BUF("unsigned char*", "c_in"),
           BI_ARG_WRITE_BUF("unsigned char*", "c_out"),
         ),
    BIKD_DBK(POCL_CDBI_DBK_KHR_MATMUL,
           "khr_matmul",
           BI_ARG_READ_BUF("unsigned char*", "a"),
           BI_ARG_READ_BUF("unsigned char*", "b"),
           BI_ARG_WRITE_BUF("unsigned char*", "c"),
         ),
  };
  memcpy(pocl_BIDescriptors, temporary_BIDescriptors,
         sizeof(pocl_BIDescriptors));
}


/* creates a deep copy of pocl_kernel_metadata_t in 'target' */
static cl_int pocl_clone_builtin_kernel_metadata(cl_device_id dev,
                                                 const char *kernel_name,
                                                 pocl_kernel_metadata_t *target) {

  pocl_kernel_metadata_t *Desc = NULL;
  for (size_t i = 0; i < BIKERNELS; ++i) {
    Desc = &pocl_BIDescriptors[i];
    if (strcmp(Desc->name, kernel_name) == 0) {
      memcpy(target, (pocl_kernel_metadata_t *)Desc,
             sizeof(pocl_kernel_metadata_t));
      target->name = strdup(Desc->name);
      target->arg_info = (struct pocl_argument_info *)calloc(
          Desc->num_args, sizeof(struct pocl_argument_info));
      memset(target->arg_info, 0,
             sizeof(struct pocl_argument_info) * Desc->num_args);
      for (unsigned Arg = 0; Arg < Desc->num_args; ++Arg) {
        memcpy(&target->arg_info[Arg], &Desc->arg_info[Arg],
               sizeof(pocl_argument_info));
        target->arg_info[Arg].name = strdup(Desc->arg_info[Arg].name);
        target->arg_info[Arg].type_name = strdup(Desc->arg_info[Arg].type_name);
        if (target->arg_info[Arg].type == POCL_ARG_TYPE_POINTER ||
            target->arg_info[Arg].type == POCL_ARG_TYPE_IMAGE)
          target->arg_info[Arg].type_size = sizeof (cl_mem);
      }
      target->builtin_max_global_work = Desc->builtin_max_global_work;
      target->has_arg_metadata =
        POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER |
        POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER  |
        POCL_HAS_KERNEL_ARG_TYPE_NAME         |
        POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER    |
        POCL_HAS_KERNEL_ARG_NAME;
    }
  }
  return 0;
}

int pocl_setup_builtin_metadata(cl_device_id device, cl_program program,
                                unsigned program_device_i) {
  if (program->builtin_kernel_names == NULL)
    return 0;

  program->num_kernels = program->num_builtin_kernels;
  if (program->num_kernels) {
    program->kernel_meta = (pocl_kernel_metadata_t *)calloc(
        program->num_kernels, sizeof(pocl_kernel_metadata_t));

    for (size_t i = 0; i < program->num_kernels; ++i) {
      pocl_clone_builtin_kernel_metadata(device,
                                       program->builtin_kernel_names[i],
                                       &program->kernel_meta[i]);
      program->kernel_meta[i].data =
          (void**)calloc(program->num_devices, sizeof(void*));
    }
  }

  return 1;
}

int pocl_sanitize_builtin_kernel_name(cl_kernel kernel, const char **saved_name) {
  assert (kernel->program->num_builtin_kernels);
  *saved_name = NULL;
  if (kernel->program->num_builtin_kernels) {
    *saved_name = kernel->meta->name;
    char* copied_name = strdup(kernel->name);
    size_t len = strlen(copied_name);
    for (uint i = 0; i < len; ++i)
      if (copied_name[i] == '.') copied_name[i] = '_';
    kernel->meta->name = copied_name;
    kernel->name = kernel->meta->name;
  }
  return 0;
}

int pocl_restore_builtin_kernel_name(cl_kernel kernel, const char *saved_name) {
  assert (kernel->program->num_builtin_kernels);
  free((void *)kernel->name);
  kernel->meta->name = saved_name;
  kernel->name = kernel->meta->name;
  return 0;
}

static int pocl_validate_khr_gemm(cl_bool TransA, cl_bool TransB,
                                  const cl_tensor_desc *TenA,
                                  const cl_tensor_desc *TenB,
                                  const cl_tensor_desc *TenCIOpt,
                                  const cl_tensor_desc *TenCOut,
                                  const cl_tensor_datatype_union *Alpha,
                                  const cl_tensor_datatype_union *Beta)
{
    POCL_RETURN_ERROR_COND((TenA == NULL), CL_INVALID_DBK_ATTRIBUTE);
    POCL_RETURN_ERROR_COND((TenB == NULL), CL_INVALID_DBK_ATTRIBUTE);
    POCL_RETURN_ERROR_COND((TenCOut == NULL), CL_INVALID_DBK_ATTRIBUTE);

    if (Alpha) {
      POCL_RETURN_ERROR_COND (Alpha->l == 0, CL_INVALID_DBK_ATTRIBUTE);
    }
    // beta can be 0
    //POCL_RETURN_ERROR_COND (Beta->l == 0, CL_INVALID_DBK_ATTRIBUTE);

    // TBC: 4D+ tensor could be supported by treating the additional
    //      dimensions as batch dimensions - but it might not be
    //      worthwhile due the extra work to support them and processing
    //      overhead they may impose.
    POCL_RETURN_ERROR_ON((TenA->rank > 3), CL_INVALID_DBK_RANK,
                         "Unsupported high-degree tensors.\n");
    POCL_RETURN_ERROR_ON((TenA->rank < 2), CL_INVALID_DBK_RANK,
                         "Rank of A/B tensors must be in {2,3}.\n");

    POCL_RETURN_ERROR_ON((TenA->rank != TenB->rank), CL_INVALID_DBK_RANK,
                         "Rank mismatch between A and B\n");
    POCL_RETURN_ERROR_ON((TenB->rank != TenCOut->rank), CL_INVALID_DBK_RANK,
                         "Rank mismatch between A/B and COut\n");

    POCL_RETURN_ERROR_ON((TenCIOpt != NULL && pocl_tensor_shape_equals(TenCIOpt, TenCOut) == CL_FALSE),
                         CL_INVALID_DBK_SHAPE, "Tensor shape mismatch between C_in and C_out.");

    size_t BatchDims = TenA->rank - 2;

    size_t Temp;
    // CO[b][m][n] = sigma_over_m_n_k(A[b][m][k] * B[b][k][n]) + CI[b][m][n].
    size_t Am = TenA->shape[BatchDims + 0];
    size_t Ak = TenA->shape[BatchDims + 1];
    if (TransA) { Temp = Am ; Am = Ak ; Ak = Temp; }

    size_t Bk = TenB->shape[BatchDims + 0];
    size_t Bn = TenB->shape[BatchDims + 1];
    if (TransB) { Temp = Bk ; Bk = Bn ; Bn = Temp; }

    size_t COm = TenCOut->shape[BatchDims + 0];
    size_t COn = TenCOut->shape[BatchDims + 1];

    POCL_RETURN_ERROR_COND((Ak != Bk), CL_INVALID_DBK_ATTRIBUTE);
    POCL_RETURN_ERROR_COND((Am != COm), CL_INVALID_DBK_ATTRIBUTE);
    POCL_RETURN_ERROR_COND((Bn != COn), CL_INVALID_DBK_ATTRIBUTE);

    // Check batch dimensions match.
    if (TenA->rank == 3) {
      size_t BatchSize = TenA->shape[0];
      POCL_RETURN_ERROR_ON ((BatchSize > 1 && (BatchSize != TenB->shape[0]
                                               || TenB->shape[0] != TenCOut->shape[0])),
          CL_INVALID_DBK_SHAPE, "Batch size mismatch.\n");

      POCL_RETURN_ERROR_ON ((BatchSize > 1 && TenCIOpt
                             && TenCIOpt->shape[0] != TenCOut->shape[0]),
          CL_INVALID_DBK_SHAPE, "Batch size mismatch.\n");
    }

    // Check datatypes
    POCL_RETURN_ERROR_ON ((TenA->dtype != TenB->dtype),
                          CL_INVALID_DBK_DATATYPE,
                          "datatype mismatch between A and B.\n");

    POCL_RETURN_ERROR_ON (TenCIOpt && (TenCIOpt->dtype != TenCOut->dtype),
                          CL_INVALID_DBK_DATATYPE,
                          "datatype mismatch between C_ind and C_out\n");

    // TODO: check validity of data layouts of the tensors. Now assumes they're ok

    return CL_SUCCESS;
}


int pocl_validate_dbk_attributes(BuiltinKernelId kernel_id,
                                 const void *kernel_attributes,
                                 pocl_validate_khr_gemm_callback_t GemmCB)
{
  if (GemmCB == NULL)
    GemmCB = pocl_validate_khr_gemm;

  switch (kernel_id)
  {
    case POCL_CDBI_DBK_KHR_GEMM:
    {
      const cl_dbk_attributes_khr_gemm *Attrs =
          (const cl_dbk_attributes_khr_gemm *)kernel_attributes;

      return GemmCB(Attrs->trans_a, Attrs->trans_b,
                    &Attrs->a, &Attrs->b, &Attrs->c_in,
                    &Attrs->c_out, &Attrs->alpha, &Attrs->beta);
    }
    case POCL_CDBI_DBK_KHR_MATMUL:
    {
      const cl_dbk_attributes_khr_matmul *Attrs =
          (const cl_dbk_attributes_khr_matmul *)kernel_attributes;

      return GemmCB(Attrs->trans_a, Attrs->trans_b,
                    &Attrs->a, &Attrs->b, NULL,
                    &Attrs->c, NULL, NULL);

    }
    default: break;
  }
  POCL_RETURN_ERROR_ON(1, CL_INVALID_DBK_ID,
                       "Unknown builtin kernel ID: %u", kernel_id);
}

void *pocl_copy_defined_builtin_attributes(BuiltinKernelId kernel_id,
                                           const void *kernel_attributes)
{
  switch (kernel_id)
  {
    case POCL_CDBI_DBK_KHR_GEMM:
    {
      cl_dbk_attributes_khr_gemm *attrs = malloc(sizeof(cl_dbk_attributes_khr_gemm));
      if (attrs == NULL) return NULL;
      memcpy(attrs, kernel_attributes, sizeof(cl_dbk_attributes_khr_gemm));
      return attrs;
    }
    case POCL_CDBI_DBK_KHR_MATMUL:
    {
      cl_dbk_attributes_khr_matmul *attrs = malloc(sizeof(cl_dbk_attributes_khr_matmul));
      if (attrs == NULL) return NULL;
      memcpy(attrs, kernel_attributes, sizeof(cl_dbk_attributes_khr_matmul));
      return attrs;
    }
    default: break;
  }
  POCL_MSG_ERR("Unknown builtin kernel ID: %u", kernel_id);
  return NULL;
}

int pocl_release_defined_builtin_attributes(BuiltinKernelId kernel_id,
                                            void *kernel_attributes)
{
  switch (kernel_id)
  {
    case POCL_CDBI_DBK_KHR_GEMM:
    case POCL_CDBI_DBK_KHR_MATMUL:
    {
      POCL_MEM_FREE(kernel_attributes);
      return CL_SUCCESS;
    }
    default: break;
  }
  POCL_RETURN_ERROR_ON(1, CL_INVALID_DBK_ID,
                       "Unknown builtin kernel ID: %u", kernel_id);
}
