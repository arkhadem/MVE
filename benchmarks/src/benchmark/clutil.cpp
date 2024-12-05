#include "clutil.hpp"

void printErrorString(int pointer, cl_int error) {
    if (error == CL_SUCCESS)
        return;
    switch (error) {
    // run-time and JIT compiler errors
    case 0:
        printf("CL ERROR @%d: CL_SUCCESS\n", pointer);
        break;
    case -1:
        printf("CL ERROR @%d: CL_DEVICE_NOT_FOUND\n", pointer);
        break;
    case -2:
        printf("CL ERROR @%d: CL_DEVICE_NOT_AVAILABLE\n", pointer);
        break;
    case -3:
        printf("CL ERROR @%d: CL_COMPILER_NOT_AVAILABLE\n", pointer);
        break;
    case -4:
        printf("CL ERROR @%d: CL_MEM_OBJECT_ALLOCATION_FAILURE\n", pointer);
        break;
    case -5:
        printf("CL ERROR @%d: CL_OUT_OF_RESOURCES\n", pointer);
        break;
    case -6:
        printf("CL ERROR @%d: CL_OUT_OF_HOST_MEMORY\n", pointer);
        break;
    case -7:
        printf("CL ERROR @%d: CL_PROFILING_INFO_NOT_AVAILABLE\n", pointer);
        break;
    case -8:
        printf("CL ERROR @%d: CL_MEM_COPY_OVERLAP\n", pointer);
        break;
    case -9:
        printf("CL ERROR @%d: CL_IMAGE_FORMAT_MISMATCH\n", pointer);
        break;
    case -10:
        printf("CL ERROR @%d: CL_IMAGE_FORMAT_NOT_SUPPORTED\n", pointer);
        break;
    case -11:
        printf("CL ERROR @%d: CL_BUILD_PROGRAM_FAILURE\n", pointer);
        break;
    case -12:
        printf("CL ERROR @%d: CL_MAP_FAILURE\n", pointer);
        break;
    case -13:
        printf("CL ERROR @%d: CL_MISALIGNED_SUB_BUFFER_OFFSET\n", pointer);
        break;
    case -14:
        printf("CL ERROR @%d: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n", pointer);
        break;
    case -15:
        printf("CL ERROR @%d: CL_COMPILE_PROGRAM_FAILURE\n", pointer);
        break;
    case -16:
        printf("CL ERROR @%d: CL_LINKER_NOT_AVAILABLE\n", pointer);
        break;
    case -17:
        printf("CL ERROR @%d: CL_LINK_PROGRAM_FAILURE\n", pointer);
        break;
    case -18:
        printf("CL ERROR @%d: CL_DEVICE_PARTITION_FAILED\n", pointer);
        break;
    case -19:
        printf("CL ERROR @%d: CL_KERNEL_ARG_INFO_NOT_AVAILABLE\n", pointer);
        break;

    // compile-time errors
    case -30:
        printf("CL ERROR @%d: CL_INVALID_VALUE\n", pointer);
        break;
    case -31:
        printf("CL ERROR @%d: CL_INVALID_DEVICE_TYPE\n", pointer);
        break;
    case -32:
        printf("CL ERROR @%d: CL_INVALID_PLATFORM\n", pointer);
        break;
    case -33:
        printf("CL ERROR @%d: CL_INVALID_DEVICE\n", pointer);
        break;
    case -34:
        printf("CL ERROR @%d: CL_INVALID_CONTEXT\n", pointer);
        break;
    case -35:
        printf("CL ERROR @%d: CL_INVALID_QUEUE_PROPERTIES\n", pointer);
        break;
    case -36:
        printf("CL ERROR @%d: CL_INVALID_COMMAND_QUEUE\n", pointer);
        break;
    case -37:
        printf("CL ERROR @%d: CL_INVALID_HOST_PTR\n", pointer);
        break;
    case -38:
        printf("CL ERROR @%d: CL_INVALID_MEM_OBJECT\n", pointer);
        break;
    case -39:
        printf("CL ERROR @%d: CL_INVALID_IMAGE_FORMAT_DESCRIPTOR\n", pointer);
        break;
    case -40:
        printf("CL ERROR @%d: CL_INVALID_IMAGE_SIZE\n", pointer);
        break;
    case -41:
        printf("CL ERROR @%d: CL_INVALID_SAMPLER\n", pointer);
        break;
    case -42:
        printf("CL ERROR @%d: CL_INVALID_BINARY\n", pointer);
        break;
    case -43:
        printf("CL ERROR @%d: CL_INVALID_BUILD_OPTIONS\n", pointer);
        break;
    case -44:
        printf("CL ERROR @%d: CL_INVALID_PROGRAM\n", pointer);
        break;
    case -45:
        printf("CL ERROR @%d: CL_INVALID_PROGRAM_EXECUTABLE\n", pointer);
        break;
    case -46:
        printf("CL ERROR @%d: CL_INVALID_KERNEL_NAME\n", pointer);
        break;
    case -47:
        printf("CL ERROR @%d: CL_INVALID_KERNEL_DEFINITION\n", pointer);
        break;
    case -48:
        printf("CL ERROR @%d: CL_INVALID_KERNEL\n", pointer);
        break;
    case -49:
        printf("CL ERROR @%d: CL_INVALID_ARG_INDEX\n", pointer);
        break;
    case -50:
        printf("CL ERROR @%d: CL_INVALID_ARG_VALUE\n", pointer);
        break;
    case -51:
        printf("CL ERROR @%d: CL_INVALID_ARG_SIZE\n", pointer);
        break;
    case -52:
        printf("CL ERROR @%d: CL_INVALID_KERNEL_ARGS\n", pointer);
        break;
    case -53:
        printf("CL ERROR @%d: CL_INVALID_WORK_DIMENSION\n", pointer);
        break;
    case -54:
        printf("CL ERROR @%d: CL_INVALID_WORK_GROUP_SIZE\n", pointer);
        break;
    case -55:
        printf("CL ERROR @%d: CL_INVALID_WORK_ITEM_SIZE\n", pointer);
        break;
    case -56:
        printf("CL ERROR @%d: CL_INVALID_GLOBAL_OFFSET\n", pointer);
        break;
    case -57:
        printf("CL ERROR @%d: CL_INVALID_EVENT_WAIT_LIST\n", pointer);
        break;
    case -58:
        printf("CL ERROR @%d: CL_INVALID_EVENT\n", pointer);
        break;
    case -59:
        printf("CL ERROR @%d: CL_INVALID_OPERATION\n", pointer);
        break;
    case -60:
        printf("CL ERROR @%d: CL_INVALID_GL_OBJECT\n", pointer);
        break;
    case -61:
        printf("CL ERROR @%d: CL_INVALID_BUFFER_SIZE\n", pointer);
        break;
    case -62:
        printf("CL ERROR @%d: CL_INVALID_MIP_LEVEL\n", pointer);
        break;
    case -63:
        printf("CL ERROR @%d: CL_INVALID_GLOBAL_WORK_SIZE\n", pointer);
        break;
    case -64:
        printf("CL ERROR @%d: CL_INVALID_PROPERTY\n", pointer);
        break;
    case -65:
        printf("CL ERROR @%d: CL_INVALID_IMAGE_DESCRIPTOR\n", pointer);
        break;
    case -66:
        printf("CL ERROR @%d: CL_INVALID_COMPILER_OPTIONS\n", pointer);
        break;
    case -67:
        printf("CL ERROR @%d: CL_INVALID_LINKER_OPTIONS\n", pointer);
        break;
    case -68:
        printf("CL ERROR @%d: CL_INVALID_DEVICE_PARTITION_COUNT\n", pointer);
        break;

    // extension errors
    case -1000:
        printf("CL ERROR @%d: CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR\n", pointer);
        break;
    case -1001:
        printf("CL ERROR @%d: CL_PLATFORM_NOT_FOUND_KHR\n", pointer);
        break;
    case -1002:
        printf("CL ERROR @%d: CL_INVALID_D3D10_DEVICE_KHR\n", pointer);
        break;
    case -1003:
        printf("CL ERROR @%d: CL_INVALID_D3D10_RESOURCE_KHR\n", pointer);
        break;
    case -1004:
        printf("CL ERROR @%d: CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR\n", pointer);
        break;
    case -1005:
        printf("CL ERROR @%d: CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR\n", pointer);
        break;
    default:
        printf("CL ERROR @%d: Unknown OpenCL error (%d)\n", pointer, error);
        break;
    }
}