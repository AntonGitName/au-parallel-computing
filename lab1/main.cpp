#include <iostream>
#include <fstream>
#include <cassert>
#include <CL/opencl.h>
#include <vector>

using std::string;


namespace {
    const char *NVIDIA = "NVIDIA";
    const char *convolute_program = "convolute_kernel.cl";
    const char *convolute_function = "convolute";

    const size_t BLOCK_SZ = 32;

    const char *INPUT = "input.txt";
    const char *OUTPUT = "output.txt";
    const size_t MAXN = 1024;
    const size_t MAXM = 9;

    typedef std::vector<float> floats;
}

cl_platform_id get_platform_id() {
    static const cl_uint max_platforms = 32;
    cl_platform_id platforms[max_platforms];
    cl_uint num_platforms = 0;

    auto status = clGetPlatformIDs(max_platforms, platforms, &num_platforms);
    assert(status == CL_SUCCESS && "Error getting platforms");

    char vendor[256];
    for (cl_int i = 0; i < num_platforms; i++) {
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), &vendor, NULL);
        assert(status == CL_SUCCESS && "Error getting platform info");

        if (string(vendor).find(NVIDIA) != string::npos) {
            return platforms[i];
        }
    }
    assert(0 && "No suitable platform found");
    return nullptr;
}

void read_input(floats &matrix, floats &kernel, size_t &n, size_t &m) {
    std::ifstream in(INPUT);
    in >> n >> m;

    assert (n >= 0 && n <= MAXN && "Invalid matrix size");
    assert (m >= 0 && m <= MAXM && "Invalid kernel size");
    assert ((m & 1) && "Kernel size is even");

    matrix.assign(n * n, 0);
    kernel.assign(m * m, 0);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            in >> matrix[i * n + j];
        }
    }
    for (size_t i = 0; i < m * m; i++) {
        in >> kernel[i];
    }
}

void write_output(const floats &matrix, size_t n) {
    std::ofstream out(OUTPUT);

//    out.precision(3);

    size_t index = 0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            out << matrix[index++] << " ";
//            out << std::fixed << matrix[index++] << " ";
        }
        out << std::endl;
    }
}

void print_build_log(cl_program program, cl_device_id device) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    string build_log(log_size, 0);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, (void *) build_log.data(), NULL);
    std::cout << build_log << std::endl;
}

template<class T>
void set_kernel_arg(cl_kernel kernel, cl_uint arg_num, const T &data) {
    auto status = clSetKernelArg(kernel, arg_num, sizeof(T), &data);
    assert(status == CL_SUCCESS && "Could not set argument");
}

template<class T>
cl_mem create_buffer(cl_context ctx, cl_mem_flags flags, size_t sz, T *ptr) {
    cl_int status;
    auto res = clCreateBuffer(ctx, flags, sz * sizeof(T), (void *) ptr, &status);
    assert(status == CL_SUCCESS && "Could not create buffer");
    return res;
}

void calculate_parallel(const floats &matrix, const floats &kernel, size_t n, size_t m, floats &result) {
    auto platform_id = get_platform_id();
    cl_device_id device_id;
    auto status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    assert(status == CL_SUCCESS && "Error getting device_id");
    const auto context = clCreateContext(nullptr, 1, &device_id, NULL, NULL, &status);
    assert(status == CL_SUCCESS && "Error creating context");

    const auto command_queue = clCreateCommandQueue(context, device_id, 0, &status);
    // No idea why it does not work this way. SIGSEGV is the least expected here.
//    const auto command_queue = clCreateCommandQueueWithProperties(context, device_id, nullptr, &status);
    assert(status == CL_SUCCESS && "Error creating command queue");

    auto matrix_buffer = create_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * n, matrix.data());
    auto kernel_buffer = create_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * m, kernel.data());
    auto result_buffer = create_buffer<float>(context, CL_MEM_WRITE_ONLY, n * n, nullptr);

    std::ifstream program_sources_file(convolute_program);
    string sources((std::istreambuf_iterator<char>(program_sources_file)), std::istreambuf_iterator<char>());
    const char *sources_cstr[] = {sources.c_str()};

    size_t length = sources.length();
    auto program = clCreateProgramWithSource(context, 1, sources_cstr, &length, &status);
    assert(status == CL_SUCCESS && "Error creating cl program source");

    status = clBuildProgram(program, 1, &device_id, ("-D BLOCK_SIZE=" + std::to_string(BLOCK_SZ)).c_str(), NULL, NULL);
    if (status != CL_SUCCESS) {
        print_build_log(program, device_id);
    }
    assert(status == CL_SUCCESS);

    auto convolute_kernel = clCreateKernel(program, convolute_function, &status);
    assert(status == CL_SUCCESS);

    set_kernel_arg(convolute_kernel, 0, matrix_buffer);
    set_kernel_arg(convolute_kernel, 1, (unsigned) n);
    set_kernel_arg(convolute_kernel, 2, kernel_buffer);
    set_kernel_arg(convolute_kernel, 3, (unsigned) m);
    set_kernel_arg(convolute_kernel, 4, result_buffer);

    auto n_rounded = n + (BLOCK_SZ - n % BLOCK_SZ);
    size_t global_ws[] = {n_rounded, n_rounded};
    size_t local_ws[] = {BLOCK_SZ, BLOCK_SZ};

    status = clEnqueueNDRangeKernel(command_queue, convolute_kernel, 2, nullptr, global_ws, local_ws, 0, NULL, NULL);
    assert(status == CL_SUCCESS);

    clEnqueueReadBuffer(command_queue, result_buffer, CL_TRUE, 0, n * n * sizeof(float), result.data(), 0, NULL, NULL);

    clReleaseKernel(convolute_kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    clReleaseMemObject(matrix_buffer);
    clReleaseMemObject(kernel_buffer);
    clReleaseMemObject(result_buffer);
}

int main(void) {
    floats matrix;
    floats kernel;
    size_t n;
    size_t m;

    read_input(matrix, kernel, n, m);

    floats result(n * n);
    calculate_parallel(matrix, kernel, n, m, result);

    write_output(result, n);

    return 0;
}