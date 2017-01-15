#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else

#include <CL/cl.hpp>

#endif

#include <vector>
#include <fstream>
#include <string>
#include <memory>
#include <ostream>
#include <iostream>
#include <cassert>

using namespace cl;

namespace {
    const char *NVIDIA = "NVIDIA";
#define PROGRAM "prefixsum"
    const char *kernel_program = PROGRAM "_kernel.cl";
    const char *kernel_function = PROGRAM;
#undef  PROGRAM

    const int BLOCK_SZ = 2;
    const int MAXN = 1048576;

    const char *INPUT = "input.txt";
    const char *OUTPUT = "output.txt";
    const char *LOG = "prefixsum.log";

    typedef std::vector<float> floats;

    const unsigned long FLOAT_SZ = sizeof(float);

    class DumpLogger {
    public:
        DumpLogger(std::string fname) : log(fname) {}
        void error(bool condition, std::string s) { if (condition) error(s); }
        void error(std::string s) { log << "ERROR: " << s << std::endl; }
        void info(std::string s) { log << "INFO: " << s << std::endl; }
    private:
        std::ofstream log;
    } logger(LOG);


    typedef make_kernel<Buffer &, Buffer &, LocalSpaceArg, LocalSpaceArg, Buffer &> kf_block_sum;
    typedef make_kernel<Buffer &, Buffer &> kf_array_sum;

    CommandQueue g_queue;
    Context g_context;
    std::shared_ptr<kf_block_sum> kf_sum_blocks;
    std::shared_ptr<kf_array_sum> kf_sum_array;
}

std::vector<Device> get_decices() {
    std::vector<Platform> platforms;
    Platform::get(&platforms);

    logger.error(platforms.empty(), "No platforms found");
    assert(!platforms.empty());

    Platform platform = platforms[0];
    logger.info("Using platform " + platform.getInfo<CL_PLATFORM_NAME>());

    std::vector<Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    logger.error(devices.empty(), "No devices found");
    assert(!devices.empty());

    return devices;
}

Program::Sources get_sources() {
    std::ifstream file(kernel_program);
    std::string source_code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    Program::Sources sources;
    sources.push_back({source_code.c_str(), source_code.length()});
    return sources;
}

floats read_input() {
    int n;
    std::ifstream in(INPUT);
    in >> n;
    floats array((unsigned long) n);
    assert (n >= 0 && n <= MAXN && "Invalid array size");
    for (int i = 0; i < n; i++) {
        in >> array[i];
    }
    return array;
}

void write_output(const floats &array) {
    std::ofstream out(OUTPUT);
    std::copy(array.begin(), array.end(), std::ostream_iterator<float>(out, " "));
}

inline floats copy_from_buffer(Buffer& buffer, unsigned long n) {
    floats blocks_sums(n);
    g_queue.enqueueReadBuffer(buffer, CL_TRUE, 0, FLOAT_SZ * n, blocks_sums.data());
    return blocks_sums;
}

floats prefixsum(const floats &array) {

    constexpr unsigned long local_memory_sz = FLOAT_SZ * BLOCK_SZ;
    unsigned long block_num = ((array.size() + BLOCK_SZ - 1) / BLOCK_SZ);
    unsigned long sz_expanded = BLOCK_SZ * block_num;

    Buffer input_buffer(g_context, CL_MEM_READ_ONLY, FLOAT_SZ * sz_expanded);
    g_queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, FLOAT_SZ * sz_expanded, array.data());

    Buffer blocks_buffer(g_context, CL_MEM_READ_WRITE, FLOAT_SZ * block_num);
    Buffer blocks_sum_buffer(g_context, CL_MEM_READ_ONLY, FLOAT_SZ * block_num);
    Buffer output_buffer(g_context, CL_MEM_READ_WRITE, FLOAT_SZ * sz_expanded);

    EnqueueArgs args(g_queue, NDRange(sz_expanded), NDRange(BLOCK_SZ));
    auto local_space = Local(local_memory_sz);
    auto event = (*kf_sum_blocks)(args, input_buffer, blocks_buffer, local_space, local_space, output_buffer);
    event.wait();

    if (block_num != 1) {
        auto blocks_sums = copy_from_buffer(blocks_buffer, block_num);
        floats pr_sums_blocks = prefixsum(blocks_sums);
        g_queue.enqueueWriteBuffer(blocks_sum_buffer, CL_TRUE, 0, FLOAT_SZ * block_num, pr_sums_blocks.data());
        event = (*kf_sum_array)(args, blocks_sum_buffer, output_buffer);
        event.wait();
    }
    return copy_from_buffer(output_buffer, array.size());
}

int main() {
    auto devices = get_decices();
    g_context = Context(devices);

    auto device = devices.back();
    logger.info("Using device " + device.getInfo<CL_DEVICE_NAME>());

    auto sources = get_sources();

    Program program(g_context, sources);
    auto ret = program.build({device});
    logger.info("PROGRAM BUILD LOG:\n" + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
    logger.error(ret != CL_SUCCESS, "Could not build *.cl sources.");
    assert(ret == CL_SUCCESS);

    g_queue = CommandQueue(g_context, device);

    kf_sum_blocks = std::make_shared<kf_block_sum>(Kernel(program, "sum_blocks"));
    kf_sum_array = std::make_shared<kf_array_sum>(Kernel(program, "sum_array"));

    auto array = read_input();
    auto result = prefixsum(array);
    write_output(result);

    return 0;
}