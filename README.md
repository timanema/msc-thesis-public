# Thesis project Tim Anema
This GitHub repository contains my compression pipeline implementations and my evaluation data.
It can be used to validate the compression pipelines, check my results, and also to continue my work.

Note that this repository does not include the version integrated with GSST, as this was done with a closed-source copy
of the GSST decompressor. This version has a closed source license from Voltron Data.

## Thesis abstract
This thesis presents a GPU-accelerated string compression algorithm based on FSST (_Fast Static Symbol Table_).
The proposed compressor leverages several advanced CUDA techniques to optimize performance, including a voting mechanism that maximizes memory bandwidth and an efficient gathering pipeline utilizing stream compaction.
Additionally, the algorithm uses GPU compute capacity to support a memory-efficient encoding table through a space-time tradeoff.

The compression task is parallelized by tiling input data and adapting the data layout.
We introduce multiple compression pipelines, each with distinct tradeoffs.
To maximize encoding kernel throughput, the design introduces sliding windows and output packing to optimize register use and maximize effective memory bandwidth.
Pipeline-level throughput is further enhanced by introducing pipelined transposition stages and stream compaction to remove intermediate padding efficiently.

We evaluate these pipelines across several benchmark datasets and compare the best-performing version against state-of-the-art GPU compression algorithms, including nvCOMP, GPULZ, and compressors generated using the LC framework.
The proposed compressor achieves a throughput of 74GB/s on an RTX4090 while maintaining compression ratios comparable to FSST.
In terms of compression ratio, it consistently outperforms ANS, Bitcomp, Cascaded, and GPULZ across all datasets.
Its overall throughput exceeds that of GPULZ and all nvCOMP compressors except ANS, Bitcomp, Cascaded, and those produced by the LC framework.
Our compressor lies on the Pareto frontier for all evaluated datasets, advancing the state-of-the-art toward ideal compression.
It achieves near-identical compression ratios to FSST (except for machine-readable datasets), while achieving a speedup of 42.06x.
Compared to multithreaded CPU compression, it achieves a 6.45x speedup.

To assess end-to-end performance, we integrate the compressor with the GSST decompressor. The resulting (de)compression pipeline achieves a combined throughput of 55GB/s, outperforming uncompressed data transfer on links with a bandwidth up to 37.5 GB/s.
It also outperforms all state-of-the-art (de)compressors when the link bandwidth ranges between 3GB/s and 20GB/s.

While further research is needed to enhance robustness and integrate the compressor into analytical engines, this work demonstrates a viable and Pareto-optimal alternative to existing string compression methods.

## Instructions
The repository is organized as follows:
```bash
.
├── data                    # Experimental results
├── include                 # Header files
│   ├── bench             # Benchmark files
│   ├── compressors       # Actual compressors
│   ├── fsst              # Modified version of FSST
│   └── gtsst             # Encoding tables, symbols, shared code
└── src                     # Source files
    ├── bench
    ├── compressors
    └── fsst

```
Every (interesting) compression pipeline will have tree header files: `*-compressor.cuh`, `*-defines.cuh`, and `*-encode.cuh`.
These contain the public methods, parameter definitions, and private definitions, respectively.
All compressors implements this template:
```c++
struct CompressionManager
{
    virtual ~CompressionManager() = default;
    virtual CompressionConfiguration configure_compression(size_t buf_size) = 0;
    virtual GTSSTStatus compress(const uint8_t* src, uint8_t* dst, const uint8_t* sample_src, uint8_t* tmp,
                                 CompressionConfiguration& config, size_t* out_size,
                                 CompressionStatistics& stats) = 0;

    virtual DecompressionConfiguration configure_decompression(size_t buf_size) = 0;

    virtual DecompressionConfiguration configure_decompression_from_compress(
        const size_t buf_size, CompressionConfiguration& config)
    {
        return DecompressionConfiguration{
            .input_buffer_size = buf_size,
            .decompression_buffer_size = config.input_buffer_size,
        };
    }

    virtual GTSSTStatus decompress(const uint8_t* src, uint8_t* dst, DecompressionConfiguration& config,
                                   size_t* out_size) = 0;

private:
    virtual GTSSTStatus validate_compression_buffers(const uint8_t* src, uint8_t* dst, uint8_t* tmp,
                                                     CompressionConfiguration& config) = 0;
};
```

### Building the project
To build the project, you need to at least have the [CUDA development library](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and CMake 3.25.2 installed, but a more complete C++/CUDA environment is recommended.
You can then build the project with the following commands:
```bash
cmake . -B build -DCMAKE_BUILD_TYPE=Release
cd build/
make
```

### Running the project
Once you have build the project, the executable can simply be run with `./gtsst`.

However, this will likely result in the following error:
`Error: filesystem error: directory iterator cannot open directory: No such file or directory [../../thesis-testing/lineitem-1gb/]`
This is because, by default, the project uses this directory to load data.
The directories to use can be given as a program argument:
`./gtsst ../../thesis-testing/lineitem-1gb/ ../../thesis-testing/lineitem-0.5gb/`

By default, the V5T pipeline is used to perform 100 compression iterations on all files in the given directories and 
a single validation decompression. This can be changed by modifying the main.cu file:
```c++
int main(int argc, char* argv[]) {
    const bool use_override = argc >= 2;

    // Set directories to use
    std::vector<std::string> directories = {
         "../../thesis-testing/lineitem-1gb/",
    };

    if (use_override) {
        directories.clear();

        for (int i = 1; i < argc; i++) {
            directories.emplace_back(argv[i]);
        }
    }

    // Uncomment the compressor you want to test (only one)
    // Uncomment the compressor you want to test (only one)
    // gtsst::compressors::CompactionV1Compressor compressor;
    // gtsst::compressors::CompactionV2Compressor compressor;
    // gtsst::compressors::CompactionV3Compressor compressor;
    // gtsst::compressors::CompactionV4Compressor compressor;
    // gtsst::compressors::CompactionV3TCompressor compressor;
    // gtsst::compressors::CompactionV4TCompressor compressor;
    gtsst::compressors::CompactionV5TCompressor compressor;
    // gtsst::compressors::CompactionV5TGSSTCompressor compressor; <---- NOT PUBLICLY DISTRIBUTED DUE TO CLOSED LICENSE FROM VOLTRON DATA

    // Set bench settings
    constexpr int compression_iterations = 100;
    constexpr int decompression_iterations = 1;
    constexpr bool strict_checking = false; // Exit program when a single decompression mismatch occurs, otherwise only report it

    // Run benchmark (use_dir=true if all files in the directory must be used, otherwise uses first file only)
    const bool match = gtsst::bench::full_cycle_directory(directories, false, compression_iterations,
                                                          decompression_iterations, compressor, false, strict_checking);
    if (!match) {
        std::cerr << "Cycle data mismatch." << std::endl;
        return 1;
    }

    return 0;
}
```
The default directories can be modified, the compressor can be chosen, and the number of iterations can be selected.

Note that in the current version we expect that the data does not contain the character `0xfe` (254).
If it does, it will break. This is expected behaviour and can be fixed relatively easily, but I didn't since it's not really an issue for text data (see thesis 'Future work'' section).
You can check if this is happening by running the compressor in debug mode: `cmake . -B build -DCMAKE_BUILD_TYPE=Debug`

### Output data
The output will be in the following format:
```
Strict output checking is disabled, decompressed data might not match original data!
encoding: 9.715
encoding: 13.033
encoding: 17.569
encoding: 17.356
encoding: 17.523
decomp: 32.744
decomp: 34.543
decomp: 35.853
decomp: 34.919
decomp: 34.763
error: 675819520 -> 32 != 0
error: 675819521 -> 97 != 0
error: 675819522 -> 114 != 0
Cycles (5, 5) completed. Stats:
	Parameters:
		Block size: 2621440
		Input size: 988282880
		Effective table size: 62914560
		File name: ../../thesis-testing/lineitem-1gb/lineitem-comments-1gb-1.txt
	Compression:
		Duration (us): 85116 
		Throughput (GB/s): 12.192
		Compressed size: 351846452
	Decompression:
		Duration (us): 28617
		Throughput (GB/s): 34.535
		Ratio: 2.8088
	Compression phases:
		Table generation (GB/s, us): 182.880 (5404)
		Precomputation (GB/s, us): 31880.093 (31)
		Encoding (GB/s, us): 14.235 (69428)
		Postprocessing (GB/s, us): 112.063 (8819)
```
The first line indicates that the `strict_checking` was set to `false` (required for GSST).
Then the individual encoding throughputs and decompression throughput will be reported for every iteration.
If there are any differences in the decompressed data compared to the original data, their location and values will be reported.
Finally, a summary will be printed. This contains the (average) throughput for compression
(and the individual stages) and decompression, as well as the compression ratio.