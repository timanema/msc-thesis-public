#include <bench/gtsst-bench.cuh>
#include <compressors/compactionv1/compaction-compressor.cuh>
#include <compressors/compactionv2/compaction-compressor.cuh>
#include <compressors/compactionv3/compaction-compressor.cuh>
#include <compressors/compactionv3t/compaction-compressor.cuh>
#include <compressors/compactionv4/compaction-compressor.cuh>
#include <compressors/compactionv4t/compaction-compressor.cuh>
#include <compressors/compactionv5t/compaction-compressor.cuh>

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    const bool use_override = argc >= 2;

    // Set directories to use
    std::vector<std::string> directories = {
        // "/home/tim/thesis-testing/lineitem-4096b/",
        // "/home/tim/thesis-testing/lineitem-0.1gb/",
        // "/home/tim/tudelft/thesis/thesis-testing/lineitem-4096b/",
        // "/home/tim/tudelft/thesis/thesis-testing/lineitem-0.01gb/",
        // "../../thesis-testing/lineitem-4096b/",
        // "../../thesis-testing/lineitem-0.01gb/",
        // "../../thesis-testing/lineitem-0.1gb/",
        // "../../thesis-testing/lineitem-0.2gb/",
        // "../../thesis-testing/lineitem-0.5gb/",
         "../../thesis-testing/lineitem-1gb/",
        // "../../thesis-testing/lineitem-2gb/",
        // "../../thesis-testing/lineitem-3gb/",
        // "../../thesis-testing/lineitem-4gb/",
        // "../../thesis-testing/lineitem-5gb/",
        // "../../thesis-testing/lineitem-6gb/",
        // "../../thesis-testing/lineitem-7gb/",
        // "../../thesis-testing/lineitem-10gb/",
        // "../../thesis-testing/customer-1gb/",
        // "../../thesis-testing/gdelt-locations-inflated-1gb/",
        // "../../thesis-testing/dbtext/",
        // "../../thesis-testing/dbtext-inflated-1gb/",
        // "/home/tim/thesis-testing/dbtext/"
        // "../../thesis-testing/lineitem-2gb/",
        // "../../thesis-testing/customer-2gb/",
        // "../../thesis-testing/gdelt-locations-inflated-2gb/",
        // "../../thesis-testing/dbtext-inflated-2gb/",
        // "../../thesis-testing/lineitem-4gb/",
        // "../../thesis-testing/customer-4gb/",
        // "../../thesis-testing/gdelt-locations-inflated-4gb/",
        // "../../thesis-testing/dbtext-inflated-4gb/",
    };

    if (use_override) {
        directories.clear();

        for (int i = 1; i < argc; i++) {
            directories.emplace_back(argv[i]);
        }
    }

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