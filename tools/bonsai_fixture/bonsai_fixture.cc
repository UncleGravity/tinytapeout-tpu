#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

namespace {

constexpr const char * kDefaultModel = "models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf";
constexpr const char * kDefaultTensor = "blk.0.attn_q.weight";
constexpr int kDefaultTileRows = 2;
constexpr int kDefaultTileCols = 4;
constexpr int kGroupSize = 128;
constexpr int kQ1BlockBytes = 18;
constexpr int kQ8BlockSize = 32;
constexpr int kQ8BlocksPerQ1Group = kGroupSize / kQ8BlockSize;

struct ModelMeta {
    gguf_context * gguf = nullptr;
    ggml_context * ggml = nullptr;
    ggml_tensor * tensor = nullptr;
    int64_t tensor_id = -1;
};

struct Q1Block {
    uint16_t scale_fp16 = 0;
    std::array<uint8_t, 16> qs{};
};

using Q1RawBlock = std::array<uint8_t, kQ1BlockBytes>;
using Q8RawBlock = std::array<uint8_t, 2 + kQ8BlockSize>;

int8_t deterministic_act(int index) {
    const int value = ((index * 37 + 19) % 253) - 126;
    return static_cast<int8_t>(value);
}

uint16_t deterministic_q8_scale_fp16(int group_index, int q8_block) {
    const float scale = 0.015625f * static_cast<float>(1 + ((group_index * 7 + q8_block * 3) % 11));
    return ggml_fp32_to_fp16(scale);
}

int bit_at(const Q1Block & block, int bit_index) {
    const int byte_index = bit_index / 8;
    const int bit_offset = bit_index % 8;
    return (block.qs[byte_index] >> bit_offset) & 1;
}

std::string hex16(uint16_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0') << std::setw(4) << value;
    return out.str();
}

Q1RawBlock q1_raw_block(const Q1Block & block) {
    Q1RawBlock raw{};
    raw[0] = static_cast<uint8_t>(block.scale_fp16 & 0xFF);
    raw[1] = static_cast<uint8_t>(block.scale_fp16 >> 8);
    for (int i = 0; i < 16; ++i) {
        raw[2 + i] = block.qs[i];
    }
    return raw;
}

std::array<Q8RawBlock, kQ8BlocksPerQ1Group> q8_raw_blocks(
    const std::vector<int> & acts,
    int group_index
) {
    std::array<Q8RawBlock, kQ8BlocksPerQ1Group> blocks{};
    for (int q8_block = 0; q8_block < kQ8BlocksPerQ1Group; ++q8_block) {
        const uint16_t scale = deterministic_q8_scale_fp16(group_index, q8_block);
        blocks[q8_block][0] = static_cast<uint8_t>(scale & 0xFF);
        blocks[q8_block][1] = static_cast<uint8_t>(scale >> 8);
        const int base = group_index * kGroupSize + q8_block * kQ8BlockSize;
        for (int i = 0; i < kQ8BlockSize; ++i) {
            blocks[q8_block][2 + i] = static_cast<uint8_t>(
                static_cast<int8_t>(acts[base + i])
            );
        }
    }
    return blocks;
}

bool load_meta(const std::string & model_path, const std::string & tensor_name, ModelMeta & meta) {
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &meta.ggml;

    meta.gguf = gguf_init_from_file(model_path.c_str(), params);
    if (meta.gguf == nullptr || meta.ggml == nullptr) {
        std::cerr << "failed to load GGUF metadata: " << model_path << "\n";
        return false;
    }

    meta.tensor_id = gguf_find_tensor(meta.gguf, tensor_name.c_str());
    if (meta.tensor_id < 0) {
        std::cerr << "tensor not found: " << tensor_name << "\n";
        return false;
    }

    meta.tensor = ggml_get_tensor(meta.ggml, tensor_name.c_str());
    if (meta.tensor == nullptr) {
        std::cerr << "tensor metadata missing from ggml context: " << tensor_name << "\n";
        return false;
    }

    if (meta.tensor->type != GGML_TYPE_Q1_0) {
        std::cerr << "expected q1_0 tensor, got " << ggml_type_name(meta.tensor->type) << "\n";
        return false;
    }

    return true;
}

void free_meta(ModelMeta & meta) {
    if (meta.gguf != nullptr) {
        gguf_free(meta.gguf);
    }
    if (meta.ggml != nullptr) {
        ggml_free(meta.ggml);
    }
}

bool read_block(
    const std::string & model_path,
    const ModelMeta & meta,
    int row,
    int group,
    Q1Block & block
) {
    const int64_t groups_per_row = meta.tensor->ne[0] / kGroupSize;
    if (row < 0 || row >= meta.tensor->ne[1] || group < 0 || group >= groups_per_row) {
        std::cerr << "row/group out of range\n";
        return false;
    }

    const size_t tensor_file_offset =
        gguf_get_data_offset(meta.gguf) + gguf_get_tensor_offset(meta.gguf, meta.tensor_id);
    const size_t block_offset = tensor_file_offset + row * meta.tensor->nb[1] + group * kQ1BlockBytes;

    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        std::cerr << "failed to open model: " << model_path << ": " << std::strerror(errno) << "\n";
        return false;
    }

    std::array<uint8_t, kQ1BlockBytes> raw{};
    file.seekg(static_cast<std::streamoff>(block_offset));
    file.read(reinterpret_cast<char *>(raw.data()), raw.size());
    if (!file) {
        std::cerr << "failed to read Q1_0 block at offset " << block_offset << "\n";
        return false;
    }

    block.scale_fp16 = static_cast<uint16_t>(raw[0]) | (static_cast<uint16_t>(raw[1]) << 8);
    for (int i = 0; i < 16; ++i) {
        block.qs[i] = raw[2 + i];
    }
    return true;
}

int inspect(const std::string & model_path, const std::string & tensor_name) {
    ModelMeta meta;
    if (!load_meta(model_path, tensor_name, meta)) {
        free_meta(meta);
        return EXIT_FAILURE;
    }

    std::cout << "model: " << model_path << "\n";
    std::cout << "gguf_version: " << gguf_get_version(meta.gguf) << "\n";
    std::cout << "data_offset: " << gguf_get_data_offset(meta.gguf) << "\n";
    std::cout << "tensor: " << tensor_name << "\n";
    std::cout << "type: " << ggml_type_name(meta.tensor->type) << "\n";
    std::cout << "size_bytes: " << gguf_get_tensor_size(meta.gguf, meta.tensor_id) << "\n";
    std::cout << "file_data_offset: "
              << (gguf_get_data_offset(meta.gguf) + gguf_get_tensor_offset(meta.gguf, meta.tensor_id))
              << "\n";
    std::cout << "ne: ["
              << meta.tensor->ne[0] << ", "
              << meta.tensor->ne[1] << ", "
              << meta.tensor->ne[2] << ", "
              << meta.tensor->ne[3] << "]\n";
    std::cout << "nb: ["
              << meta.tensor->nb[0] << ", "
              << meta.tensor->nb[1] << ", "
              << meta.tensor->nb[2] << ", "
              << meta.tensor->nb[3] << "]\n";

    free_meta(meta);
    return EXIT_SUCCESS;
}

void write_int_array(std::ostream & out, const std::vector<int> & values) {
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << "]";
}

void write_float_array(std::ostream & out, const std::vector<float> & values) {
    out << "[";
    out << std::setprecision(9);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << values[i];
    }
    out << "]";
}

int generate(
    const std::string & model_path,
    const std::string & output_path,
    const std::string & tensor_name,
    int row0,
    int group_start,
    int group_count,
    int tile_rows,
    int tile_cols,
    bool all_groups
) {
    const int row_last = row0 + tile_rows - 1;
    ModelMeta meta;
    if (!load_meta(model_path, tensor_name, meta)) {
        free_meta(meta);
        return EXIT_FAILURE;
    }

    if (tile_rows <= 0 || tile_cols <= 0) {
        std::cerr << "tile rows/cols must be positive\n";
        free_meta(meta);
        return EXIT_FAILURE;
    }
    if (kGroupSize % tile_cols != 0) {
        std::cerr << "tile cols must divide Q1_0 group size " << kGroupSize << "\n";
        free_meta(meta);
        return EXIT_FAILURE;
    }
    if (kQ8BlockSize % tile_cols != 0) {
        std::cerr << "tile cols must divide Q8_0 block size " << kQ8BlockSize << "\n";
        free_meta(meta);
        return EXIT_FAILURE;
    }
    if (meta.tensor->ne[0] % kGroupSize != 0) {
        std::cerr << "tensor width is not divisible by Q1_0 group size\n";
        free_meta(meta);
        return EXIT_FAILURE;
    }
    if (row0 < 0 || row_last >= meta.tensor->ne[1]) {
        std::cerr << "tile row range out of bounds for tensor output rows: "
                  << row0 << ".." << row_last << "\n";
        free_meta(meta);
        return EXIT_FAILURE;
    }
    const int groups_per_row = static_cast<int>(meta.tensor->ne[0] / kGroupSize);
    if (group_start < 0 || group_count <= 0 || group_start + group_count > groups_per_row) {
        std::cerr << "group range out of bounds for tensor input groups: "
                  << group_start << ".." << (group_start + group_count - 1) << "\n";
        free_meta(meta);
        return EXIT_FAILURE;
    }

    std::vector<int> acts;
    acts.reserve(static_cast<size_t>(group_count * kGroupSize));
    for (int i = 0; i < group_count * kGroupSize; ++i) {
        acts.push_back(deterministic_act(i));
    }

    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "failed to open output: " << output_path << "\n";
        free_meta(meta);
        return EXIT_FAILURE;
    }

    std::vector<int> seeds(static_cast<size_t>(tile_rows), 0);
    std::vector<float> scaled_expected(static_cast<size_t>(tile_rows), 0.0f);
    std::vector<float> ggml_vec_dot_expected(static_cast<size_t>(tile_rows), 0.0f);
    const ggml_type_traits_cpu * q1_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q1_0);
    if (q1_traits == nullptr || q1_traits->vec_dot == nullptr ||
        q1_traits->vec_dot_type != GGML_TYPE_Q8_0) {
        std::cerr << "ggml Q1_0 CPU vec_dot with Q8_0 is unavailable\n";
        free_meta(meta);
        return EXIT_FAILURE;
    }
    out << "{\n";
    out << "  \"source\": {\n";
    out << "    \"model\": \"" << model_path << "\",\n";
    out << "    \"tensor\": \"" << tensor_name << "\",\n";
    out << "    \"type\": \"" << ggml_type_name(meta.tensor->type) << "\",\n";
    out << "    \"shape\": [" << meta.tensor->ne[0] << ", " << meta.tensor->ne[1] << "],\n";
    out << "    \"row_stride_bytes\": " << meta.tensor->nb[1] << ",\n";
    out << "    \"q1_0_group_size\": " << kGroupSize << ",\n";
    out << "    \"q8_0_block_size\": " << kQ8BlockSize << "\n";
    out << "  },\n";
    out << "  \"tile\": {\"rows\": " << tile_rows << ", \"cols\": " << tile_cols << "},\n";
    out << "  \"selection\": {\"rows\": [" << row0 << ", " << row_last
        << "], ";
    if (all_groups) {
        out << "\"groups\": [" << group_start << ", " << (group_start + group_count - 1)
            << "], \"cols\": [0, " << (group_count * kGroupSize - 1) << "]},\n";
    } else {
        out << "\"group\": " << group_start << ", \"cols\": [0, 127]},\n";
    }
    out << "  \"q1_0_scales_fp16_hex_by_group\": [\n";
    for (int group_index = 0; group_index < group_count; ++group_index) {
        const int group = group_start + group_index;
        std::vector<Q1Block> blocks(static_cast<size_t>(tile_rows));
        for (int row = 0; row < tile_rows; ++row) {
            if (!read_block(model_path, meta, row0 + row, group, blocks[row])) {
                free_meta(meta);
                return EXIT_FAILURE;
            }
        }

        out << "    [";
        for (int row = 0; row < tile_rows; ++row) {
            out << (row == 0 ? "" : ", ") << "\"" << hex16(blocks[row].scale_fp16) << "\"";
        }
        out << "]" << (group_index == group_count - 1 ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"q8_0_scales_fp16_hex_by_group\": [\n";
    for (int group_index = 0; group_index < group_count; ++group_index) {
        out << "    [";
        for (int q8_block = 0; q8_block < kQ8BlocksPerQ1Group; ++q8_block) {
            out << (q8_block == 0 ? "" : ", ") << "\""
                << hex16(deterministic_q8_scale_fp16(group_index, q8_block)) << "\"";
        }
        out << "]" << (group_index == group_count - 1 ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"activations\": ";
    write_int_array(out, acts);
    out << ",\n";
    out << "  \"transactions\": [\n";

    const int transaction_count = kGroupSize / tile_cols;
    const int total_transactions = group_count * transaction_count;
    int global_txn = 0;
    for (int group_index = 0; group_index < group_count; ++group_index) {
        const int group = group_start + group_index;
        std::vector<Q1Block> blocks(static_cast<size_t>(tile_rows));
        for (int row = 0; row < tile_rows; ++row) {
            if (!read_block(model_path, meta, row0 + row, group, blocks[row])) {
                free_meta(meta);
                return EXIT_FAILURE;
            }
        }
        const auto q8_blocks = q8_raw_blocks(acts, group_index);

        std::vector<std::vector<int>> q8_block_sums(
            static_cast<size_t>(tile_rows),
            std::vector<int>(kQ8BlocksPerQ1Group, 0)
        );

        for (int txn = 0; txn < transaction_count; ++txn) {
            const int group_col = txn * tile_cols;
            const int fixture_col = group_index * kGroupSize + group_col;
            std::vector<int> expected = seeds;
            std::vector<std::vector<int>> weights(
                static_cast<size_t>(tile_rows),
                std::vector<int>(static_cast<size_t>(tile_cols), 0)
            );

            for (int row = 0; row < tile_rows; ++row) {
                for (int col = 0; col < tile_cols; ++col) {
                    weights[row][col] = bit_at(blocks[row], group_col + col);
                    const int contrib = weights[row][col] ? acts[fixture_col + col] : -acts[fixture_col + col];
                    expected[row] += contrib;
                    q8_block_sums[row][group_col / kQ8BlockSize] += contrib;
                }
            }

            out << "    {\n";
            out << "      \"index\": " << global_txn << ",\n";
            out << "      \"group\": " << group << ",\n";
            out << "      \"cols\": [" << fixture_col << ", " << (fixture_col + tile_cols - 1) << "],\n";
            out << "      \"weights\": [";
            for (int row = 0; row < tile_rows; ++row) {
                out << (row == 0 ? "" : ", ") << "[";
                for (int col = 0; col < tile_cols; ++col) {
                    out << (col == 0 ? "" : ", ") << weights[row][col];
                }
                out << "]";
            }
            out << "],\n";
            out << "      \"acts\": [";
            for (int col = 0; col < tile_cols; ++col) {
                out << (col == 0 ? "" : ", ") << acts[fixture_col + col];
            }
            out << "],\n";
            out << "      \"seeds\": ";
            write_int_array(out, seeds);
            out << ",\n";
            out << "      \"expected\": ";
            write_int_array(out, expected);
            out << "\n";
            out << "    }" << (global_txn == (total_transactions - 1) ? "\n" : ",\n");

            seeds = expected;
            ++global_txn;
        }

        for (int row = 0; row < tile_rows; ++row) {
            const float q1_scale = ggml_fp16_to_fp32(blocks[row].scale_fp16);
            for (int q8_block = 0; q8_block < kQ8BlocksPerQ1Group; ++q8_block) {
                const float q8_scale =
                    ggml_fp16_to_fp32(deterministic_q8_scale_fp16(group_index, q8_block));
                scaled_expected[row] +=
                    q1_scale * q8_scale * static_cast<float>(q8_block_sums[row][q8_block]);
            }

            const Q1RawBlock q1_raw = q1_raw_block(blocks[row]);
            float ggml_dot = 0.0f;
            q1_traits->vec_dot(
                kGroupSize,
                &ggml_dot,
                0,
                q1_raw.data(),
                0,
                q8_blocks.data(),
                0,
                1
            );
            ggml_vec_dot_expected[row] += ggml_dot;
        }
    }

    for (int row = 0; row < tile_rows; ++row) {
        const float diff = std::fabs(scaled_expected[row] - ggml_vec_dot_expected[row]);
        if (diff > 1e-4f) {
            std::cerr << "scaled reference does not match ggml vec_dot for row "
                      << row << ": formula=" << scaled_expected[row]
                      << " ggml=" << ggml_vec_dot_expected[row]
                      << " diff=" << diff << "\n";
            free_meta(meta);
            return EXIT_FAILURE;
        }
    }

    out << "  ],\n";
    out << "  \"final_expected\": ";
    write_int_array(out, seeds);
    out << ",\n";
    out << "  \"ggml_scaled_expected_float\": ";
    write_float_array(out, ggml_vec_dot_expected);
    out << "\n";
    out << "}\n";

    free_meta(meta);
    return EXIT_SUCCESS;
}

int usage(const char * program) {
    std::cerr << "usage:\n";
    std::cerr << "  " << program << " inspect [model.gguf] [tensor]\n";
    std::cerr << "  " << program << " generate [model.gguf] [output.json] [tensor] [row0] [group] [tile_rows] [tile_cols]\n";
    std::cerr << "  " << program << " generate-row-tile [model.gguf] [output.json] [tensor] [row0] [tile_rows] [tile_cols]\n";
    return EXIT_FAILURE;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        return usage(argv[0]);
    }

    const std::string mode = argv[1];
    const std::string model_path = argc > 2 ? argv[2] : kDefaultModel;
    const std::string tensor_name =
        mode == "inspect" ? (argc > 3 ? argv[3] : kDefaultTensor) :
        mode == "generate-row-tile" ? (argc > 4 ? argv[4] : kDefaultTensor) :
        (argc > 4 ? argv[4] : kDefaultTensor);

    if (mode == "inspect") {
        return inspect(model_path, tensor_name);
    }

    if (mode == "generate") {
        const std::string output_path =
            argc > 3 ? argv[3] : "test/fixtures/bonsai_blk0_attn_q_r0_r1_g0.json";
        const int row0 = argc > 5 ? std::stoi(argv[5]) : 0;
        const int group = argc > 6 ? std::stoi(argv[6]) : 0;
        const int tile_rows = argc > 7 ? std::stoi(argv[7]) : kDefaultTileRows;
        const int tile_cols = argc > 8 ? std::stoi(argv[8]) : kDefaultTileCols;
        return generate(model_path, output_path, tensor_name, row0, group, 1, tile_rows, tile_cols, false);
    }

    if (mode == "generate-row-tile") {
        const std::string output_path =
            argc > 3 ? argv[3] : "test/fixtures/bonsai_blk0_attn_q_rows0_1_all_groups.json";
        const int row0 = argc > 5 ? std::stoi(argv[5]) : 0;
        const int tile_rows = argc > 6 ? std::stoi(argv[6]) : kDefaultTileRows;
        const int tile_cols = argc > 7 ? std::stoi(argv[7]) : kDefaultTileCols;

        ModelMeta meta;
        if (!load_meta(model_path, tensor_name, meta)) {
            free_meta(meta);
            return EXIT_FAILURE;
        }
        const int group_count = static_cast<int>(meta.tensor->ne[0] / kGroupSize);
        free_meta(meta);

        return generate(model_path, output_path, tensor_name, row0, 0, group_count, tile_rows, tile_cols, true);
    }

    return usage(argv[0]);
}
