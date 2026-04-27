#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

namespace {

constexpr const char * DEFAULT_MODEL = "models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf";
constexpr const char * DEFAULT_TENSOR = "blk.0.attn_q.weight";
constexpr int DEFAULT_TILE_ROWS = 2;
constexpr int DEFAULT_TILE_COLS = 4;
constexpr int Q1_GROUP_SIZE = 128;
constexpr int Q8_BLOCK_SIZE = 32;
constexpr int Q8_BLOCKS_PER_Q1 = Q1_GROUP_SIZE / Q8_BLOCK_SIZE;
constexpr int Q1_BLOCK_BYTES = 18;
constexpr int Q8_BLOCK_BYTES = 2 + Q8_BLOCK_SIZE;

using Q1Raw = std::array<uint8_t, Q1_BLOCK_BYTES>;
using Q8Raw = std::array<uint8_t, Q8_BLOCK_BYTES>;

struct Tile {
    int rows = DEFAULT_TILE_ROWS;
    int cols = DEFAULT_TILE_COLS;
};

struct Selection {
    int row0 = 0;
    int group0 = 0;
    int group_count = 1;
    bool all_groups = false;
};

struct Config {
    std::string model = DEFAULT_MODEL;
    std::string output;
    std::string tensor = DEFAULT_TENSOR;
    Tile tile;
    Selection select;
};

struct TensorMeta {
    gguf_context * gguf = nullptr;
    ggml_context * ggml = nullptr;
    ggml_tensor * tensor = nullptr;
    int64_t tensor_id = -1;

    ~TensorMeta() {
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }

    TensorMeta() = default;
    TensorMeta(const TensorMeta &) = delete;
    TensorMeta & operator=(const TensorMeta &) = delete;
    TensorMeta(TensorMeta && other) noexcept
        : gguf(other.gguf), ggml(other.ggml), tensor(other.tensor), tensor_id(other.tensor_id) {
        other.gguf = nullptr;
        other.ggml = nullptr;
        other.tensor = nullptr;
        other.tensor_id = -1;
    }
    TensorMeta & operator=(TensorMeta &&) = delete;
};

struct Q1Block {
    uint16_t d = 0;
    std::array<uint8_t, 16> qs{};
};

struct Reference {
    std::vector<int> integer_final;
    std::vector<float> scaled_float;
};

int8_t activation_qs(int index) {
    return static_cast<int8_t>(((index * 37 + 19) % 253) - 126);
}

uint16_t q8_scale_fp16(int group_index, int q8_block) {
    const float scale = 0.015625f * static_cast<float>(1 + ((group_index * 7 + q8_block * 3) % 11));
    return ggml_fp32_to_fp16(scale);
}

std::string hex16(uint16_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0') << std::setw(4) << value;
    return out.str();
}

template <typename T>
void write_array(std::ostream & out, const std::vector<T> & values) {
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

TensorMeta load_tensor(const std::string & model, const std::string & tensor_name) {
    TensorMeta meta;
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &meta.ggml;
    meta.gguf = gguf_init_from_file(model.c_str(), params);
    if (meta.gguf == nullptr || meta.ggml == nullptr) {
        throw std::runtime_error("failed to load GGUF metadata: " + model);
    }

    meta.tensor_id = gguf_find_tensor(meta.gguf, tensor_name.c_str());
    if (meta.tensor_id < 0) {
        throw std::runtime_error("tensor not found: " + tensor_name);
    }

    meta.tensor = ggml_get_tensor(meta.ggml, tensor_name.c_str());
    if (meta.tensor == nullptr) {
        throw std::runtime_error("tensor metadata missing from ggml context: " + tensor_name);
    }
    if (meta.tensor->type != GGML_TYPE_Q1_0) {
        throw std::runtime_error(std::string("expected q1_0 tensor, got ") + ggml_type_name(meta.tensor->type));
    }
    return meta;
}

void validate(const Config & cfg, const TensorMeta & meta) {
    const int row_last = cfg.select.row0 + cfg.tile.rows - 1;
    const int groups_per_row = static_cast<int>(meta.tensor->ne[0] / Q1_GROUP_SIZE);

    if (cfg.tile.rows <= 0 || cfg.tile.cols <= 0) {
        throw std::runtime_error("tile rows/cols must be positive");
    }
    if (Q1_GROUP_SIZE % cfg.tile.cols != 0) {
        throw std::runtime_error("tile cols must divide Q1_0 group size");
    }
    if (Q8_BLOCK_SIZE % cfg.tile.cols != 0) {
        throw std::runtime_error("tile cols must divide Q8_0 block size");
    }
    if (meta.tensor->ne[0] % Q1_GROUP_SIZE != 0) {
        throw std::runtime_error("tensor width is not divisible by Q1_0 group size");
    }
    if (cfg.select.row0 < 0 || row_last >= meta.tensor->ne[1]) {
        throw std::runtime_error("selected row tile is out of range");
    }
    if (cfg.select.group0 < 0 || cfg.select.group_count <= 0 ||
        cfg.select.group0 + cfg.select.group_count > groups_per_row) {
        throw std::runtime_error("selected group range is out of range");
    }
}

Q1Block read_q1_block(std::ifstream & file, const TensorMeta & meta, int row, int group) {
    const size_t tensor_offset = gguf_get_data_offset(meta.gguf) + gguf_get_tensor_offset(meta.gguf, meta.tensor_id);
    const size_t block_offset = tensor_offset + row * meta.tensor->nb[1] + group * Q1_BLOCK_BYTES;

    std::array<uint8_t, Q1_BLOCK_BYTES> raw{};
    file.seekg(static_cast<std::streamoff>(block_offset));
    file.read(reinterpret_cast<char *>(raw.data()), raw.size());
    if (!file) {
        throw std::runtime_error("failed to read Q1_0 block");
    }

    Q1Block block;
    block.d = static_cast<uint16_t>(raw[0]) | (static_cast<uint16_t>(raw[1]) << 8);
    for (int i = 0; i < 16; ++i) {
        block.qs[i] = raw[2 + i];
    }
    return block;
}

int q1_bit(const Q1Block & block, int bit) {
    return (block.qs[bit / 8] >> (bit % 8)) & 1;
}

Q1Raw q1_raw(const Q1Block & block) {
    Q1Raw raw{};
    raw[0] = static_cast<uint8_t>(block.d & 0xFF);
    raw[1] = static_cast<uint8_t>(block.d >> 8);
    for (int i = 0; i < 16; ++i) {
        raw[2 + i] = block.qs[i];
    }
    return raw;
}

std::array<Q8Raw, Q8_BLOCKS_PER_Q1> q8_raw_blocks(const std::vector<int> & acts, int group_index) {
    std::array<Q8Raw, Q8_BLOCKS_PER_Q1> blocks{};
    for (int q8 = 0; q8 < Q8_BLOCKS_PER_Q1; ++q8) {
        const uint16_t scale = q8_scale_fp16(group_index, q8);
        blocks[q8][0] = static_cast<uint8_t>(scale & 0xFF);
        blocks[q8][1] = static_cast<uint8_t>(scale >> 8);

        const int base = group_index * Q1_GROUP_SIZE + q8 * Q8_BLOCK_SIZE;
        for (int i = 0; i < Q8_BLOCK_SIZE; ++i) {
            blocks[q8][2 + i] = static_cast<uint8_t>(static_cast<int8_t>(acts[base + i]));
        }
    }
    return blocks;
}

std::vector<int> activations(const Selection & sel) {
    std::vector<int> acts;
    acts.reserve(static_cast<size_t>(sel.group_count * Q1_GROUP_SIZE));
    for (int i = 0; i < sel.group_count * Q1_GROUP_SIZE; ++i) {
        acts.push_back(activation_qs(i));
    }
    return acts;
}

float ggml_vec_dot_q1_q8(const ggml_type_traits_cpu * traits, const Q1Block & q1, const std::vector<int> & acts, int group_index) {
    const auto q1_bytes = q1_raw(q1);
    const auto q8_bytes = q8_raw_blocks(acts, group_index);
    float result = 0.0f;
    traits->vec_dot(Q1_GROUP_SIZE, &result, 0, q1_bytes.data(), 0, q8_bytes.data(), 0, 1);
    return result;
}

void write_scales(std::ostream & out, std::ifstream & model, const TensorMeta & meta, const Config & cfg) {
    out << "  \"quantization\": {\n";
    out << "    \"q1_0_group_size\": " << Q1_GROUP_SIZE << ",\n";
    out << "    \"q8_0_block_size\": " << Q8_BLOCK_SIZE << ",\n";

    out << "    \"q1_scales_fp16_hex_by_group\": [\n";
    for (int gi = 0; gi < cfg.select.group_count; ++gi) {
        const int group = cfg.select.group0 + gi;
        out << "      [";
        for (int row = 0; row < cfg.tile.rows; ++row) {
            const auto block = read_q1_block(model, meta, cfg.select.row0 + row, group);
            out << (row == 0 ? "" : ", ") << "\"" << hex16(block.d) << "\"";
        }
        out << "]" << (gi + 1 == cfg.select.group_count ? "\n" : ",\n");
    }
    out << "    ],\n";

    out << "    \"q8_scales_fp16_hex_by_group\": [\n";
    for (int gi = 0; gi < cfg.select.group_count; ++gi) {
        out << "      [";
        for (int q8 = 0; q8 < Q8_BLOCKS_PER_Q1; ++q8) {
            out << (q8 == 0 ? "" : ", ") << "\"" << hex16(q8_scale_fp16(gi, q8)) << "\"";
        }
        out << "]" << (gi + 1 == cfg.select.group_count ? "\n" : ",\n");
    }
    out << "    ]\n";
    out << "  },\n";
}

Reference write_transactions(
    std::ostream & out,
    std::ifstream & model,
    const TensorMeta & meta,
    const Config & cfg,
    const std::vector<int> & acts
) {
    const auto * traits = ggml_get_type_traits_cpu(GGML_TYPE_Q1_0);
    if (traits == nullptr || traits->vec_dot == nullptr || traits->vec_dot_type != GGML_TYPE_Q8_0) {
        throw std::runtime_error("GGML Q1_0 x Q8_0 CPU vec_dot is unavailable");
    }

    Reference ref;
    ref.integer_final = std::vector<int>(static_cast<size_t>(cfg.tile.rows), 0);
    ref.scaled_float = std::vector<float>(static_cast<size_t>(cfg.tile.rows), 0.0f);

    out << "  \"transactions\": [\n";
    const int txns_per_group = Q1_GROUP_SIZE / cfg.tile.cols;
    const int total_txns = cfg.select.group_count * txns_per_group;
    int txn_index = 0;

    for (int gi = 0; gi < cfg.select.group_count; ++gi) {
        const int group = cfg.select.group0 + gi;
        std::vector<Q1Block> blocks;
        blocks.reserve(static_cast<size_t>(cfg.tile.rows));
        for (int row = 0; row < cfg.tile.rows; ++row) {
            blocks.push_back(read_q1_block(model, meta, cfg.select.row0 + row, group));
        }

        for (int row = 0; row < cfg.tile.rows; ++row) {
            ref.scaled_float[row] += ggml_vec_dot_q1_q8(traits, blocks[row], acts, gi);
        }

        for (int tx = 0; tx < txns_per_group; ++tx, ++txn_index) {
            const int group_col = tx * cfg.tile.cols;
            const int fixture_col = gi * Q1_GROUP_SIZE + group_col;
            std::vector<int> next = ref.integer_final;

            out << "    {\n";
            out << "      \"index\": " << txn_index << ",\n";
            out << "      \"group\": " << group << ",\n";
            out << "      \"cols\": [" << fixture_col << ", " << (fixture_col + cfg.tile.cols - 1) << "],\n";
            out << "      \"weights\": [";
            for (int row = 0; row < cfg.tile.rows; ++row) {
                out << (row == 0 ? "" : ", ") << "[";
                for (int col = 0; col < cfg.tile.cols; ++col) {
                    const int bit = q1_bit(blocks[row], group_col + col);
                    out << (col == 0 ? "" : ", ") << bit;
                    next[row] += bit ? acts[fixture_col + col] : -acts[fixture_col + col];
                }
                out << "]";
            }
            out << "],\n";

            out << "      \"acts\": [";
            for (int col = 0; col < cfg.tile.cols; ++col) {
                out << (col == 0 ? "" : ", ") << acts[fixture_col + col];
            }
            out << "],\n";

            out << "      \"seeds\": ";
            write_array(out, ref.integer_final);
            out << ",\n";
            out << "      \"expected\": ";
            write_array(out, next);
            out << "\n";
            out << "    }" << (txn_index + 1 == total_txns ? "\n" : ",\n");

            ref.integer_final = std::move(next);
        }
    }
    out << "  ],\n";
    return ref;
}

void write_fixture(const Config & cfg) {
    auto meta = load_tensor(cfg.model, cfg.tensor);
    validate(cfg, meta);

    std::ifstream model(cfg.model, std::ios::binary);
    if (!model) {
        throw std::runtime_error("failed to open model: " + cfg.model);
    }

    std::ofstream out(cfg.output);
    if (!out) {
        throw std::runtime_error("failed to open output: " + cfg.output);
    }

    const int row_last = cfg.select.row0 + cfg.tile.rows - 1;
    const int last_col = cfg.select.group_count * Q1_GROUP_SIZE - 1;
    const auto acts = activations(cfg.select);

    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"source\": {\n";
    out << "    \"model\": \"" << cfg.model << "\",\n";
    out << "    \"tensor\": \"" << cfg.tensor << "\",\n";
    out << "    \"type\": \"" << ggml_type_name(meta.tensor->type) << "\",\n";
    out << "    \"shape\": [" << meta.tensor->ne[0] << ", " << meta.tensor->ne[1] << "],\n";
    out << "    \"row_stride_bytes\": " << meta.tensor->nb[1] << "\n";
    out << "  },\n";
    out << "  \"tile\": {\"rows\": " << cfg.tile.rows << ", \"cols\": " << cfg.tile.cols << "},\n";
    out << "  \"selection\": {\"rows\": [" << cfg.select.row0 << ", " << row_last << "], ";
    if (cfg.select.all_groups) {
        out << "\"groups\": [" << cfg.select.group0 << ", " << (cfg.select.group0 + cfg.select.group_count - 1)
            << "], \"cols\": [0, " << last_col << "]},\n";
    } else {
        out << "\"group\": " << cfg.select.group0 << ", \"cols\": [0, 127]},\n";
    }
    write_scales(out, model, meta, cfg);
    out << "  \"activations\": ";
    write_array(out, acts);
    out << ",\n";

    auto ref = write_transactions(out, model, meta, cfg, acts);
    out << "  \"reference\": {\n";
    out << "    \"integer_final\": ";
    write_array(out, ref.integer_final);
    out << ",\n";
    out << "    \"ggml_scaled_float\": ";
    write_float_array(out, ref.scaled_float);
    out << "\n";
    out << "  }\n";
    out << "}\n";
}

void inspect(const std::string & model, const std::string & tensor_name) {
    auto meta = load_tensor(model, tensor_name);
    std::cout << "model: " << model << "\n";
    std::cout << "gguf_version: " << gguf_get_version(meta.gguf) << "\n";
    std::cout << "data_offset: " << gguf_get_data_offset(meta.gguf) << "\n";
    std::cout << "tensor: " << tensor_name << "\n";
    std::cout << "type: " << ggml_type_name(meta.tensor->type) << "\n";
    std::cout << "size_bytes: " << gguf_get_tensor_size(meta.gguf, meta.tensor_id) << "\n";
    std::cout << "file_data_offset: "
              << (gguf_get_data_offset(meta.gguf) + gguf_get_tensor_offset(meta.gguf, meta.tensor_id)) << "\n";
    std::cout << "ne: [" << meta.tensor->ne[0] << ", " << meta.tensor->ne[1]
              << ", " << meta.tensor->ne[2] << ", " << meta.tensor->ne[3] << "]\n";
    std::cout << "nb: [" << meta.tensor->nb[0] << ", " << meta.tensor->nb[1]
              << ", " << meta.tensor->nb[2] << ", " << meta.tensor->nb[3] << "]\n";
}

int groups_per_row(const std::string & model, const std::string & tensor_name) {
    auto meta = load_tensor(model, tensor_name);
    return static_cast<int>(meta.tensor->ne[0] / Q1_GROUP_SIZE);
}

[[noreturn]] void usage(const char * program) {
    std::cerr << "usage:\n";
    std::cerr << "  " << program << " inspect [model.gguf] [tensor]\n";
    std::cerr << "  " << program << " generate-group [model.gguf] [output.json] [tensor] [row0] [group] [tile_rows] [tile_cols]\n";
    std::cerr << "  " << program << " generate-row-tile [model.gguf] [output.json] [tensor] [row0] [tile_rows] [tile_cols]\n";
    std::exit(EXIT_FAILURE);
}

Config group_config(int argc, char ** argv) {
    Config cfg;
    cfg.model = argc > 2 ? argv[2] : DEFAULT_MODEL;
    cfg.output = argc > 3 ? argv[3] : "test/fixtures/bonsai_blk0_attn_q_r0_r1_g0.json";
    cfg.tensor = argc > 4 ? argv[4] : DEFAULT_TENSOR;
    cfg.select.row0 = argc > 5 ? std::stoi(argv[5]) : 0;
    cfg.select.group0 = argc > 6 ? std::stoi(argv[6]) : 0;
    cfg.tile.rows = argc > 7 ? std::stoi(argv[7]) : DEFAULT_TILE_ROWS;
    cfg.tile.cols = argc > 8 ? std::stoi(argv[8]) : DEFAULT_TILE_COLS;
    return cfg;
}

Config row_tile_config(int argc, char ** argv) {
    Config cfg;
    cfg.model = argc > 2 ? argv[2] : DEFAULT_MODEL;
    cfg.output = argc > 3 ? argv[3] : "test/fixtures/bonsai_blk0_attn_q_rows0_1_all_groups.json";
    cfg.tensor = argc > 4 ? argv[4] : DEFAULT_TENSOR;
    cfg.select.row0 = argc > 5 ? std::stoi(argv[5]) : 0;
    cfg.tile.rows = argc > 6 ? std::stoi(argv[6]) : DEFAULT_TILE_ROWS;
    cfg.tile.cols = argc > 7 ? std::stoi(argv[7]) : DEFAULT_TILE_COLS;
    cfg.select.group_count = groups_per_row(cfg.model, cfg.tensor);
    cfg.select.all_groups = true;
    return cfg;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 2) {
            usage(argv[0]);
        }

        const std::string mode = argv[1];
        if (mode == "inspect") {
            inspect(argc > 2 ? argv[2] : DEFAULT_MODEL, argc > 3 ? argv[3] : DEFAULT_TENSOR);
            return EXIT_SUCCESS;
        }
        if (mode == "generate-group") {
            write_fixture(group_config(argc, argv));
            return EXIT_SUCCESS;
        }
        if (mode == "generate-row-tile") {
            write_fixture(row_tile_config(argc, argv));
            return EXIT_SUCCESS;
        }

        usage(argv[0]);
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
