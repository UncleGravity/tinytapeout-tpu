CXX ?= c++
PKG_CONFIG ?= pkg-config

CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS += $(shell $(PKG_CONFIG) --cflags ggml)
LDLIBS += $(shell $(PKG_CONFIG) --libs ggml) -lggml-base -lggml-cpu

TARGET = bonsai_fixture
FIXTURE = ../../test/fixtures/bonsai_blk0_attn_q_r0_r1_g0.json
ROW_TILE_FIXTURE = ../../test/fixtures/bonsai_blk0_attn_q_rows0_1_all_groups.json
TENSOR_FIXTURE = /tmp/tt-tpu-bonsai-full-tensor.json

.PHONY: all clean inspect generate-group generate-row-tile generate-tensor

all: $(TARGET)

$(TARGET): bonsai_fixture.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDLIBS)

inspect: $(TARGET)
	./$(TARGET) inspect ../../models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf blk.0.attn_q.weight

generate-group: $(TARGET)
	./$(TARGET) generate-group ../../models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf $(FIXTURE)

generate-row-tile: $(TARGET)
	./$(TARGET) generate-row-tile ../../models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf $(ROW_TILE_FIXTURE)

generate-tensor: $(TARGET)
	./$(TARGET) generate-tensor ../../models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf $(TENSOR_FIXTURE)

clean:
	$(RM) $(TARGET)
