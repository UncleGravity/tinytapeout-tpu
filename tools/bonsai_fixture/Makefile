CXX ?= c++
PKG_CONFIG ?= pkg-config

CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS += $(shell $(PKG_CONFIG) --cflags ggml)
LDLIBS += $(shell $(PKG_CONFIG) --libs ggml) -lggml-base -lggml-cpu

TARGET = bonsai_fixture

# Paths recorded into the fixture as `source.model`. Recipes invoke the tool
# from the repo root so these stay repo-relative rather than ../../-prefixed.
MODEL = models/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf
TENSOR = blk.0.attn_q.weight
FIXTURE_DIR = test/fixtures
TENSOR_FIXTURE = /tmp/tt-tpu-bonsai-full-tensor.json

# Must match the chip tile in src/rtl/project.v; bonsai_fixture.py asserts
# the fixture's `tile` matches the DUT and skips replay otherwise.
TILE_ROWS ?= 2
TILE_COLS ?= 2

TOOL = ./tools/bonsai_fixture/$(TARGET)
RUN  = cd ../.. &&

.PHONY: all clean inspect \
        generate-group generate-group-r42 generate-row-tile generate-tensor \
        regenerate-fixtures

all: $(TARGET)

$(TARGET): bonsai_fixture.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDLIBS)

inspect: $(TARGET)
	$(RUN) $(TOOL) inspect $(MODEL) $(TENSOR)

generate-group: $(TARGET)
	$(RUN) $(TOOL) generate-group $(MODEL) $(FIXTURE_DIR)/bonsai_blk0_attn_q_r0_r1_g0.json $(TENSOR) 0 0 $(TILE_ROWS) $(TILE_COLS)

generate-group-r42: $(TARGET)
	$(RUN) $(TOOL) generate-group $(MODEL) $(FIXTURE_DIR)/bonsai_blk0_attn_q_r42_r43_g7.json $(TENSOR) 42 7 $(TILE_ROWS) $(TILE_COLS)

generate-row-tile: $(TARGET)
	$(RUN) $(TOOL) generate-row-tile $(MODEL) $(FIXTURE_DIR)/bonsai_blk0_attn_q_rows0_1_all_groups.json $(TENSOR) 0 $(TILE_ROWS) $(TILE_COLS)

generate-tensor: $(TARGET)
	$(RUN) $(TOOL) generate-tensor $(MODEL) $(TENSOR_FIXTURE) $(TENSOR) $(TILE_ROWS) $(TILE_COLS)

regenerate-fixtures: generate-group generate-group-r42 generate-row-tile

clean:
	$(RM) $(TARGET)
