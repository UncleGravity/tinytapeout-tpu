# Builds libbonsai_ps.so. OUT_DIR is mandatory: the board passes `.` for
# in-place builds, host tests pass a tmpdir. Keeping the source tree free
# of build artifacts keeps the nix flake source clean.

LIB := libbonsai_ps.so

ifndef OUT_DIR
$(error OUT_DIR must be set (e.g. OUT_DIR=. or OUT_DIR=/tmp/build))
endif

CFLAGS  ?= -O3 -std=c99 -fPIC
LDFLAGS ?= -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    SHARED := -dynamiclib
else
    SHARED := -shared
endif

# proto/q1a8_layout.c lives outside this dir but is part of the same .so:
# both PS kernels and the PL matmul driver call its merge_acts entry point.
PROTO_DIR := $(abspath ../../../proto)
INCLUDES  := -I$(PROTO_DIR)/..

SOURCES := native.c $(PROTO_DIR)/q1a8_layout.c

$(OUT_DIR)/$(LIB): $(SOURCES)
	@mkdir -p $(OUT_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(SHARED) -o $@ $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(OUT_DIR)/$(LIB)

.PHONY: clean
