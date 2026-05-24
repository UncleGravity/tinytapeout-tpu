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

$(OUT_DIR)/$(LIB): native.c
	@mkdir -p $(OUT_DIR)
	$(CC) $(CFLAGS) $(SHARED) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(OUT_DIR)/$(LIB)

.PHONY: clean
