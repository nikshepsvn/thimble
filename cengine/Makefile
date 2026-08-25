CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDLIBS = -lm

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
CFLAGS += -DACCELERATE_NEW_LAPACK
LDLIBS += -framework Accelerate
endif

thimble: thimble.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f thimble

.PHONY: clean

wasm: thimble.c
	emcc thimble.c -O3 -msimd128 -o web/thimble.js \
	  -sMODULARIZE=1 -sEXPORT_NAME=createThimble \
	  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=268435456 \
	  -sSTACK_SIZE=4194304 \
	  -sEXPORTED_FUNCTIONS='["_th_init","_th_call","_malloc","_free"]' \
	  -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","FS"]' \
	  --no-entry

.PHONY: wasm
