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
