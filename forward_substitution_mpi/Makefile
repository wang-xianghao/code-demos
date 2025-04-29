CC = mpicc
CFLAGS = -O2 -Wall -std=c99
LDFLAGS = -lm

SRC = solve.c
EXEC = solve


all: $(EXEC)

$(EXEC): $(SRC)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(EXEC)

.PHONY: all clean