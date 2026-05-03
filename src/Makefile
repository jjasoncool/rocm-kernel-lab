HIPCC = hipcc
HIPCC_FLAGS = -O2 -std=c++17

SRC_DIR = .
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
BINS = $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%, $(SRCS))

all: $(BIN_DIR) $(BINS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: $(SRC_DIR)/%.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
