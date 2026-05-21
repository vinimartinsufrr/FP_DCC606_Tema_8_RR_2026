CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 $(shell pkg-config --cflags opencv4)
LDFLAGS = $(shell pkg-config --libs opencv4)

SRC_DIR = core/src
INC_DIR = core/include
OUT_DIR = output
TOOLS_DIR = tools

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:.cpp=.o)
TARGET = cluster_engine

# Diretórios/listas padrão (portáveis no repo)
COREL_DIR = data/corel1k
COREL_LIST = $(COREL_DIR)/lista_imagens.txt

ACERVO_DIR ?= data/acervo_real
ACERVO_LIST = $(ACERVO_DIR)/lista_imagens.txt

PYTHON ?= python3

.PHONY: all benchmark pipeline visualize organize \
        list-corel list-acervo \
        clean clean-build clean-output

all: $(OUT_DIR) $(TARGET)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INC_DIR) -c $< -o $@

# -----------------------------
# Geração de listas de imagens
# -----------------------------
# Gera uma lista de caminhos RELATIVOS (ex: data/corel1k/...).
# Motivo: portável entre máquinas; não depende do /home/usuario/...
list-corel:
	@test -d "$(COREL_DIR)" || (echo "Erro: diretorio $(COREL_DIR) nao encontrado" && exit 1)
	@echo "Gerando $(COREL_LIST)..."
	@find "$(COREL_DIR)" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort > "$(COREL_LIST)"
	@echo "OK: $(COREL_LIST) gerado com $$(wc -l < "$(COREL_LIST)") imagens."

list-acervo:
	@test -d "$(ACERVO_DIR)" || (echo "Erro: diretorio $(ACERVO_DIR) nao encontrado. Crie-o ou rode: mkdir -p $(ACERVO_DIR)" && exit 1)
	@echo "Gerando $(ACERVO_LIST) a partir de $(ACERVO_DIR)..."
	@find "$(ACERVO_DIR)" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort > "$(ACERVO_LIST)"
	@echo "OK: $(ACERVO_LIST) gerado com $$(wc -l < "$(ACERVO_LIST)") imagens."

# -----------------------------
# Execução do protocolo (Tabela 7)
# -----------------------------
benchmark: all
	@# Garante que a lista do Corel existe antes de rodar
	@test -f "$(COREL_LIST)" || $(MAKE) list-corel
	@./$(TARGET)

# -----------------------------
# Scripts Python (pós-processamento)
# -----------------------------
visualize:
	@$(PYTHON) $(TOOLS_DIR)/visualizer.py

organize:
	@$(PYTHON) $(TOOLS_DIR)/organizer.py

pipeline: benchmark visualize organize

# -----------------------------
# Limpeza
# -----------------------------
clean-build:
	rm -f $(SRC_DIR)/*.o $(TARGET)

clean-output:
	rm -rf $(OUT_DIR)/*

clean: clean-build clean-output