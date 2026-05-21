# Clusterização Adaptativa de Acervos Visuais (Corel‑1k + Acervo Real)  
**DCC606 — Análise de Algoritmos (UFRR) — Projeto Final (2026)**  
**Tema 8 (RR): Clusterização Adaptativa de Acervos Visuais: uma abordagem baseada em aprendizado não supervisionado e otimização combinatória de borda**

**Autores (Dupla):**
- Vinícius Cavalcante Martins
- José Carvalho Neto

---

## 1. Visão geral

Este projeto implementa um **motor de clusterização não supervisionado** para **organização automática de acervos de imagens**, alinhado ao enunciado do Projeto Final da disciplina **DCC606 (UFRR)** e fundamentado na análise do artigo:

> PAPPAS, T. N.; JAYANT, N. S. *An adaptive clustering algorithm for image segmentation*. ICASSP, 1989. DOI: 10.1109/ICASSP.1989.266767.

A motivação é construir uma solução **eficiente**, aplicável em cenários de **Edge Computing**, capaz de agrupar imagens com base em similaridade **cromática (cores)** e **estrutural (bordas/textura)** sem depender de rotulagem manual ou modelos supervisionados (Deep Learning).

Restrições atendidas (conforme enunciado):
- Implementação **manual** dos algoritmos de clusterização (sem usar `KMeans`/`DBSCAN` prontos de bibliotecas).
- Extração de descritores clássicos a partir de imagens:
  - **Histograma de Cores multicanal**
  - **HOG (Histogram of Oriented Gradients)**
- Estudo comparativo de desempenho com dataset de controle (Corel‑1k) e aplicação prática (Acervo Real).

---

## 2. Estrutura do repositório

```
.
├── core/
│   ├── include/                 # Headers (C++)
│   └── src/                     # Implementações (C++)
├── data/
│   ├── corel1k/                 # dataset de controle acadêmico
│   │   └── lista_imagens.txt     # lista gerada automaticamente (make list-corel)
│   └── acervo_real/             # fotos reais do acervo (não versionadas)
│       └── lista_imagens.txt     # lista gerada automaticamente (make list-acervo)
├── output/                      # artefatos gerados (CSV, PNG, clusters organizados)
├── tools/                       # scripts Python para visualização e organização
│   ├── visualizer.py
│   └── organizer.py
├── docs/                        # relatório final (IEEE/SBC) e anexos
├── Makefile
└── README.md
```

> **Observação sobre datasets:**  
> O **acervo real** não é versionado no GitHub (por tamanho/privacidade). O repositório fornece o pipeline para gerar `lista_imagens.txt` e reproduzir os experimentos localmente.

---

## 3. Dependências

### 3.1 C++ / Compilação
- Linux (testado em ambiente com `g++`)
- `g++` com suporte a **C++17**
- OpenCV 4 (`opencv4`) + `pkg-config`

Instalação típica (Debian/Ubuntu):
```bash
sudo apt update
sudo apt install -y build-essential pkg-config libopencv-dev
```

### 3.2 Python (pós-processamento)
Os scripts em `tools/` são usados apenas para:
- **visualização** (PCA 2D)
- **organização do acervo** em pastas `Cluster_k`

Bibliotecas usadas:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Instalação:
```bash
python3 -m pip install --user pandas numpy matplotlib seaborn scikit-learn
```

> Importante: **scikit-learn NÃO é usado para clusterização**, apenas para PCA/visualização. A clusterização é implementada manualmente em C++.

---

## 4. Algoritmos implementados (Módulo 2)

### 4.1 K-Means++ (variante gulosa estruturada)
Implementação manual do K-Means com inicialização **K-Means++** (semeadura probabilística baseada em distância²), reduzindo o risco de convergência para ótimos locais ruins.

- Métrica reportada: **WCSS** (*Within-Cluster Sum of Squares*).
- Função objetivo padrão:
  - minimiza a soma das distâncias ao quadrado aos centroides.

### 4.2 PAM (Partitioning Around Medoids) — busca local
Implementação manual do algoritmo **PAM**:
- centroides são **medoides** (pontos reais do dataset)
- busca local por **trocas** (swap medoid ↔ non-medoid) para reduzir custo

PAM é tipicamente mais robusto a outliers, porém **muito mais caro** computacionalmente.

---

## 5. Extração de características (Módulo 1)

Para cada imagem, o sistema:
1. Lê a imagem com OpenCV.
2. Redimensiona para **64×64** (padronização para custo fixo e comparabilidade).
3. Extrai:
   - **Histograma de Cores** RGB quantizado
   - **HOG simplificado** (orientações de gradiente)

### 5.1 Histograma de Cores (RGB)
- `bins_canal_cor = 4` por canal  
- total: `4 × 4 × 4 = 64` dimensões

Motivação:
- Dimensão pequena → eficiência (edge/tempo).
- Captura distribuição cromática global.

### 5.2 HOG (Histogram of Oriented Gradients) simplificado
- grade espacial: **4×4** células (fixa)
- bins angulares: `bins_hog = 9`
- total: `4 × 4 × 9 = 144` dimensões

Motivação:
- Captura bordas e textura (componentes estruturais).
- Compromisso custo × qualidade: com 64×64, cada célula fica ~16×16 px.

### 5.3 Dimensionalidade final do vetor de features
- `64 (cor) + 144 (HOG) = 208 dimensões` por imagem.

---

## 6. Penalidade topológica (Eq. 14) / inspiração MRF

O projeto inclui um termo opcional de penalidade `lambda` inspirado em formulações com regularização tipo Potts (delta de Kronecker), aproximando uma ideia de “continuidade” entre vizinhos na sequência do dataset.

- `deltaKronecker(a, b) = 0` se `a == b`, senão `1`.
- Penalidade adicionada considera vizinhança linear `(i, i+1)` na ordem do dataset.

**Observação:**  
Essa penalidade faz mais sentido quando a ordem do dataset possui significado (por exemplo, fotos ordenadas cronologicamente). Para o dataset de controle, frequentemente utiliza-se `lambda=0` (modo nativo).

---

## 7. Como compilar e executar (Makefile)

### 7.1 Compilar
```bash
make
```

### 7.2 Limpar build/saídas
```bash
make clean
```

---

## 8. Datasets e listas de imagens (`lista_imagens.txt`)

O binário lê um arquivo texto com **um caminho por linha**.  
Para automatizar a criação dessas listas, o projeto fornece alvos no Makefile.

### 8.1 Corel‑1k (controle acadêmico)
1) Coloque o dataset em `data/corel1k/` (pode ter subpastas).  
2) Gere/atualize a lista:
```bash
make list-corel
```
Isso cria/atualiza:
- `data/corel1k/lista_imagens.txt`

### 8.2 Acervo real (aplicação prática)
1) Coloque suas imagens em `data/acervo_real/` (pode ter subpastas).  
2) Gere/atualize a lista:
```bash
make list-acervo
```
Isso cria/atualiza:
- `data/acervo_real/lista_imagens.txt`

> Caso `data/acervo_real/lista_imagens.txt` exista mas esteja vazio (0 linhas), o programa usa **fallback** para o Corel‑1k para não quebrar o pipeline.

---

## 9. Protocolo experimental (Tabela 7)

O projeto executa o protocolo de coleta de métricas em três etapas:

1. **Corel‑1k + K-Means++ (nativo: `lambda = 0`)**
2. **Corel‑1k + PAM (busca local)**
3. **Acervo Real + K-Means++ (nativo)**  
   - com fallback para Corel‑1k caso o acervo real não exista ou esteja vazio

Execução:
```bash
make benchmark
```

Saída típica no terminal:
- tempo total por algoritmo (ms)
- WCSS para K-Means++ e PAM
- logs de iterações (K-Means++)

---

## 10. Pipeline completo (C++ + Python)

Executa:
1) benchmark (C++)
2) visualização (PCA em `output/cluster_plot.png`)
3) organização do acervo (cópias em `output/acervo_organizado/Cluster_*`)

```bash
make pipeline
```

Artefatos gerados em `output/`:
- `features.csv`
- `clusters_results.csv`
- `cluster_plot.png`
- `acervo_organizado/Cluster_0/ ... Cluster_(k-1)/`

---

## 11. Como testar com um acervo real

### Opção A (mais simples — recomendada)
1) Colocar imagens em:
```
data/acervo_real/
```

2) Gerar lista automaticamente:
```bash
make list-acervo
```

3) Rodar pipeline:
```bash
make pipeline
```

### Opção B (usar outro diretório fora do repositório)
O Makefile aceita sobrescrever o diretório do acervo real:

```bash
make list-acervo ACERVO_DIR=/caminho/absoluto/para/o/acervo
make pipeline
```

---

## 12. Observações sobre reprodutibilidade

- K-Means++ utiliza inicialização probabilística, então:
  - o número de iterações e o WCSS podem variar levemente a cada execução.
- PAM é significativamente mais caro (tempo alto) para `N=1000` e `k=10`, pois avalia muitas trocas de medoides.

---

## 13. Troubleshooting (erros comuns)

### 13.1 `pkg-config --cflags opencv4` falha
Verifique se OpenCV está instalado e se existe o pacote `opencv4.pc`:
```bash
pkg-config --modversion opencv4
```

### 13.2 Falha ao plotar / `EmptyDataError` no visualizer
Isso ocorre quando `output/features.csv` está vazio. Os scripts em `tools/` detectam esse caso e exibem aviso ao invés de encerrar com erro.

### 13.3 `Acervo real sem imagens (lista vazia)`
Significa que `data/acervo_real/lista_imagens.txt` existe, porém não contém caminhos (ou o diretório está vazio). Para gerar a lista:
```bash
make list-acervo
```

---

## 14. Documentação técnica (docs/)

O relatório técnico (formato IEEE/SBC) e a análise teórica do artigo base estão em:
- `docs/`

O documento inclui:
- descrição do problema e cenário de aplicação
- fundamentação teórica (Pappas & Jayant, 1989)
- análise de complexidade (K-Means++ e PAM)
- descrição modular da implementação (extração de features, motor de clusterização, pós-processamento)
- estudo comparativo de desempenho (Tabela 7) e discussão de resultados
- link/URL do repositório
