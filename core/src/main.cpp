#include "../include/ImageLoader.hpp"
#include "../include/FeatureExtractor.hpp"
#include "../include/KMeansPP.hpp"
#include "../include/PAMedoids.hpp"
#include "../include/Utils.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>  // C++17: para garantir que output/ exista

using namespace ClusterEngine;

/**
 * @brief Exporta:
 *  - features.csv: cada linha = filepath + vetor de features (usado pelo visualizer.py)
 *  - clusters_results.csv: filepath + cluster_id (usado pelo organizer.py)
 *
 * Observação: o Python é responsável por criar pastas Cluster_0..Cluster_(k-1) e copiar imagens.
 * O C++ só salva o "mapa" (CSV) de qual imagem pertence a qual cluster.
 */
static void exportarResultados(const std::string& out_features,
                               const std::string& out_clusters,
                               const std::vector<Point>& dataset) {
    std::ofstream f_feat(out_features);
    std::ofstream f_clus(out_clusters);

    f_clus << "filepath,cluster_id\n";
    for (const auto& p : dataset) {
        f_clus << p.filepath << "," << p.cluster_id << "\n";

        // features.csv: primeira coluna é o filepath (para permitir rastrear pontos)
        f_feat << p.filepath;
        for (float v : p.features) f_feat << "," << v;
        f_feat << "\n";
    }
}

int main() {
    std::cout << "=== PROTOCOLO DE COLETA DE METRICAS (TABELA 7) ===\n\n";

    // Garante que a pasta output/ exista (evita falhas ao salvar CSVs)
    std::filesystem::create_directories("output");

    Timer timer_global;
    timer_global.start();

    // =================================================================
    // PARTE 1 e 2: BENCHMARK DE CONTROLE ACADÊMICO (COREL-1K)
    // Tabela 7:
    //  - Corel-1k com K-Means++ (nativo)
    //  - Corel-1k com PAM (busca local)
    // =================================================================
    std::cout << "Carregando Dataset Corel-1k...\n";
    std::vector<Point> corel_data;

    try {
        corel_data = ImageLoader::carregarListaImagens("data/corel1k/lista_imagens.txt");

        // Hiperparâmetros de features (escolhas de engenharia):
        //  - bins_canal_cor=4 => 4^3 = 64 dimensões de cor
        //  - bins_hog=9 e grade 4x4 (no FeatureExtractor) => 4*4*9 = 144 dimensões de HOG
        // Total: 208 dimensões por imagem
        FeatureExtractor extrator_corel(4, 9);
        extrator_corel.processarDatasetImagens(corel_data);

        // [1] Corel-1k com K-Means++ (nativo: lambda=0)
        std::cout << "\n[1] Executando K-Means++ no Corel-1k...\n";
        KMeansPP kmeans_corel(10, 0.0f); // k=10 pois o Corel-1k possui 10 categorias de referência

        Timer t1;
        t1.start();
        std::vector<Cluster> clusters_kmeans = kmeans_corel.executar(corel_data);

        std::cout << "-> Metrica WCSS: " << calcularWCSS(corel_data, clusters_kmeans) << "\n";
        std::cout << "-> Tempo Total: " << t1.elapsedMilliseconds() << " ms\n";

        // [2] Corel-1k com PAM (busca local)
        std::cout << "\n[2] Executando PAM (Busca Local) no Corel-1k...\n";
        PAM pam_corel(10, 0.01f, 50); // lambda pequeno para não "colapsar" clusters; max_steps=50

        Timer t2;
        t2.start();
        std::vector<Cluster> clusters_pam = pam_corel.executar(corel_data);

        // Atenção: WCSS usa dist² ao centro. No PAM o custo interno é dist "bruta".
        // Mesmo assim, calcular WCSS é útil para comparação uniforme (Tabela 7).
        std::cout << "-> Metrica WCSS: " << calcularWCSS(corel_data, clusters_pam) << "\n";
        std::cout << "-> Tempo Total: " << t2.elapsedMilliseconds() << " ms\n";

    } catch (const std::exception& e) {
        std::cerr << "Erro no Corel-1k: " << e.what() << "\n";
    }

    // =================================================================
    // PARTE 3: APLICAÇÃO PRÁTICA (ACERVO REAL EVENTOS)
    // Tabela 7:
    //  - Acervo Real Eventos com K-Means++ (nativo)
    //
    // Requisito do projeto:
    //  - Se o acervo real não existir OU estiver vazio (lista vazia),
    //    usar Corel-1k como fallback para não quebrar o pipeline.
    // =================================================================
    std::cout << "\n===================================================\n";
    std::cout << "Carregando Dataset Acervo Real Eventos...\n";

    std::vector<Point> acervo_data;
    try {
        std::string lista_acervo = "data/acervo_real/lista_imagens.txt";
        const std::string lista_corel  = "data/corel1k/lista_imagens.txt";

        // (A) Se o arquivo do acervo não existe: fallback imediato
        {
            std::ifstream testa_acervo(lista_acervo);
            if (!testa_acervo.good()) {
                std::cout << "[Aviso] Lista do acervo real nao encontrada. Usando Corel-1k como fallback para a aplicacao pratica.\n";
                lista_acervo = lista_corel;
            }
        }

        // (B) Carrega a lista escolhida
        acervo_data = ImageLoader::carregarListaImagens(lista_acervo);

        // (C) Se a lista existir mas estiver vazia: fallback para Corel-1k
        if (acervo_data.empty()) {
            std::cout << "[Aviso] Acervo real sem imagens (lista vazia). Usando Corel-1k como fallback para a aplicacao pratica.\n";
            lista_acervo = lista_corel;
            acervo_data = ImageLoader::carregarListaImagens(lista_acervo);
        }

        // Se ainda assim estiver vazio, não faz sentido prosseguir
        if (acervo_data.empty()) {
            std::cout << "[Aviso] Dataset de aplicacao pratica ainda vazio. Etapa 3 sera ignorada.\n";
        } else {
            FeatureExtractor extrator_acervo(4, 9);
            extrator_acervo.processarDatasetImagens(acervo_data);

            // [3] Acervo Real com K-Means++ (nativo)
            std::cout << "\n[3] Executando K-Means++ no Acervo Real...\n";
            KMeansPP kmeans_acervo(10, 0.0f);

            Timer t3;
            t3.start();
            std::vector<Cluster> clusters_acervo = kmeans_acervo.executar(acervo_data);

            std::cout << "-> Metrica WCSS: " << calcularWCSS(acervo_data, clusters_acervo) << "\n";
            std::cout << "-> Tempo Total: " << t3.elapsedMilliseconds() << " ms\n";

            // Exporta o "mapa" para os scripts Python (visualização/organização em pastas)
            exportarResultados("output/features.csv", "output/clusters_results.csv", acervo_data);
            std::cout << "\n[Info] Resultados do Acervo exportados para reordenacao em diretorios.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Erro no Acervo Real: " << e.what() << "\n";
    }

    std::cout << "\n=== TEMPO TOTAL EXECUCAO: " << timer_global.elapsedMilliseconds() << " ms ===\n";
    return 0;
}