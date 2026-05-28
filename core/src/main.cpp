#include "../include/ImageLoader.hpp"
#include "../include/FeatureExtractor.hpp"
#include "../include/KMeansPP.hpp"
#include "../include/PAMedoids.hpp"
#include "../include/Utils.hpp"

#include <iostream>
#include <fstream>
#include <filesystem>

using namespace ClusterEngine;

static void exportarResultados(const std::string& out_features,
                               const std::string& out_clusters,
                               const std::vector<Point>& dataset) {
    std::ofstream f_feat(out_features);
    std::ofstream f_clus(out_clusters);

    f_clus << "filepath,cluster_id\n";
    for (const auto& p : dataset) {
        f_clus << p.filepath << "," << p.cluster_id << "\n";

        f_feat << p.filepath;
        for (float v : p.features) f_feat << "," << v;
        f_feat << "\n";
    }
}

int main() {
    std::cout << "===================================================\n";
    std::cout << "=== PROTOCOLO DE COLETA DE METRICAS (TABELA 7) ===\n";
    std::cout << "===================================================\n\n";

    std::cout << "Selecione o modo de execucao:\n";
    std::cout << "[1] Apenas Benchmark de Controle Academico (Corel-1k)\n";
    std::cout << "[2] Execucao Completa (Corel-1k + Acervo Real Eventos)\n";
    std::cout << "Escolha uma opcao (1 ou 2): ";
    
    int opcao = 0;
    std::cin >> opcao;

    if (opcao != 1 && opcao != 2) {
        std::cout << "\n[Erro] Opcao invalida. Encerrando o programa.\n";
        return 1;
    }

    int k_acervo = 3; // Valor padrão inicial
    if (opcao == 2) {
        std::cout << "Digite a quantidade de clusters (K) para o Acervo Real: ";
        std::cin >> k_acervo;
        if (k_acervo <= 0) {
            std::cout << "\n[Erro] O numero de clusters deve ser maior que 0. Encerrando.\n";
            return 1;
        }
    }

    std::cout << "\nIniciando processamento...\n\n";
    std::filesystem::create_directories("output");

    Timer timer_global;
    timer_global.start();

    // =================================================================
    // PARTE 1 e 2: BENCHMARK DE CONTROLE ACADÊMICO (COREL-1K)
    // Executado em ambas as opções (1 e 2)
    // =================================================================
    std::cout << "Carregando Dataset Corel-1k...\n";
    std::vector<Point> corel_data;

    try {
        corel_data = ImageLoader::carregarListaImagens("data/corel1k/lista_imagens.txt");

        FeatureExtractor extrator_corel(4, 9);
        extrator_corel.processarDatasetImagens(corel_data);

        // [1] Corel-1k com K-Means++ (nativo: lambda=0)
        std::cout << "\n[1] Executando K-Means++ no Corel-1k (K=10)...\n";
        KMeansPP kmeans_corel(10, 0.0f); 

        Timer t1;
        t1.start();
        std::vector<Cluster> clusters_kmeans = kmeans_corel.executar(corel_data);
        double tempo_kmeans_corel = t1.elapsedMilliseconds();

        float silhouette_kmeans_corel = calcularSilhouette(corel_data, clusters_kmeans);

        std::cout << "-> Metrica WCSS:       " << calcularWCSS(corel_data, clusters_kmeans) << "\n";
        std::cout << "-> Silhouette Score:   " << silhouette_kmeans_corel << "\n";
        std::cout << "-> Tempo Total:        " << tempo_kmeans_corel << " ms\n";

        exportarResultados("output/corel_kmeans_features.csv",
                           "output/corel_kmeans_clusters.csv",
                           corel_data);
        std::cout << "[Info] Resultados Corel-1k K-Means++ exportados.\n";

        // [2] Corel-1k com PAM (busca local)
        std::cout << "\n[2] Executando PAM (Busca Local) no Corel-1k (K=10)...\n";
        PAM pam_corel(10, 0.01f, 50); 

        Timer t2;
        t2.start();
        std::vector<Cluster> clusters_pam = pam_corel.executar(corel_data);
        double tempo_pam_corel = t2.elapsedMilliseconds();

        float silhouette_pam_corel = calcularSilhouette(corel_data, clusters_pam);

        std::cout << "-> Metrica WCSS:       " << calcularWCSS(corel_data, clusters_pam) << "\n";
        std::cout << "-> Silhouette Score:   " << silhouette_pam_corel << "\n";
        std::cout << "-> Tempo Total:        " << tempo_pam_corel << " ms\n";

    } catch (const std::exception& e) {
        std::cerr << "Erro no Corel-1k: " << e.what() << "\n";
    }

    // =================================================================
    // PARTE 3: APLICAÇÃO PRÁTICA (ACERVO REAL EVENTOS)
    // Executado apenas se o usuário escolher a Opção 2
    // =================================================================
    if (opcao == 2) {
        std::cout << "\n===================================================\n";
        std::cout << "Carregando Dataset Acervo Real Eventos...\n";

        std::vector<Point> acervo_data;
        try {
            std::string lista_acervo = "data/acervo_real/lista_imagens.txt";
            bool prosseguir_acervo = true;

            // Mantém a proteção por Warnings mesmo se o usuário escolheu rodar
            std::ifstream testa_acervo(lista_acervo);
            if (!testa_acervo.good()) {
                std::cout << "[Aviso] Arquivo '" << lista_acervo << "' nao encontrado.\n";
                prosseguir_acervo = false;
            }

            if (prosseguir_acervo) {
                acervo_data = ImageLoader::carregarListaImagens(lista_acervo);
                if (acervo_data.empty()) {
                    std::cout << "[Aviso] Lista do acervo real carregada mas esta vazia.\n";
                    prosseguir_acervo = false;
                }
            }

            if (!prosseguir_acervo) {
                std::cout << "[Aviso] Etapa 3 (Aplicacao Pratica) sera ignorada por falta de dados de entrada.\n";
            } else {
                // Validação para garantir que o K escolhido não seja maior que a quantidade de imagens disponíveis
                if (static_cast<size_t>(k_acervo) > acervo_data.size()) {
                    std::cout << "[Aviso] K=" << k_acervo << " eh maior que o numero de imagens (" << acervo_data.size() 
                              << "). Ajustando K para " << acervo_data.size() << ".\n";
                    k_acervo = acervo_data.size();
                }

                FeatureExtractor extrator_acervo(4, 9);
                extrator_acervo.processarDatasetImagens(acervo_data);

                // Execução do K-Means++ com o K definido pelo usuário
                std::cout << "\n[3] Executando K-Means++ no Acervo Real com K=" << k_acervo << "...\n";
                KMeansPP kmeans_acervo(k_acervo, 0.0f);

                Timer t3;
                t3.start();
                std::vector<Cluster> clusters_acervo = kmeans_acervo.executar(acervo_data);
                double tempo_kmeans_acervo = t3.elapsedMilliseconds();

                float silhouette_kmeans_acervo = calcularSilhouette(acervo_data, clusters_acervo);

                std::cout << "-> Metrica WCSS:       " << calcularWCSS(acervo_data, clusters_acervo) << "\n";
                std::cout << "-> Silhouette Score:   " << silhouette_kmeans_acervo << "\n";
                std::cout << "-> Tempo Total:        " << tempo_kmeans_acervo << " ms\n";

                exportarResultados("output/acervo_features.csv",
                                   "output/acervo_clusters.csv",
                                   acervo_data);
                std::cout << "\n[Info] Resultados do Acervo exportados com sucesso.\n";
            }

        } catch (const std::exception& e) {
            std::cerr << "Erro no Acervo Real: " << e.what() << "\n";
        }
    }

    std::cout << "\n=== TEMPO TOTAL EXECUCAO: " << timer_global.elapsedMilliseconds() << " ms ===\n";
    return 0;
}