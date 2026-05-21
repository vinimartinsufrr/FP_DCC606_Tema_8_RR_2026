#include "../include/KMeansPP.hpp"
#include <iostream>
#include <random>
#include <limits>

namespace ClusterEngine {

    // Importante: a ordem no initializer list deve seguir a ordem de declaração no header
    // para evitar warnings -Wreorder.
    KMeansPP::KMeansPP(int k_clusters, float lambda_topologico, int max_iter, float tol)
        : k(k_clusters),
          max_iteracoes(max_iter),
          tolerancia(tol),
          lambda(lambda_topologico) {}

    void KMeansPP::inicializarCentroidesSmart(const std::vector<Point>& dataset, std::vector<Cluster>& clusters) {
        clusters.clear();
        clusters.resize(k);

        // Garantimos IDs estáveis (útil para deltaKronecker / penalidade topológica).
        for (int i = 0; i < k; ++i) clusters[i].id = i;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, dataset.size() - 1);

        // 1) Primeiro centro aleatório
        clusters[0].center = dataset[dis(gen)].features;

        // 2) Para cada ponto, mantemos a menor distância² ao conjunto de centros já escolhidos
        std::vector<float> dists(dataset.size(), std::numeric_limits<float>::max());

        for (int c = 1; c < k; ++c) {
            for (size_t i = 0; i < dataset.size(); ++i) {
                const float dist = calcularDistanciaEuclidiana(dataset[i].features, clusters[c - 1].center);
                const float dist2 = dist * dist;
                if (dist2 < dists[i]) dists[i] = dist2;
            }

            // Escolhe próximo centro com probabilidade proporcional a dist².
            std::discrete_distribution<size_t> dis_prop(dists.begin(), dists.end());
            clusters[c].center = dataset[dis_prop(gen)].features;
        }
    }

    std::vector<Cluster> KMeansPP::executar(std::vector<Point>& dataset) {
        if (dataset.empty()) return {};

        std::vector<Cluster> clusters;
        inicializarCentroidesSmart(dataset, clusters);

        bool convergiu = false;
        int iteracao = 0;
        const size_t dimensao = dataset[0].features.size();

        // Boa prática: limpar cluster_id antes (evita "herdar" resultado de execuções anteriores)
        for (auto& p : dataset) p.cluster_id = -1;

        while (iteracao < max_iteracoes && !convergiu) {
            for (auto& cluster : clusters) cluster.point_indices.clear();

            // ==========================
            // PASSO 1: ATRIBUIÇÃO
            // ==========================
            // Para cada ponto, escolhe o cluster com menor energia:
            // energia = dist^2 + lambda * deltaKronecker(cluster_{i-1}, cluster_atual)
            for (size_t i = 0; i < dataset.size(); ++i) {
                float menor_energia = std::numeric_limits<float>::max();
                int melhor_cluster = -1;

                for (int c = 0; c < k; ++c) {
                    const float dist = calcularDistanciaEuclidiana(dataset[i].features, clusters[c].center);
                    float energia = dist * dist;

                    // Penalidade topológica (Eq. 14): depende da ordem do dataset
                    if (lambda > 0.0f && i > 0 && dataset[i - 1].cluster_id != -1) {
                        energia += lambda * deltaKronecker(dataset[i - 1].cluster_id, c);
                    }

                    if (energia < menor_energia) {
                        menor_energia = energia;
                        melhor_cluster = c;
                    }
                }

                dataset[i].cluster_id = melhor_cluster;
                clusters[melhor_cluster].point_indices.push_back(i);
            }

            // ==========================
            // PASSO 2: ATUALIZAÇÃO
            // ==========================
            convergiu = true;

            for (int c = 0; c < k; ++c) {
                if (clusters[c].point_indices.empty()) continue;

                std::vector<float> novo_centro(dimensao, 0.0f);
                for (size_t p_idx : clusters[c].point_indices) {
                    for (size_t d = 0; d < dimensao; ++d) novo_centro[d] += dataset[p_idx].features[d];
                }

                const float total = static_cast<float>(clusters[c].point_indices.size());
                for (size_t d = 0; d < dimensao; ++d) novo_centro[d] /= total;

                // Convergência: se algum centro se moveu mais que a tolerância, continua iterando
                if (calcularDistanciaEuclidiana(clusters[c].center, novo_centro) > tolerancia) {
                    convergiu = false;
                }
                clusters[c].center = std::move(novo_centro);
            }

            iteracao++;
        }

        std::cout << "[K-Means++] Finalizado. Iteracoes: " << iteracao
                  << " | Energia (Eq 14): " << calcularEnergiaTotal(dataset, clusters, lambda) << "\n";

        return clusters;
    }

} // namespace ClusterEngine