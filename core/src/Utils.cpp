#include "../include/Utils.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>

namespace ClusterEngine {

    float calcularDistanciaEuclidiana(const std::vector<float>& v1, const std::vector<float>& v2) {
        if (v1.size() != v2.size()) {
            throw std::invalid_argument("Tamanhos incompativeis nos vetores.");
        }

        float soma = 0.0f;
        for (size_t i = 0; i < v1.size(); ++i) {
            const float diff = v1[i] - v2[i];
            soma += diff * diff;
        }
        return std::sqrt(soma);
    }

    float calcularWCSS(const std::vector<Point>& dataset, const std::vector<Cluster>& clusters) {
        float wcss_total = 0.0f;

        for (const auto& cluster : clusters) {
            for (size_t ponto_idx : cluster.point_indices) {
                if (ponto_idx >= dataset.size()) continue;

                const float dist = calcularDistanciaEuclidiana(dataset[ponto_idx].features, cluster.center);
                wcss_total += dist * dist; // "Sum of Squares"
            }
        }

        return wcss_total;
    }

    float calcularEnergiaTotal(const std::vector<Point>& dataset, const std::vector<Cluster>& clusters, float lambda) {
        // Parte "compactação": WCSS (dist^2)
        float energia = calcularWCSS(dataset, clusters);
        if (lambda <= 0.0f) return energia;

        // Parte "topológica": penaliza trocas de rótulo entre vizinhos consecutivos
        if (dataset.size() < 2) return energia;

        float penalidade = 0.0f;
        for (size_t i = 0; i < dataset.size() - 1; ++i) {
            const int c1 = dataset[i].cluster_id;
            const int c2 = dataset[i + 1].cluster_id;

            if (c1 != -1 && c2 != -1) {
                penalidade += deltaKronecker(c1, c2);
            }
        }

        return energia + (lambda * penalidade);
    }

    float calcularSilhouette(const std::vector<Point>& dataset, const std::vector<Cluster>& clusters) {
        if (dataset.empty() || clusters.size() < 2) return 0.0f;

        float soma_silhuetas = 0.0f;
        size_t pontos_validos = 0;

        for (size_t i = 0; i < dataset.size(); ++i) {
            const int c_ponto = dataset[i].cluster_id;
            if (c_ponto == -1 || c_ponto >= static_cast<int>(clusters.size())) continue;

            // Se o cluster tem 0/1 ponto, silhouette não é informativo; consideramos como 0 para o ponto.
            if (clusters[c_ponto].point_indices.size() <= 1) {
                pontos_validos++;
                continue;
            }

            // a(i): distância média para pontos do mesmo cluster
            float soma_dist_interna = 0.0f;
            for (size_t idx_interno : clusters[c_ponto].point_indices) {
                if (i == idx_interno) continue;
                soma_dist_interna += calcularDistanciaEuclidiana(dataset[i].features, dataset[idx_interno].features);
            }
            const float a_i = soma_dist_interna / (clusters[c_ponto].point_indices.size() - 1);

            // b(i): menor distância média para pontos de outros clusters
            float b_i = std::numeric_limits<float>::max();

            for (size_t c = 0; c < clusters.size(); ++c) {
                if (static_cast<int>(c) == c_ponto) continue;
                if (clusters[c].point_indices.empty()) continue;

                float soma_dist_externa = 0.0f;
                for (size_t idx_externo : clusters[c].point_indices) {
                    soma_dist_externa += calcularDistanciaEuclidiana(dataset[i].features, dataset[idx_externo].features);
                }

                const float media_dist_externa = soma_dist_externa / clusters[c].point_indices.size();
                if (media_dist_externa < b_i) b_i = media_dist_externa;
            }

            // s(i) = (b-a)/max(a,b)
            float s_i = 0.0f;
            const float max_val = std::max(a_i, b_i);
            if (max_val > 0.0f) s_i = (b_i - a_i) / max_val;

            soma_silhuetas += s_i;
            pontos_validos++;
        }

        return (pontos_validos == 0) ? 0.0f : (soma_silhuetas / pontos_validos);
    }

} // namespace ClusterEngine