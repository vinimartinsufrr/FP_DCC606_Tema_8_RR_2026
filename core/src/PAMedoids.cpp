#include "../include/PAMedoids.hpp"
#include <iostream>
#include <limits>
#include <random>
#include <algorithm>

namespace ClusterEngine {

    // Ordem do initializer list segue a ordem do header para evitar -Wreorder
    PAM::PAM(int k_clusters, float lambda_topologico, int max_steps)
        : k(k_clusters),
          max_passos(max_steps),
          lambda(lambda_topologico) {}

    void PAM::inicializarMedoidesSmart(const std::vector<Point>& dataset, std::vector<size_t>& indices_medoides) {
        indices_medoides.clear();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> dis(0, dataset.size() - 1);

        // 1) Primeiro medoide aleatório
        indices_medoides.push_back(dis(gen));

        // 2) Mantém menor dist² de cada ponto ao conjunto de medoides já escolhidos
        std::vector<float> dists(dataset.size(), std::numeric_limits<float>::max());

        for (int c = 1; c < k; ++c) {
            const size_t ultimo = indices_medoides.back();

            for (size_t i = 0; i < dataset.size(); ++i) {
                const float dist = calcularDistanciaEuclidiana(dataset[i].features, dataset[ultimo].features);
                const float dist2 = dist * dist;
                if (dist2 < dists[i]) dists[i] = dist2;
            }

            // Seleciona próximo medoide proporcional à dist² (espalhamento tipo K-Means++).
            std::discrete_distribution<size_t> dis_prop(dists.begin(), dists.end());
            indices_medoides.push_back(dis_prop(gen));
        }
    }

    float PAM::calcularEnergiaConfiguracao(const std::vector<Point>& dataset,
                                          const std::vector<size_t>& indices_medoides) {
        float energia_total = 0.0f;
        std::vector<int> alocacao_temp(dataset.size(), -1);

        // 1) Atribuição provisória ao medoide mais próximo (distância Euclidiana).
        for (size_t i = 0; i < dataset.size(); ++i) {
            float menor_dist = std::numeric_limits<float>::max();
            int cluster_alocado = -1;

            for (int m = 0; m < k; ++m) {
                const float dist = calcularDistanciaEuclidiana(dataset[i].features,
                                                               dataset[indices_medoides[m]].features);
                if (dist < menor_dist) {
                    menor_dist = dist;
                    cluster_alocado = m;
                }
            }

            // Custo base do PAM:
            // Usamos distância "bruta" (L2) em vez de dist². Isso é uma escolha comum em k-medoids.
            energia_total += menor_dist;
            alocacao_temp[i] = cluster_alocado;
        }

        // 2) Penalidade topológica (Eq. 14) na sequência do dataset:
        // Soma deltaKronecker entre alocações consecutivas i e i+1.
        // Importante: isso só faz sentido se a ordem do dataset tiver algum significado.
        if (lambda > 0.0f && dataset.size() > 1) {
            float penalidade = 0.0f;
            for (size_t i = 0; i < dataset.size() - 1; ++i) {
                penalidade += deltaKronecker(alocacao_temp[i], alocacao_temp[i + 1]);
            }
            energia_total += (lambda * penalidade);
        }

        return energia_total;
    }

    std::vector<Cluster> PAM::executar(std::vector<Point>& dataset) {
        if (dataset.empty()) return {};

        // Boa prática: limpar cluster_id antes (evita "herdar" resultado de execuções anteriores)
        for (auto& p : dataset) p.cluster_id = -1;

        std::vector<size_t> medoides;
        inicializarMedoidesSmart(dataset, medoides);

        float custo_corrente = calcularEnergiaConfiguracao(dataset, medoides);

        bool houve_melhoria = true;
        int passo = 0;

        // Busca local: tenta trocar medoides por não-medoides para reduzir custo.
        while (houve_melhoria && passo < max_passos) {
            houve_melhoria = false;

            size_t melhor_m_idx = 0;
            size_t melhor_p_idx = 0;
            float melhor_reducao = 0.0f;

            for (size_t m = 0; m < medoides.size(); ++m) {
                for (size_t o = 0; o < dataset.size(); ++o) {
                    // Não faz sentido trocar um medoide por ele mesmo/outro medoide já selecionado
                    if (std::find(medoides.begin(), medoides.end(), o) != medoides.end()) continue;

                    std::vector<size_t> medoides_teste = medoides;
                    medoides_teste[m] = o;

                    const float custo_teste = calcularEnergiaConfiguracao(dataset, medoides_teste);
                    const float reducao = custo_corrente - custo_teste;

                    if (reducao > melhor_reducao) {
                        melhor_reducao = reducao;
                        melhor_m_idx = m;
                        melhor_p_idx = o;
                        houve_melhoria = true;
                    }
                }
            }

            if (houve_melhoria) {
                medoides[melhor_m_idx] = melhor_p_idx;
                custo_corrente -= melhor_reducao;
            }

            passo++;
        }

        // ==========================
        // Atribuição final definitiva
        // ==========================
        std::vector<Cluster> clusters(k);
        for (int c = 0; c < k; ++c) {
            clusters[c].id = c;
            clusters[c].center = dataset[medoides[c]].features; // centro = medoide (ponto real)
            clusters[c].point_indices.clear();
        }

        for (size_t i = 0; i < dataset.size(); ++i) {
            float menor_dist = std::numeric_limits<float>::max();
            int melhor_c = -1;

            for (int c = 0; c < k; ++c) {
                const float dist = calcularDistanciaEuclidiana(dataset[i].features, clusters[c].center);
                if (dist < menor_dist) {
                    menor_dist = dist;
                    melhor_c = c;
                }
            }

            dataset[i].cluster_id = melhor_c;
            clusters[melhor_c].point_indices.push_back(i);
        }

        std::cout << "[PAM] Estabilizado. Energia Final (Eq 14): " << custo_corrente << "\n";
        return clusters;
    }

} // namespace ClusterEngine