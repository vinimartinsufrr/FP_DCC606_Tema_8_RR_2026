#ifndef KMEANSPP_HPP
#define KMEANSPP_HPP

#include "Utils.hpp"
#include <vector>

namespace ClusterEngine {

    /**
     * @brief Implementação de K-Means com inicialização K-Means++ e custo com penalidade topológica (lambda).
     *
     * - K-Means++: escolhe centroides iniciais de forma probabilística para reduzir risco de mínimos ruins.
     * - lambda (Eq. 14): adiciona uma penalidade caso o ponto i fique em cluster diferente do ponto i-1,
     *   modelando uma "vizinhança temporal/ordem" no dataset.
     *
     * Observação: o termo topológico só faz sentido se a ordem do dataset tiver significado
     * (por exemplo, lista cronológica, sequência temporal, ou uma ordenação desejada).
     */
    class KMeansPP {
    private:
        // Hiperparâmetro: número de clusters (k).
        int k;

        // Hiperparâmetro: limite de iterações.
        // Default 300: valor comum na prática para evitar loops longos em casos degenerados.
        int max_iteracoes;

        // Hiperparâmetro: critério de convergência baseado na movimentação do centroide.
        // Default 1e-4: tolerância pequena, suficiente para estabilizar sem gastar iterações demais.
        float tolerancia;

        // Multiplicador da penalidade topológica (Eq. 14).
        // lambda = 0 -> K-Means "nativo" (puro).
        float lambda;

        /**
         * @brief Inicialização K-Means++ ("smart init").
         *
         * Ideia: escolhe o primeiro centro aleatoriamente, e os próximos com probabilidade
         * proporcional à distância² ao centro mais próximo já escolhido. Isso tende a espalhar os centros.
         */
        void inicializarCentroidesSmart(const std::vector<Point>& dataset, std::vector<Cluster>& clusters);

    public:
        /**
         * @param k_clusters número de clusters
         * @param lambda_topologico penalidade topológica (Eq. 14). Use 0.0 para K-Means++ nativo.
         * @param max_iter máximo de iterações
         * @param tol tolerância para convergência
         */
        KMeansPP(int k_clusters, float lambda_topologico = 0.0f, int max_iter = 300, float tol = 1e-4f);

        /**
         * @brief Executa o K-Means++ no dataset e atribui Point::cluster_id.
         * @return vetores de Cluster (centros e índices dos pontos associados).
         */
        std::vector<Cluster> executar(std::vector<Point>& dataset);
    };

} // namespace ClusterEngine

#endif // KMEANSPP_HPP