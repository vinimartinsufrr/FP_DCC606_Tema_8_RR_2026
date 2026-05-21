#ifndef PAMEDOIDS_HPP
#define PAMEDOIDS_HPP

#include "Utils.hpp"
#include <vector>

namespace ClusterEngine {

    /**
     * @brief PAM (Partitioning Around Medoids) com busca local e penalidade topológica (lambda).
     *
     * Diferença para K-Means:
     *  - O "centro" de cada cluster é um medoide: um ponto real do dataset (não uma média).
     *  - É mais robusto a outliers, mas tende a ser muito mais caro computacionalmente,
     *    pois avalia trocas de medoides na busca local.
     *
     * Observação sobre a métrica:
     *  - Nesta implementação, o custo base usa distância Euclidiana "bruta" (não ao quadrado),
     *    que é comum em versões de k-medoids (a literatura varia).
     *  - A penalidade topológica (Eq. 14) entra como lambda * soma(deltaKronecker(...)) na sequência.
     */
    class PAM {
    private:
        // Hiperparâmetro: número de clusters/medoides.
        int k;

        // Hiperparâmetro: limite de passos de busca local (trocas tentadas até estabilizar).
        int max_passos;

        // Penalidade topológica (Eq. 14). lambda = 0 => PAM "nativo".
        float lambda;

        /**
         * @brief Calcula o custo/energia para uma configuração de medoides.
         *
         * Passos:
         *  1) Atribui provisoriamente cada ponto ao medoide mais próximo (por distância Euclidiana).
         *  2) Soma custo base (distância ao medoide mais próximo).
         *  3) Se lambda>0, adiciona penalidade por "quebra" entre i e i+1 na ordem do dataset.
         */
        float calcularEnergiaConfiguracao(const std::vector<Point>& dataset,
                                          const std::vector<size_t>& indices_medoides);

        /**
         * @brief Inicializa medoides usando ideia inspirada em K-Means++ (escolha probabilística).
         *
         * Motivo: reduzir chance de iniciar com medoides muito próximos, melhorando a convergência.
         * Importante: é uma heurística (boa o suficiente para benchmark e aplicação prática).
         */
        void inicializarMedoidesSmart(const std::vector<Point>& dataset,
                                      std::vector<size_t>& indices_medoides);

    public:
        /**
         * @param k_clusters número de medoides (clusters)
         * @param lambda_topologico penalidade topológica (Eq. 14)
         * @param max_steps limite de passos de busca local
         */
        PAM(int k_clusters, float lambda_topologico = 0.0f, int max_steps = 100);

        /**
         * @brief Executa PAM e atribui Point::cluster_id.
         * @return clusters com centers (features do medoide) e índices dos pontos associados.
         */
        std::vector<Cluster> executar(std::vector<Point>& dataset);
    };

} // namespace ClusterEngine

#endif // PAMEDOIDS_HPP