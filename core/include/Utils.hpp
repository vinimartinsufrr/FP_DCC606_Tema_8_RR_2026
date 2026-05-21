#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <chrono>

namespace ClusterEngine {

    /**
     * @brief Representa uma instância do dataset (uma imagem).
     *
     * - features: vetor numérico que descreve a imagem (ex.: histograma de cores + HOG).
     *   O tamanho desse vetor é a "dimensionalidade" do ponto no espaço de características.
     * - cluster_id: rótulo atribuído pelo algoritmo de clusterização.
     * - filepath: caminho da imagem no disco (usado na etapa de exportação/organização).
     */
    struct Point {
        std::vector<float> features;
        int cluster_id = -1;
        std::string filepath = "";
    };

    /**
     * @brief Estrutura de cluster usada por K-Means e PAM.
     *
     * - id: identificador do cluster (0..k-1)
     * - center: centro do cluster (centroide no K-Means, medoide no PAM)
     * - point_indices: índices dos pontos do dataset pertencentes ao cluster
     */
    struct Cluster {
        int id;
        std::vector<float> center;
        std::vector<size_t> point_indices;
    };

    /**
     * @brief Distância Euclidiana (L2) entre dois vetores de features.
     */
    float calcularDistanciaEuclidiana(const std::vector<float>& v1, const std::vector<float>& v2);

    /**
     * @brief Delta de Kronecker usada como penalidade tipo Potts (Eq. 14):
     *  - 0 se labels iguais
     *  - 1 se labels diferentes
     */
    inline int deltaKronecker(int label_i, int label_j) {
        return (label_i == label_j) ? 0 : 1;
    }

    /**
     * @brief WCSS (Within-Cluster Sum of Squares).
     *
     * Métrica padrão do K-Means: soma das distâncias ao quadrado de cada ponto ao centro do cluster.
     * Quanto menor, mais "compactos" ficam os clusters (no espaço de features).
     */
    float calcularWCSS(const std::vector<Point>& dataset, const std::vector<Cluster>& clusters);

    /**
     * @brief Silhouette score médio.
     *
     * Observação: é caro computacionalmente (O(N^2) no pior caso).
     * Use para análise, não necessariamente para benchmark de tempo.
     */
    float calcularSilhouette(const std::vector<Point>& dataset, const std::vector<Cluster>& clusters);

    /**
     * @brief Energia Total inspirada na Eq. 14: WCSS + lambda * penalidade topológica.
     *
     * Penalidade topológica:
     *  - considera a ordem linear do dataset (i com i+1),
     *  - soma 1 quando dois vizinhos consecutivos caem em clusters diferentes.
     *
     * lambda = 0 -> EnergiaTotal == WCSS.
     */
    float calcularEnergiaTotal(const std::vector<Point>& dataset, const std::vector<Cluster>& clusters, float lambda);

    /**
     * @brief Timer simples para benchmark de tempo (ms).
     */
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_time;
    public:
        void start() { start_time = std::chrono::high_resolution_clock::now(); }
        double elapsedMilliseconds() {
            auto end_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }
    };

} // namespace ClusterEngine

#endif // UTILS_HPP