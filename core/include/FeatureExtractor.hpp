#ifndef FEATUREEXTRACTOR_HPP
#define FEATUREEXTRACTOR_HPP

#include "Utils.hpp"
#include <vector>
#include <opencv2/opencv.hpp>

namespace ClusterEngine {

    /**
     * @brief Extrai um vetor de características (features) de imagens para uso em clusterização.
     *
     * Pipeline de features implementado:
     *  - Histograma de Cores (RGB quantizado): captura distribuição de cores global.
     *  - HOG simplificado (Histograma de Gradientes Orientados): captura informação de bordas/formas.
     *
     * Observação importante:
     *  - O algoritmo de clusterização (K-Means/PAM) só "enxerga" números; este módulo transforma
     *    a imagem (pixels) em um vetor numérico de dimensão fixa.
     */
    class FeatureExtractor {
    private:
        // Hiperparâmetro: número de bins por canal de cor (R, G, B).
        // Ex.: 4 bins -> 4*4*4 = 64 bins no histograma total.
        int num_bins_cor;

        // Hiperparâmetro: quantização angular do HOG.
        // Ex.: 9 bins -> orientação em 0..180° dividida em 9 fatias.
        int bins_orientacao_hog;

        /**
         * @brief Extrai histograma de cores RGB quantizado.
         *
         * Escolha típica para manter baixo custo computacional e ainda capturar informação de cor:
         *  - bins_canal_cor = 4 (padrão) -> 64 dimensões.
         *
         * Valores maiores (ex.: 8 -> 512 dimensões) aumentam custo e podem piorar por "maldição da dimensionalidade".
         */
        std::vector<float> extrairHistogramaCores(const cv::Mat& imagem);

        /**
         * @brief Extrai um descritor HOG simplificado com orientação.
         *
         * Implementação:
         *  - Converte para grayscale
         *  - Divide a imagem em uma grade fixa (atualmente 4x4 células)
         *  - Em cada célula, acumula um histograma de orientações (bins_orientacao_hog)
         *  - Faz normalização L2 global do descritor
         *
         * Isso não é o HOG completo (Dalal-Triggs com blocos sobrepostos e interpolação),
         * mas é suficiente para o objetivo do trabalho (clusterização + benchmark).
         */
        std::vector<float> extrairGradientesOrientados(const cv::Mat& imagem);

    public:
        /**
         * @param bins_canal_cor bins por canal no histograma de cor (padrão 4)
         * @param bins_hog bins de orientação do HOG (padrão 9)
         */
        FeatureExtractor(int bins_canal_cor = 4, int bins_hog = 9);

        /**
         * @brief Percorre o dataset e preenche Point::features para cada imagem.
         *
         * Pré-condição: Point::filepath deve apontar para uma imagem existente.
         * Pós-condição: Point::features terá dimensão (bins_cor^3 + grade^2 * bins_hog).
         */
        void processarDatasetImagens(std::vector<Point>& dataset);
    };

} // namespace ClusterEngine

#endif // FEATUREEXTRACTOR_HPP