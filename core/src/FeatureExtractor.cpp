#include "../include/FeatureExtractor.hpp"
#include "../include/ImageLoader.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>

namespace ClusterEngine {

    FeatureExtractor::FeatureExtractor(int bins_canal_cor, int bins_hog)
        : num_bins_cor(bins_canal_cor), bins_orientacao_hog(bins_hog) {}

    std::vector<float> FeatureExtractor::extrairHistogramaCores(const cv::Mat& imagem) {
        // total_bins = (bins_R * bins_G * bins_B) = bins^3
        // Ex.: 4^3 = 64 dimensões
        const int total_bins = num_bins_cor * num_bins_cor * num_bins_cor;
        std::vector<float> histograma(total_bins, 0.0f);

        // OpenCV carrega em BGR por padrão; convertemos para RGB só para ficar semanticamente correto.
        cv::Mat img_rgb;
        cv::cvtColor(imagem, img_rgb, cv::COLOR_BGR2RGB);

        // Cada bin cobre um intervalo uniforme de intensidades [0, 255].
        // Ex.: bins=4 -> cada bin ~64 níveis.
        const float normalizador_bin = 256.0f / num_bins_cor;

        for (int y = 0; y < img_rgb.rows; ++y) {
            for (int x = 0; x < img_rgb.cols; ++x) {
                const cv::Vec3b pixel = img_rgb.at<cv::Vec3b>(y, x);

                // Quantização por canal
                const int bin_r = std::clamp(static_cast<int>(pixel[0] / normalizador_bin), 0, num_bins_cor - 1);
                const int bin_g = std::clamp(static_cast<int>(pixel[1] / normalizador_bin), 0, num_bins_cor - 1);
                const int bin_b = std::clamp(static_cast<int>(pixel[2] / normalizador_bin), 0, num_bins_cor - 1);

                // Indexação 3D -> 1D
                const int idx = bin_r * (num_bins_cor * num_bins_cor) + bin_g * num_bins_cor + bin_b;
                histograma[idx] += 1.0f;
            }
        }

        // Normalização para distribuição (soma ~ 1.0)
        const size_t total_pixels = static_cast<size_t>(img_rgb.rows) * static_cast<size_t>(img_rgb.cols);
        if (total_pixels > 0) {
            for (int i = 0; i < total_bins; ++i) {
                histograma[i] /= static_cast<float>(total_pixels);
            }
        }

        return histograma;
    }

    std::vector<float> FeatureExtractor::extrairGradientesOrientados(const cv::Mat& imagem) {
        // Hiperparâmetro: grade de células (4x4).
        // Motivo: mantém o descritor compacto e rápido, e combina bem com resize 64x64 (cada célula ~16x16 px).
        const int grade = 4;

        // Dimensão do descritor HOG simplificado:
        // grade*grade*canais_de_orientacao => 4*4*9 = 144 dimensões (com bins_hog=9)
        std::vector<float> descritor(grade * grade * bins_orientacao_hog, 0.0f);

        cv::Mat cinza;
        cv::cvtColor(imagem, cinza, cv::COLOR_BGR2GRAY);

        // Tamanho de cada célula
        const int bloco_w = cinza.cols / grade;
        const int bloco_h = cinza.rows / grade;

        // Orientações "sem sinal": 0..180 graus
        const float angulo_por_bin = 180.0f / bins_orientacao_hog;

        for (int bg = 0; bg < grade; ++bg) {
            for (int lg = 0; lg < grade; ++lg) {
                const int start_x = lg * bloco_w;
                const int start_y = bg * bloco_h;

                // Evita bordas para não acessar fora.
                // Observação: se a célula for pequena demais, o loop pode não rodar.
                for (int y = start_y + 1; y < start_y + bloco_h - 1; ++y) {
                    for (int x = start_x + 1; x < start_x + bloco_w - 1; ++x) {
                        const float dx = static_cast<float>(cinza.at<uchar>(y, x + 1) - cinza.at<uchar>(y, x - 1));
                        const float dy = static_cast<float>(cinza.at<uchar>(y + 1, x) - cinza.at<uchar>(y - 1, x));

                        const float magnitude = std::sqrt(dx * dx + dy * dy);

                        // atan2 retorna em [-pi, pi]. Convertendo para graus e ajustando para [0, 180).
                        float angulo = std::atan2(dy, dx) * (180.0f / static_cast<float>(M_PI));
                        if (angulo < 0.0f) angulo += 180.0f;

                        const int bin = std::clamp(static_cast<int>(angulo / angulo_por_bin), 0, bins_orientacao_hog - 1);
                        const int idx_base = (bg * grade + lg) * bins_orientacao_hog;
                        descritor[idx_base + bin] += magnitude;
                    }
                }
            }
        }

        // Normalização L2 global:
        // reduz sensibilidade à iluminação/contraste e evita que imagens maiores "dominem" por magnitude.
        float soma_quadrados = 0.0f;
        for (float v : descritor) soma_quadrados += v * v;

        if (soma_quadrados > 0.0f) {
            const float norma = std::sqrt(soma_quadrados);
            for (float& v : descritor) v /= norma;
        }

        return descritor;
    }

    void FeatureExtractor::processarDatasetImagens(std::vector<Point>& dataset) {
        std::cout << "[Processamento] Extraindo features (Histograma + HOG)...\n";

        size_t falhas = 0;
        for (size_t i = 0; i < dataset.size(); ++i) {
            cv::Mat img = ImageLoader::lerImagem(dataset[i].filepath);
            if (img.empty()) {
                falhas++;
                continue;
            }

            // Escolha de engenharia:
            // padronizamos o tamanho para que o HOG (grade fixa) produza vetores comparáveis
            // e para estabilizar tempo de execução.
            // 64x64 é um tamanho pequeno (rápido) e ainda preserva bordas principais.
            cv::resize(img, img, cv::Size(64, 64));

            const std::vector<float> hist = extrairHistogramaCores(img);
            const std::vector<float> hog  = extrairGradientesOrientados(img);

            dataset[i].features.clear();
            dataset[i].features.reserve(hist.size() + hog.size());
            dataset[i].features.insert(dataset[i].features.end(), hist.begin(), hist.end());
            dataset[i].features.insert(dataset[i].features.end(), hog.begin(), hog.end());
        }

        std::cout << "[Sucesso] Processadas: " << (dataset.size() - falhas) << " | Falhas: " << falhas << "\n";
    }

} // namespace ClusterEngine