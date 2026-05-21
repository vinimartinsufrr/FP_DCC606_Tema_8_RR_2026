#include "../include/ImageLoader.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace ClusterEngine {

    std::vector<Point> ImageLoader::carregarListaImagens(const std::string& lista_path) {
        std::vector<Point> dataset;
        std::ifstream arquivo(lista_path);

        if (!arquivo.is_open()) {
            throw std::runtime_error("Erro ao abrir a lista de imagens: " + lista_path);
        }

        std::string caminho;
        while (std::getline(arquivo, caminho)) {
            if (caminho.empty()) continue;

            // Cada linha vira uma instância (Point). Features serão preenchidas depois pelo FeatureExtractor.
            Point ponto;
            ponto.filepath = caminho;
            dataset.push_back(ponto);
        }

        return dataset;
    }

    cv::Mat ImageLoader::lerImagem(const std::string& filepath) {
        // Lemos em modo COLOR pois o descritor de cores (histograma) depende de informação RGB.
        cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);

        // Aviso não-fatal: algumas imagens podem falhar (arquivo corrompido, caminho inválido etc.)
        if (img.empty()) {
            std::cerr << "[Aviso] OpenCV falhou ao carregar: " << filepath << "\n";
        }

        return img;
    }

} // namespace ClusterEngine