#ifndef IMAGELOADER_HPP
#define IMAGELOADER_HPP

#include "Utils.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace ClusterEngine {

    /**
     * @brief Utilitários de carga de dados (Corel-1k e Acervo Real).
     *
     * Este módulo cuida apenas de:
     *  - Ler um arquivo TXT contendo caminhos de imagens (um por linha)
     *  - Ler cada imagem via OpenCV (imread)
     *
     * Observação: Qualquer lógica de MNIST (CSV vetorizado) foi removida para
     * ficar estritamente alinhado com o protocolo da Tabela 7 (Corel-1k + Acervo Real).
     */
    class ImageLoader {
    public:
        /**
         * @brief Carrega um dataset a partir de um arquivo de lista.
         *
         * Formato esperado:
         *  - Um caminho de imagem por linha (relativo ao diretório do projeto ou absoluto).
         * Exemplo:
         *  data/corel1k/africans/0.jpg
         *  data/corel1k/africans/1.jpg
         *  ...
         *
         * @param lista_path caminho do TXT com a lista de imagens.
         * @return vetor de Points, cada um com filepath preenchido (features ficam vazias nesta etapa).
         */
        static std::vector<Point> carregarListaImagens(const std::string& lista_path);

        /**
         * @brief Lê a imagem do disco usando OpenCV.
         *
         * Decisão: usar cv::imread com cv::IMREAD_COLOR, pois as features incluem histograma de cores.
         * @param filepath caminho do arquivo de imagem
         * @return cv::Mat (BGR). Se falhar, retorna Mat vazio.
         */
        static cv::Mat lerImagem(const std::string& filepath);
    };

} // namespace ClusterEngine

#endif // IMAGELOADER_HPP