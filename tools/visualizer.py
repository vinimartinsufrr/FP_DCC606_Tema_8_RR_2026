import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

def main():
    """
    Gera uma visualização 2D (PCA) dos vetores de features.

    Entrada:
      - ../output/features.csv  (gerado pelo C++)
        formato: filepath, f1, f2, ... fN
      - ../output/clusters_results.csv (gerado pelo C++)
        formato: filepath, cluster_id

    Saída:
      - ../output/cluster_plot.png

    Observação:
      - PCA reduz N dimensões para 2D, então sobreposição é esperada.
      - O objetivo é visual qualitativo (não é prova formal de separação).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    features_path = os.path.join(project_root, "output", "features.csv")
    clusters_path = os.path.join(project_root, "output", "clusters_results.csv")
    out_png = os.path.join(project_root, "output", "cluster_plot.png")

    if not os.path.exists(features_path) or not os.path.exists(clusters_path):
        print("Arquivos de entrada não encontrados. Rode o C++ primeiro para gerar os CSVs em output/.")
        return

    # Evita pandas.errors.EmptyDataError quando o arquivo existe mas está vazio
    if os.path.getsize(features_path) == 0:
        print("features.csv está vazio — nada para plotar (dataset pode estar vazio ou export falhou).")
        return
    if os.path.getsize(clusters_path) == 0:
        print("clusters_results.csv está vazio — nada para plotar (dataset pode estar vazio ou export falhou).")
        return

    print("Carregando features...")
    try:
        features_df = pd.read_csv(features_path, header=None)
    except Exception as e:
        print(f"Falha ao ler features.csv: {e}")
        return

    if features_df.shape[1] < 2:
        # Esperado no mínimo: coluna 0 (filepath) + 1 feature
        print("features.csv não tem colunas suficientes (esperado: filepath + features).")
        return

    features_df.rename(columns={0: "filepath"}, inplace=True)

    print("Carregando clusters...")
    try:
        clusters_df = pd.read_csv(clusters_path)
    except Exception as e:
        print(f"Falha ao ler clusters_results.csv: {e}")
        return

    if "filepath" not in clusters_df.columns or "cluster_id" not in clusters_df.columns:
        print("clusters_results.csv não contém colunas esperadas: filepath, cluster_id")
        return

    print("Sincronizando dados...")
    merged_df = pd.merge(clusters_df, features_df, on="filepath", how="inner")

    if merged_df.empty:
        print("Merge resultou em 0 linhas. Verifique se os filepaths em features.csv e clusters_results.csv são iguais.")
        return

    y = merged_df["cluster_id"].astype(int).values
    X = merged_df.drop(["filepath", "cluster_id"], axis=1).values

    if X.shape[0] < 2:
        print("Poucos pontos para PCA/plot (menos de 2 instâncias).")
        return

    print(f"Reduzindo dimensionalidade de {X.shape[1]} para 2D usando PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame(data=X_pca, columns=["PCA1", "PCA2"])
    df_pca["Cluster"] = y

    print("Plotando grafico...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="PCA1", y="PCA2",
        hue="Cluster",
        palette=sns.color_palette("hsv", len(np.unique(y))),
        data=df_pca,
        legend="full",
        alpha=0.7
    )
    plt.title("Visualização dos Clusters de Imagens (PCA 2D)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Grafico salvo em: {out_png}")

if __name__ == "__main__":
    main()