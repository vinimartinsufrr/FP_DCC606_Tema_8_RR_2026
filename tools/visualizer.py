import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

def plotar_clusters(project_root: str, features_file: str, clusters_file: str, output_image: str, titulo: str):
    """
    Função genérica para carregar as características e mapeamentos de clusters, 
    aplicar PCA e salvar o gráfico correspondente.
    """
    features_path = os.path.join(project_root, "output", features_file)
    clusters_path = os.path.join(project_root, "output", clusters_file)
    out_png = os.path.join(project_root, "output", output_image)

    if not os.path.exists(features_path) or not os.path.exists(clusters_path):
        print(f"\n[Aviso] Arquivos para '{titulo}' não encontrados em output/. Etapa pulada.")
        return

    if os.path.getsize(features_path) == 0 or os.path.getsize(clusters_path) == 0:
        print(f"\n[Aviso] Arquivos de '{titulo}' estão vazios — nada para plotar.")
        return

    print(f"\n[Processando] {titulo}...")
    
    # Carregando Features
    try:
        features_df = pd.read_csv(features_path, header=None)
    except Exception as e:
        print(f"Falha ao ler {features_file}: {e}")
        return

    if features_df.shape[1] < 2:
        print(f"Estrutura incorreta em {features_file} (esperado filepath + características).")
        return

    features_df.rename(columns={0: "filepath"}, inplace=True)

    # Carregando Clusters
    try:
        clusters_df = pd.read_csv(clusters_path)
    except Exception as e:
        print(f"Falha ao ler {clusters_file}: {e}")
        return

    if "filepath" not in clusters_df.columns or "cluster_id" not in clusters_df.columns:
        print(f"Colunas ausentes em {clusters_file} (esperado: filepath, cluster_id)")
        return

    # Merge de sincronização
    merged_df = pd.merge(clusters_df, features_df, on="filepath", how="inner")
    if merged_df.empty:
        print(f"Merge vazio para {titulo}. Verifique a consistência dos filepaths.")
        return

    y = merged_df["cluster_id"].astype(int).values
    X = merged_df.drop(["filepath", "cluster_id"], axis=1).values

    if X.shape[0] < 2:
        print(f"Poucos pontos de dados para plotar {titulo}.")
        return

    # Redução de dimensionalidade via PCA
    print(f"  -> Aplicando PCA 2D (Reduzindo de {X.shape[1]} dimensões)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame(data=X_pca, columns=["PCA1", "PCA2"])
    df_pca["Cluster"] = y

    # Renderização Gráfica
    print("  -> Renderizando e salvando gráfico...")
    plt.figure(figsize=(10, 8))
    
    num_clusters = len(np.unique(y))
    palette_choice = "tab10" if num_clusters <= 10 else "hsv"

    sns.scatterplot(
        x="PCA1", y="PCA2",
        hue="Cluster",
        palette=sns.color_palette(palette_choice, num_clusters),
        data=df_pca,
        legend="full",
        alpha=0.7,
        edgecolor='w',
        linewidth=0.5
    )
    
    plt.title(titulo, fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close() # Libera memória do pyplot
    print(f"  [Sucesso] Gráfico salvo em: {out_png}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    print("=== EXECUTANDO MOTOR DE VISUALIZAÇÃO GRÁFICA ===")

    # 1. Plot do Controle Acadêmico (Corel-1k com K-Means++)
    plotar_clusters(
        project_root=project_root,
        features_file="corel_kmeans_features.csv",
        clusters_file="corel_kmeans_clusters.csv",
        output_image="corel_kmeans_plot.png",
        titulo="Visualização Corel-1k — K-Means++ Nativo (K=10)"
    )

    # 2. Plot da Aplicação Prática (Acervo Real com K-Means++)
    plotar_clusters(
        project_root=project_root,
        features_file="acervo_features.csv",
        clusters_file="acervo_clusters.csv",
        output_image="acervo_plot.png",
        titulo="Visualização Acervo Real — K-Means++ Nativo (K=3)"
    )

if __name__ == "__main__":
    main()