import os
import shutil
import pandas as pd

def _copiar_sem_sobrescrever(src: str, dst_dir: str, filename: str) -> bool:
    """
    Copia src para dst_dir/filename. Se já existir, cria um novo nome com sufixo _1, _2, ...
    Retorna True se copiou com sucesso, False caso contrário.
    """
    base, ext = os.path.splitext(filename)
    dst_path = os.path.join(dst_dir, filename)

    if not os.path.exists(dst_path):
        shutil.copy2(src, dst_path)
        return True

    for i in range(1, 10000):
        alt_name = f"{base}_{i}{ext}"
        alt_path = os.path.join(dst_dir, alt_name)
        if not os.path.exists(alt_path):
            shutil.copy2(src, alt_path)
            return True

    return False

def organizar_dataset(project_root: str, csv_filename: str, output_folder_name: str, apelido_dataset: str):
    """
    Função genérica que lê um CSV de clusters e organiza os arquivos em uma pasta específica.
    """
    csv_path = os.path.join(project_root, "output", csv_filename)
    if not os.path.exists(csv_path):
        print(f"\n[Aviso] Arquivo {csv_filename} não encontrado. Etapa do {apelido_dataset} ignorada.")
        return

    if os.path.getsize(csv_path) == 0:
        print(f"\n[Aviso] {csv_filename} está vazio — nada para organizar no {apelido_dataset}.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Falha ao ler {csv_filename}: {e}")
        return

    if df.empty:
        print(f"{csv_filename} não contém linhas — nada para organizar.")
        return

    if "filepath" not in df.columns or "cluster_id" not in df.columns:
        print(f"{csv_filename} não contém as colunas esperadas: filepath, cluster_id")
        return

    output_base = os.path.join(project_root, "output", output_folder_name)
    os.makedirs(output_base, exist_ok=True)

    print(f"\nIniciando organização de {len(df)} imagens do {apelido_dataset}...")

    sucesso = 0
    falhas = 0
    nao_encontradas = 0

    for _, row in df.iterrows():
        caminho_original = str(row["filepath"])
        filepath_real = os.path.join(project_root, caminho_original)

        try:
            cluster_id = int(row["cluster_id"])
        except Exception:
            print(f"[Aviso] cluster_id inválido para {caminho_original}: {row['cluster_id']}")
            falhas += 1
            continue

        if not os.path.exists(filepath_real):
            nao_encontradas += 1
            continue

        cluster_dir = os.path.join(output_base, f"Cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        try:
            filename = os.path.basename(caminho_original)
            ok = _copiar_sem_sobrescrever(filepath_real, cluster_dir, filename)
            if ok:
                sucesso += 1
            else:
                falhas += 1
        except Exception as e:
            print(f"Erro ao copiar {filepath_real}: {e}")
            falhas += 1

    print(
        f"Finalizado ({apelido_dataset})!\n"
        f"  - Copiados com sucesso: {sucesso}\n"
        f"  - Falhas: {falhas}\n"
        f"  - Não encontradas no disco: {nao_encontradas}\n"
        f"  - Destino: {output_base}"
    )

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    print("=== EXECUTANDO ORGANIZADOR DE DIRETÓRIOS ===")

    # 1. Organiza o Controle Acadêmico (Corel-1k K-Means++)
    organizar_dataset(
        project_root=project_root,
        csv_filename="corel_kmeans_clusters.csv",
        output_folder_name="corel_organizado",
        apelido_dataset="Corel-1k (K-Means++)"
    )

    # 2. Organiza a Aplicação Prática (Acervo Real K-Means++)
    organizar_dataset(
        project_root=project_root,
        csv_filename="acervo_clusters.csv",
        output_folder_name="acervo_organizado",
        apelido_dataset="Acervo Real (K-Means++)"
    )

if __name__ == "__main__":
    main()