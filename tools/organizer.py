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

    # Se já existe, tenta nomes alternativos
    for i in range(1, 10000):
        alt_name = f"{base}_{i}{ext}"
        alt_path = os.path.join(dst_dir, alt_name)
        if not os.path.exists(alt_path):
            shutil.copy2(src, alt_path)
            return True

    return False

def main():
    """
    Organiza as imagens em diretórios por cluster.

    Entrada:
      - ../output/clusters_results.csv (gerado pelo C++)
        colunas: filepath, cluster_id

    Saída:
      - ../output/acervo_organizado/Cluster_<id>/*.jpg|png...

    Observação:
      - Este script NÃO faz clusterização. Ele só aplica no filesystem o "mapa" (CSV)
        produzido pelo C++.
      - Usa cópia (shutil.copy2) para preservar metadados e não destruir o dataset original.
    """
    # Resolve caminhos de forma robusta (independente do diretório onde o usuário roda o script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    csv_path = os.path.join(project_root, "output", "clusters_results.csv")
    if not os.path.exists(csv_path):
        print(f"Arquivo {csv_path} nao encontrado. Rode o C++ primeiro!")
        return

    # Evita pandas.errors.EmptyDataError quando o arquivo existe mas está vazio
    if os.path.getsize(csv_path) == 0:
        print("clusters_results.csv está vazio — nada para organizar.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Falha ao ler clusters_results.csv: {e}")
        return

    if df.empty:
        print("clusters_results.csv não contém linhas — nada para organizar.")
        return

    if "filepath" not in df.columns or "cluster_id" not in df.columns:
        print("clusters_results.csv não contém colunas esperadas: filepath, cluster_id")
        return

    output_base = os.path.join(project_root, "output", "acervo_organizado")
    os.makedirs(output_base, exist_ok=True)

    print(f"Iniciando organizacao de {len(df)} imagens...")

    sucesso = 0
    falhas = 0
    nao_encontradas = 0

    for _, row in df.iterrows():
        caminho_original = str(row["filepath"])

        # O CSV deve conter caminhos relativos ao projeto (ex: data/corel1k/...jpg)
        filepath_real = os.path.join(project_root, caminho_original)

        # Garante que cluster_id vire inteiro (evita pasta "Cluster_8.0")
        try:
            cluster_id = int(row["cluster_id"])
        except Exception:
            print(f"[Aviso] cluster_id invalido para {caminho_original}: {row['cluster_id']}")
            falhas += 1
            continue

        if not os.path.exists(filepath_real):
            print(f"[Aviso] Arquivo nao encontrado no disco: {filepath_real}")
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
                print(f"[Aviso] Nao foi possivel gerar nome unico para: {filepath_real}")
                falhas += 1
        except Exception as e:
            print(f"Erro ao copiar {filepath_real}: {e}")
            falhas += 1

    print(
        f"\nFinalizado!\n"
        f"- Copiados com sucesso: {sucesso}\n"
        f"- Falhas: {falhas}\n"
        f"- Nao encontradas no disco: {nao_encontradas}\n"
        f"Destino: {output_base}"
    )

if __name__ == "__main__":
    main()