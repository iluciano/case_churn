from pathlib import Path
import zipfile

DATASET = "gcenachi/case-data-master-2024"
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print(f"Baixando {DATASET} para {OUT_DIR} ...")
    api.dataset_download_files(DATASET, path=str(OUT_DIR), quiet=False)

    zips = list(OUT_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("Nenhum .zip encontrado. Verifique credenciais Kaggle.")
    zip_path = zips[0]
    print("Descompactando:", zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(OUT_DIR)

    print("OK. Arquivos em:", OUT_DIR.resolve())
    print("Conteúdo:", [p.name for p in OUT_DIR.iterdir()])

if __name__ == "__main__":
    main()