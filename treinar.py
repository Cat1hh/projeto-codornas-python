#!/usr/bin/env python3
"""
Treina o YOLO26 com o seu dataset de codornas.

Estrutura esperada do dataset (formato YOLO, exportável do Roboflow/CVAT):
    dataset/
      images/train/*.jpg      labels/train/*.txt
      images/val/*.jpg        labels/val/*.txt

O detector também pode gerar amostras pseudo-rotuladas em
`dataset/images/train` e `dataset/labels/train`. Revise essas imagens e
corrija os arquivos `.txt` antes de treinar; elas não substituem exemplos
rotulados manualmente nem devem ser usadas cegamente para validar o modelo.

Exemplo:
    python treinar.py --data codornas.yaml --model yolo26n.pt --epochs 100

Ao final, o melhor modelo fica em runs/codorna/weights/best.pt.
Copie-o para esta pasta como best.pt para o detectar_codornas.py usar automaticamente.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preparar_validacao(data_dir, proporcao=0.2):
    """Cria uma validação inicial separando imagens já rotuladas do treino."""
    train_images = data_dir / "images" / "train"
    train_labels = data_dir / "labels" / "train"
    val_images = data_dir / "images" / "val"
    val_labels = data_dir / "labels" / "val"
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    if any(val_images.iterdir()):
        return

    pares = sorted(
        imagem for imagem in train_images.iterdir()
        if imagem.is_file()
        and imagem.suffix.lower() in EXTENSOES_IMAGEM
        and (train_labels / f"{imagem.stem}.txt").is_file()
    )
    if len(pares) < 2:
        raise RuntimeError(
            f"Validação vazia: são necessárias pelo menos 2 imagens rotuladas em {train_images} "
            "para separar treino e validação."
        )

    quantidade = max(1, round(len(pares) * proporcao))
    for imagem in pares[-quantidade:]:
        rotulo = train_labels / f"{imagem.stem}.txt"
        shutil.move(str(imagem), str(val_images / imagem.name))
        shutil.move(str(rotulo), str(val_labels / rotulo.name))

    print(f"[info] Validação criada: {quantidade} imagens em {val_images}")


def main():
    p = argparse.ArgumentParser(description="Treino YOLO26 para codornas")
    p.add_argument("--data", default="codornas.yaml", help="Arquivo YAML do dataset")
    p.add_argument("--model", default="yolo26n.pt",
                   help="Ponto de partida: yolo26n.pt (leve) ou yolo26s.pt (mais preciso)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640,
                   help="Aumente (ex.: 960) se as aves forem pequenas na imagem")
    p.add_argument("--batch", type=int, default=16, help="Reduza se faltar memória")
    p.add_argument("--device", default=None, help="cpu, 0 (GPU)... (padrão: automático)")
    p.add_argument("--patience", type=int, default=30, help="Para cedo se não melhorar")
    args = p.parse_args()

    data_dir = Path(args.data).resolve().parent / "dataset"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Pasta do dataset não encontrada: {data_dir}")
    preparar_validacao(data_dir)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project="runs",
        name="codorna",
        exist_ok=True,
    )

    metricas = model.val()
    print(f"[info] mAP50: {metricas.box.map50:.3f} | mAP50-95: {metricas.box.map:.3f}")

    melhor = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"[info] Melhor modelo: {melhor}")
    print("[info] Copie para a pasta do projeto como best.pt para usar na detecção.")


if __name__ == "__main__":
    main()