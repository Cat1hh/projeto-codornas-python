#!/usr/bin/env python3
"""
Captura frames da câmera IP em intervalos regulares para montar o dataset.

Exemplo:
    python capturar_frames.py --source "rtsp://usuario:senha@192.168.0.50:554/stream1" \
        --every 10 --max 500

Dica: capture em horários diferentes (manhã, tarde, noite com luz artificial)
e em gaiolas/ângulos diferentes. Depois rotule as imagens no Roboflow, CVAT ou Label Studio.
"""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Captura de frames para o dataset")
    p.add_argument("--source", default=os.getenv("CAMERA_URL", "0"), help="URL RTSP ou índice da webcam")
    p.add_argument("--out", default="dataset/images/raw", help="Pasta de saída")
    p.add_argument("--every", type=float, default=10, help="Segundos entre capturas")
    p.add_argument("--max", type=int, default=500, help="Quantidade máxima de imagens")
    args = p.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    salvos = 0
    ultimo = 0.0
    print(f"[info] Salvando em {out}/ a cada {args.every}s. Ctrl+C para parar.")

    try:
        while salvos < args.max:
            ok, frame = cap.read()
            if not ok:
                print("[aviso] Falha na leitura. Reconectando...")
                cap.release()
                time.sleep(3)
                cap = cv2.VideoCapture(source)
                continue

            agora = time.time()
            if agora - ultimo >= args.every:
                nome = out / f"codorna_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(nome), frame)
                salvos += 1
                ultimo = agora
                print(f"[{salvos}/{args.max}] {nome.name}")
    except KeyboardInterrupt:
        print("\n[info] Interrompido.")
    finally:
        cap.release()
        print(f"[info] {salvos} imagens salvas em {out}/")


if __name__ == "__main__":
    main()