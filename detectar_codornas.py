#!/usr/bin/env python3
"""
Detecção, rastreamento e contagem de codornas em câmera IP com YOLO26.
 
Uso básico (as opções ficam no arquivo config.YAML):
    python detectar_codornas.py
 
Qualquer opção pode ser sobrescrita na linha de comando (ela vence o config.YAML):
    python detectar_codornas.py --source "http://192.168.1.13:8080/video" --conf 0.3
 
O que este programa faz:
  - Lê a câmera em uma thread própria e processa sempre o quadro MAIS RECENTE.
  - Reduz imagens grandes antes de detectar (--max-width).
  - Mostra uma contagem ESTÁVEL (mediana dos últimos quadros), sem ficar pulando.
  - Desenha caixas com cor por ave, um painel limpo e um mini-gráfico da contagem.
  - Avisa na tela quando a câmera perde o sinal.
  - Registra a contagem em CSV e salva foto quando a contagem cai abaixo do mínimo.
 
Teclas (com janela aberta): q ou ESC = sair.
"""
import argparse
import csv
import os
import re
import statistics
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
 
# RTSP via TCP é mais estável que UDP. Deve ser definido ANTES de abrir a câmera.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
 
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from registro import Registro  # noqa: E402

COCO_BIRD_CLASS = 14  # classe "bird" no COCO (só usada com modelo pré-treinado)
NOME_JANELA = "Codornas - YOLO26"
 
# Cores (BGR)
BRANCO = (255, 255, 255)
CINZA = (190, 190, 190)
VERDE = (90, 200, 60)
AMARELO = (0, 200, 255)
VERMELHO = (60, 60, 235)
CIANO = (230, 200, 60)
# Paleta amigável a daltônicos (Okabe-Ito), em RGB; convertida para BGR ao usar
PALETA_RGB = [(86, 180, 233), (230, 159, 0), (0, 158, 115), (240, 228, 66),
              (0, 114, 178), (213, 94, 0), (204, 121, 167), (170, 170, 170)]
FONTE = cv2.FONT_HERSHEY_DUPLEX
 
 
# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
def criar_parser():
    p = argparse.ArgumentParser(description="Contagem de codornas com YOLO26 + câmera IP")
    p.add_argument("--config", default="config.YAML", help="Arquivo de configuração (YAML)")
    p.add_argument("--source", default=os.getenv("CAMERA_URL", "0"),
                   help="URL da câmera (RTSP/HTTP), arquivo de vídeo ou índice da webcam")
    p.add_argument("--model", default="best.pt",
                   help="Pesos do modelo (padrão: best.pt; se não existir, usa yolo26n.pt)")
    p.add_argument("--conf", type=float, default=0.4, help="Confiança mínima (0 a 1)")
    p.add_argument("--imgsz", type=int, default=640, help="Tamanho de entrada do modelo")
    p.add_argument("--max-width", type=int, default=960,
                   help="Reduz quadros mais largos que isso antes de processar (0 = não reduzir)")
    p.add_argument("--device", default=None, help="cpu, 0 (GPU NVIDIA), mps... (padrão: automático)")
    p.add_argument("--threads", type=int, default=0,
                   help="Threads de CPU para o PyTorch (0 = padrão). Ex.: 4")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="Filtrar classes (ex.: --classes 14 para 'bird' no COCO)")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="Configuração do rastreador (ex.: tracker_codornas.yaml)")
    p.add_argument("--smooth", type=int, default=15,
                   help="Quantos quadros usar para a contagem estável (mediana)")
    p.add_argument("--stride", type=int, default=1,
                   help="Processa 1 a cada N quadros novos (1 = todos os que der tempo)")
    p.add_argument("--max-fps", type=float, default=0,
                   help="Limita a taxa de processamento para poupar CPU (0 = sem limite)")
    p.add_argument("--no-show", action="store_true", help="Não abrir janela (modo servidor)")
    p.add_argument("--save-video", default=None, help="Caminho para gravar o vídeo anotado (.mp4)")
    p.add_argument("--save-fps", type=float, default=10, help="FPS do vídeo gravado")
    p.add_argument("--log", default="contagem.csv", help="Arquivo CSV com o histórico de contagem")
    p.add_argument("--log-interval", type=int, default=60, help="Segundos entre registros no CSV")
    p.add_argument("--min-count", type=int, default=0,
                   help="Alerta se a contagem ficar abaixo deste valor (0 = desligado)")
    p.add_argument("--alert-seconds", type=int, default=30,
                   help="Tempo abaixo do mínimo antes de disparar o alerta")
    p.add_argument("--alert-dir", default="alertas", help="Pasta onde salvar fotos dos alertas")
    p.add_argument("--db", default="dados.db", help="Banco SQLite lido pelo dashboard")
    p.add_argument("--db-interval", type=int, default=10, help="Segundos entre amostras no banco")
    p.add_argument("--captures-dir", default="capturas", help="Pasta das fotos periódicas")
    p.add_argument("--capture-interval", type=int, default=300, help="Segundos entre fotos (0 = desligado)")
    p.add_argument("--train-dir", default="dataset/images/raw", help="Pasta das fotos sem caixas, para treino manual")
    p.add_argument("--train-interval", type=int, default=0, help="Segundos entre fotos de treino (0 = desligado)")
    p.add_argument("--auto-dataset", default="dataset",
                   help="Dataset YOLO onde salvar imagens e rótulos das detecções automáticas")
    p.add_argument("--auto-interval", type=int, default=30,
                   help="Segundos entre amostras detectadas com rótulo (0 = desligado)")
    p.add_argument("--keep-days", type=int, default=30, help="Dias de histórico (0 = guardar tudo)")
    return p
 
 
def carregar_config(caminho):
    """Lê o arquivo YAML (se existir). Retorna um dicionário vazio se não houver."""
    if not Path(caminho).is_file():
        return {}
    try:
        import yaml
        with open(caminho, "r", encoding="utf-8") as f:
            dados = yaml.safe_load(f) or {}
        if not isinstance(dados, dict):
            print(f"[aviso] {caminho} não tem o formato esperado (chave: valor). Ignorado.")
            return {}
        return dados
    except Exception as e:  # arquivo malformado ou PyYAML ausente
        print(f"[aviso] Não consegui ler {caminho}: {e}")
        return {}
 
 
def parse_args():
    p = criar_parser()
    pre, _ = p.parse_known_args()
    cfg = carregar_config(pre.config)
    validos = {a.dest for a in p._actions}
    desconhecidos = [k for k in cfg if k not in validos]
    if desconhecidos:
        print(f"[aviso] Opções desconhecidas em {pre.config}: {', '.join(map(str, desconhecidos))}")
    p.set_defaults(**{k: v for k, v in cfg.items() if k in validos})
    return p.parse_args()
 
 
def mascarar_url(url) -> str:
    """Esconde usuário e senha ao imprimir a URL no terminal."""
    return re.sub(r"//[^@/]+@", "//***:***@", str(url))
 
 
# ----------------------------------------------------------------------------
# Leitura da câmera
# ----------------------------------------------------------------------------
class LeitorCamera:
    """Lê a câmera numa thread própria e guarda apenas o quadro mais recente.
 
    Assim a detecção nunca trabalha em cima de quadros velhos acumulados no
    buffer, e a leitura da rede não fica esperando o modelo terminar.
    """
 
    def __init__(self, source):
        self.source = source
        self.arquivo = isinstance(source, str) and Path(source).is_file()
        self.fps = 0.0                          # fps que a câmera está entregando
        self.fim = False                        # True quando um vídeo de teste termina
        self.ultimo_em = time.perf_counter()    # instante do último quadro recebido
        self._frame = None
        self._id = 0
        self._lock = threading.Lock()
        self._parar = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
 
    def iniciar(self):
        self._thread.start()
        return self
 
    def _abrir(self):
        if isinstance(self.source, str):
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
 
    def _loop(self):
        cap = self._abrir()
        falhas = 0
        t_ant = time.perf_counter()
        intervalo_arquivo = 0.0
        if self.arquivo:
            fps_arq = cap.get(cv2.CAP_PROP_FPS)
            intervalo_arquivo = 1.0 / fps_arq if 1 <= fps_arq <= 240 else 1.0 / 30
 
        while not self._parar.is_set():
            ok, frame = cap.read()
 
            if not ok:
                if self.arquivo:        # fim do vídeo de teste
                    self.fim = True
                    break
                # Reconexão automática se a câmera cair ou o stream travar
                falhas += 1
                espera = min(2 * falhas, 10)
                print(f"[aviso] Falha ao ler o stream (tentativa {falhas}). Reconectando em {espera}s...")
                cap.release()
                if self._parar.wait(espera):
                    return
                cap = self._abrir()
                continue
 
            falhas = 0
            agora = time.perf_counter()
            dt = agora - t_ant
            t_ant = agora
            self.ultimo_em = agora
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
 
            with self._lock:
                self._frame = frame
                self._id += 1
 
            if self.arquivo:            # simula a velocidade real do vídeo gravado
                time.sleep(intervalo_arquivo)
 
        cap.release()
 
    def ler(self):
        """Retorna (quadro mais recente, número do quadro)."""
        with self._lock:
            return self._frame, self._id
 
    def parar(self):
        self._parar.set()
 
 
# ----------------------------------------------------------------------------
# Desenho (caixas, painel, gráfico)
# ----------------------------------------------------------------------------
def cor_id(i):
    r, g, b = PALETA_RGB[i % len(PALETA_RGB)]
    return (b, g, r)
 
 
def texto(img, s, pos, escala, cor=BRANCO, esp=1, alinhar="esq"):
    """Texto com contorno escuro (legível sobre qualquer fundo). Retorna a largura."""
    (tw, _), _ = cv2.getTextSize(s, FONTE, escala, esp)
    x, y = pos
    if alinhar == "dir":
        x -= tw
    elif alinhar == "centro":
        x -= tw // 2
    cv2.putText(img, s, (x, y), FONTE, escala, (0, 0, 0), esp + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONTE, escala, cor, esp, cv2.LINE_AA)
    return tw
 
 
def desenhar_caixas(img, r):
    """Caixas finas, uma cor por ave (ID do rastreador) e rótulo discreto."""
    n = len(r.boxes)
    if n == 0:
        return
    xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
    ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * n
    for (x1, y1, x2, y2), i in zip(xyxy, ids):
        cor = cor_id(i) if i is not None else (200, 200, 200)
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2, cv2.LINE_AA)
        if i is not None:
            rotulo = f"#{i}"
            (tw, th), _ = cv2.getTextSize(rotulo, FONTE, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), cor, -1)
            cv2.putText(img, rotulo, (x1 + 4, y1 - 4), FONTE, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
 
 
def escurecer(regiao, fator):
    """Escurece uma região da imagem (fundo translúcido do painel), de forma barata."""
    regiao[:] = cv2.convertScaleAbs(regiao, alpha=fator)
 
 
def desenhar_hud(img, e):
    """Painel superior com a contagem, estado, hora e desempenho + mini-gráfico."""
    h, w = img.shape[:2]
    s = min(max(w / 960, 0.6), 1.6)         # escala conforme a largura da imagem
    barra = int(64 * s)
 
    escurecer(img[:barra], 0.3)
 
    # Indicador de estado: verde = ok, amarelo = abaixo do mínimo, vermelho = alerta
    cv2.circle(img, (int(26 * s), barra // 2), int(9 * s), e["cor_estado"], -1, cv2.LINE_AA)
 
    # Contagem estável em destaque
    x = int(46 * s)
    larg = texto(img, str(e["estavel"]), (x, int(46 * s)), 1.5 * s, BRANCO, max(1, int(2 * s)))
    x2 = x + larg + int(12 * s)
    texto(img, "codornas", (x2, int(46 * s)), 0.7 * s, CINZA)
    texto(img, f"agora {e['agora']}  |  pico {e['pico']}", (x2, int(22 * s)), 0.5 * s, CINZA)
 
    # Alerta no centro da barra
    if e["alerta"]:
        texto(img, f"ALERTA: abaixo de {e['minimo']}", (w // 2, int(40 * s)), 0.8 * s, VERMELHO,
              max(1, int(2 * s)), "centro")
 
    # Hora e desempenho à direita
    texto(img, datetime.now().strftime("%d/%m/%Y  %H:%M:%S"), (w - int(16 * s), int(26 * s)),
          0.6 * s, BRANCO, 1, "dir")
    texto(img, f"cam {e['cam_fps']:.0f} fps | proc {e['proc_fps']:.0f} fps | modelo {e['ms']:.0f} ms",
          (w - int(16 * s), int(50 * s)), 0.45 * s, CINZA, 1, "dir")
 
    # Mini-gráfico da contagem recente (canto inferior esquerdo)
    hist = e["historico"]
    if len(hist) >= 2:
        gw, gh = int(220 * s), int(56 * s)
        x0, y0 = int(12 * s), h - gh - int(12 * s)
        if y0 > barra:
            escurecer(img[y0:y0 + gh, x0:x0 + gw], 0.35)
            vmin, vmax = min(hist), max(hist)
            faixa = max(vmax - vmin, 1)
            n = len(hist)
            pts = np.array([[x0 + 6 + int(i * (gw - 12) / (n - 1)),
                             y0 + gh - 8 - int((v - vmin) * (gh - 16) / faixa)]
                            for i, v in enumerate(hist)], np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, CIANO, max(1, int(2 * s)), cv2.LINE_AA)
            texto(img, f"{vmin} a {vmax}", (x0 + gw - 4, y0 + int(14 * s)), 0.4 * s, CINZA, 1, "dir")
 
    # Moldura vermelha quando há alerta
    if e["alerta"]:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), VERMELHO, max(2, int(4 * s)))
 
 
def desenhar_sem_sinal(img):
    """Escurece a imagem e avisa que a câmera perdeu o sinal."""
    h, w = img.shape[:2]
    s = min(max(w / 960, 0.6), 1.6)
    escurecer(img, 0.4)
    texto(img, "SEM SINAL DA CAMERA", (w // 2, h // 2), 1.1 * s, VERMELHO, max(1, int(2 * s)), "centro")
    texto(img, "tentando reconectar...", (w // 2, h // 2 + int(40 * s)), 0.6 * s, CINZA, 1, "centro")
 
 
# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def registrar_csv(caminho, contagens):
    novo = not Path(caminho).exists()
    with open(caminho, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["data_hora", "media", "maximo", "minimo"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(sum(contagens) / len(contagens), 1),
            max(contagens),
            min(contagens),
        ])
 
 
def suavizar(atual, novo):
    """Média móvel exponencial para os números do painel não ficarem pulando."""
    return novo if atual == 0 else 0.9 * atual + 0.1 * novo
 
 
# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
 
    source = int(args.source) if str(args.source).isdigit() else args.source
 
    # Se não houver modelo treinado, cai para o pré-treinado (apenas "bird" genérico).
    if args.model == "best.pt" and not Path("best.pt").exists():
        print("[aviso] best.pt não encontrado. Usando yolo26n.pt (COCO, classe 'bird').")
        print("        Para resultado confiável com codornas, treine seu modelo (treinar.py).")
        args.model = "yolo26n.pt"
        if args.classes is None:
            args.classes = [COCO_BIRD_CLASS]
 
    print(f"[info] Carregando modelo: {args.model}")
    model = YOLO(args.model)
    # Aquecimento: a primeira inferência é sempre mais lenta; fazemos antes de abrir a câmera.
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
                  imgsz=args.imgsz, device=args.device, verbose=False)
 
    print(f"[info] Conectando em: {mascarar_url(source)}")
    leitor = LeitorCamera(source).iniciar()
 
    writer = None
    ids_vistos = set()
    mostrar = not args.no_show
    janela_aberta = False
    recentes = deque(maxlen=max(1, args.smooth))   # contagens brutas recentes
    historico = deque(maxlen=240)                  # contagens estáveis p/ o mini-gráfico
    contagens_janela = []
    pico = 0
    ultimo_log = time.time()
    t_amostra = 0.0
    baixo_desde = None
    alerta_enviado = False
    ultimo_id = 0
    t_ultimo_proc = 0.0
    t_aviso = time.time()
    t_aviso_sinal = 0.0
    t_banner = 0.0
    t_resumo = time.time()
    ultimo_anotado = None
    fps = ms_modelo = ms_janela = 0.0
    t_anterior = time.perf_counter()
    Path(args.alert_dir).mkdir(parents=True, exist_ok=True)
    reg = Registro(args.db, args.db_interval, args.captures_dir, args.capture_interval,
                args.train_dir, args.train_interval, args.keep_days,
                args.auto_dataset, args.auto_interval)
    try:
        while True:
            frame, fid = leitor.ler()
 
            # Sem quadro novo: espera um pouquinho (e mantém a janela responsiva)
            if frame is None or fid == ultimo_id:
                if leitor.fim:
                    print("[info] Fim do vídeo.")
                    break
                sem_sinal = (not leitor.arquivo) and (time.perf_counter() - leitor.ultimo_em > 3)
                if frame is None and time.time() - t_aviso > 5:
                    print("[aviso] Aguardando imagem da câmera...")
                    t_aviso = time.time()
                elif sem_sinal and time.time() - t_aviso_sinal > 10:
                    print(f"[aviso] Sem imagem da câmera há {time.perf_counter() - leitor.ultimo_em:.0f}s.")
                    t_aviso_sinal = time.time()
                if mostrar and janela_aberta:
                    if sem_sinal and ultimo_anotado is not None and time.time() - t_banner > 0.5:
                        img = ultimo_anotado.copy()
                        desenhar_sem_sinal(img)
                        cv2.imshow(NOME_JANELA, img)
                        t_banner = time.time()
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                else:
                    time.sleep(0.002)
                continue
 
            ultimo_id = fid
            if args.stride > 1 and fid % args.stride:
                continue
            agora_t = time.perf_counter()
            if args.max_fps > 0 and agora_t - t_ultimo_proc < 1.0 / args.max_fps:
                continue
            t_ultimo_proc = agora_t
 
            # Reduz imagens grandes: acelera detecção, desenho e janela
            if args.max_width and frame.shape[1] > args.max_width:
                esc = args.max_width / frame.shape[1]
                frame = cv2.resize(frame, (args.max_width, int(frame.shape[0] * esc)),
                                   interpolation=cv2.INTER_AREA)
 
            # Detecção + rastreamento (ByteTrack mantém um ID por ave entre os quadros)
            t0 = time.perf_counter()
            r = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                imgsz=args.imgsz,
                classes=args.classes,
                device=args.device,
                verbose=False,
            )[0]
            t1 = time.perf_counter()
 
            total = len(r.boxes)
            if r.boxes.id is not None:
                ids_vistos.update(int(i) for i in r.boxes.id.int().cpu().tolist())
 
            # Contagem estável: mediana dos últimos quadros (não pula quando uma ave some 1 quadro)
            recentes.append(total)
            estavel = int(round(statistics.median(recentes)))
            pico = max(pico, estavel)
            reg.quadro(total, estavel, r)
            agora = time.time()
            if agora - t_amostra >= 0.5:
                historico.append(estavel)
                t_amostra = agora
 
            # Alerta de contagem baixa (ex.: aves fora do campo de visão, queda de lote)
            alerta_ativo = False
            em_atencao = False
            if args.min_count > 0:
                if estavel < args.min_count:
                    baixo_desde = baixo_desde or agora
                    em_atencao = True
                    if agora - baixo_desde >= args.alert_seconds:
                        alerta_ativo = True
                else:
                    baixo_desde = None
                    alerta_enviado = False
            cor_estado = VERMELHO if alerta_ativo else (AMARELO if em_atencao else VERDE)
 
            # Desenho direto no quadro (sem cópia)
            limpo = frame.copy() if reg.vence_treino() else None
            amostra = reg.amostra_detectada(frame, r)
            anotado = frame
            desenhar_caixas(anotado, r)
            reg.fotos(anotado, limpo, total)
            if amostra is not None:
                print(f"[info] Amostra rotulada salva para revisão: {amostra}")
            t2 = time.perf_counter()
 
            fps = suavizar(fps, 1.0 / max(t2 - t_anterior, 1e-6))
            t_anterior = t2
            ms_modelo = suavizar(ms_modelo, (t1 - t0) * 1000)
 
            desenhar_hud(anotado, {
                "estavel": estavel, "agora": total, "pico": pico,
                "cor_estado": cor_estado, "alerta": alerta_ativo, "minimo": args.min_count,
                "cam_fps": leitor.fps, "proc_fps": fps, "ms": ms_modelo,
                "historico": list(historico),
            })
            ultimo_anotado = anotado
 
            if alerta_ativo and not alerta_enviado:
                nome = Registro.salvar(args.alert_dir, "alerta", anotado, estavel)
                print(f"[ALERTA] Contagem {estavel} abaixo de {args.min_count} "
                      f"há {args.alert_seconds}s. Foto salva em {nome}")
                alerta_enviado = True
 
            # Resumo no terminal a cada 10 s (útil para achar o gargalo)
            if agora - t_resumo >= 10:
                print(f"[info] contagem {estavel} (agora {total}, pico {pico}) | "
                      f"camera {leitor.fps:.1f} fps | proc {fps:.1f} fps | modelo {ms_modelo:.0f} ms | "
                      f"janela {ms_janela:.0f} ms | quadro {frame.shape[1]}x{frame.shape[0]}")
                t_resumo = agora
 
            # Registro periódico em CSV
            contagens_janela.append(estavel)
            if agora - ultimo_log >= args.log_interval:
                registrar_csv(args.log, contagens_janela)
                contagens_janela.clear()
                ultimo_log = agora
 
            # Gravação do vídeo anotado
            if args.save_video:
                if writer is None:
                    h, w = anotado.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.save_fps, (w, h))
                writer.write(anotado)
 
            # Janela de visualização
            if mostrar:
                t3 = time.perf_counter()
                try:
                    if not janela_aberta:
                        cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.imshow(NOME_JANELA, anotado)
                    janela_aberta = True
                    tecla = cv2.waitKey(1) & 0xFF
                    ms_janela = suavizar(ms_janela, (time.perf_counter() - t3) * 1000)
                    if tecla in (ord("q"), 27):
                        break
                except cv2.error:
                    print("[aviso] Sem ambiente gráfico. Rodando sem janela (use --no-show).")
                    mostrar = False
 
    except KeyboardInterrupt:
        print("\n[info] Encerrado pelo usuário.")
    finally:
        reg.fechar()
        leitor.parar()
        if contagens_janela:
            registrar_csv(args.log, contagens_janela)
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[info] Fim. Pico de contagem: {pico} | IDs únicos vistos: {len(ids_vistos)} "
              f"(pode superestimar se o ID trocar).")
 
 
if __name__ == "__main__":
    main()