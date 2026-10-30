#!/usr/bin/env python3
"""
Grava o que o dashboard.py lê: o banco dados.db e as fotos (capturas, alertas, treino).
É usado pelo detectar_codornas.py; não precisa rodar sozinho.

Tabela `amostras` (uma linha a cada `intervalo` segundos):
    ts         instante (segundos desde 1970, inteiro)
    quadros    quadros analisados nesse intervalo
    marcacoes  soma das codornas marcadas nesses quadros
    agora      contagem bruta do último quadro
    estavel    contagem estável (mediana) do último quadro
    centros    centro de cada codorna do último quadro, "x,y;x,y" (0 a 1)

Nome das fotos: <prefixo>_AAAAMMDD_HHMMSS_n<contagem>.jpg (o painel lê o "_n" para o selo).
As amostras automáticas também geram um arquivo YOLO com o mesmo nome na pasta
de rótulos.
"""
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import cv2


class Registro:
    def __init__(self, db="dados.db", intervalo=10, capturas="capturas", cap_intervalo=300,
                 treino="dataset/images/raw", treino_intervalo=0, dias=30,
                 auto_dataset="dataset", auto_intervalo=30):
        self.intervalo = max(1, intervalo)
        self.capturas = Path(capturas)
        self.cap_intervalo = cap_intervalo
        self.treino = Path(treino)
        self.treino_intervalo = treino_intervalo
        self.dias = dias
        self.auto_dataset = Path(auto_dataset)
        self.auto_intervalo = auto_intervalo
        self.capturas.mkdir(parents=True, exist_ok=True)
        if treino_intervalo > 0:
            self.treino.mkdir(parents=True, exist_ok=True)
        if auto_intervalo > 0:
            (self.auto_dataset / "images" / "train").mkdir(parents=True, exist_ok=True)
            (self.auto_dataset / "labels" / "train").mkdir(parents=True, exist_ok=True)

        self.con = sqlite3.connect(db, timeout=10)
        # WAL: o painel lê enquanto o detector grava, sem "database is locked"
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS amostras (ts INTEGER, quadros INTEGER, "
                         "marcacoes INTEGER, agora INTEGER, estavel INTEGER, centros TEXT)")
        self.con.execute("CREATE INDEX IF NOT EXISTS i_amostras_ts ON amostras(ts)")
        self.con.commit()

        agora = time.time()
        self._quadros = self._marc = self._agora = self._estavel = 0
        self._centros = ""
        self._t_amostra = agora
        self._t_cap = 0.0           # a primeira captura sai logo no começo
        self._t_treino = agora
        self._t_auto = 0.0          # a primeira detecção pode ser salva logo no começo
        self._limpar(agora)

    # -- banco ---------------------------------------------------------------
    def quadro(self, total, estavel, r):
        """Chame a cada quadro processado. `r` é o resultado do model.track()."""
        self._quadros += 1
        self._marc += total
        self._agora, self._estavel = total, estavel
        self._centros = (";".join(f"{x:.3f},{y:.3f}" for x, y in r.boxes.xywhn[:, :2].tolist())
                         if total else "")
        agora = time.time()
        if agora - self._t_amostra >= self.intervalo:
            self._gravar(agora)

    def _gravar(self, agora):
        if self._quadros:
            self.con.execute("INSERT INTO amostras VALUES (?,?,?,?,?,?)",
                             (int(agora), self._quadros, self._marc, self._agora,
                              self._estavel, self._centros))
            self.con.commit()
        self._quadros = self._marc = 0
        self._t_amostra = agora
        if agora - self._t_limpeza >= 3600:
            self._limpar(agora)

    def _limpar(self, agora):
        """Apaga amostras e capturas mais velhas que `dias` (alertas e treino ficam)."""
        self._t_limpeza = agora
        if self.dias <= 0:
            return
        corte = agora - self.dias * 86400
        self.con.execute("DELETE FROM amostras WHERE ts<?", (int(corte),))
        self.con.commit()
        for f in self.capturas.glob("*.jpg"):
            try:
                if f.stat().st_mtime < corte:
                    f.unlink()
            except OSError:
                pass

    # -- fotos ---------------------------------------------------------------
    def vence_treino(self):
        """True se já está na hora de guardar uma foto SEM caixas para treino."""
        return self.treino_intervalo > 0 and time.time() - self._t_treino >= self.treino_intervalo

    def fotos(self, anotado, limpo, n):
        """Salva a captura periódica (com caixas) e, se houver, a foto limpa de treino."""
        agora = time.time()
        if self.cap_intervalo > 0 and agora - self._t_cap >= self.cap_intervalo:
            self.salvar(self.capturas, "captura", anotado, n)
            self._t_cap = agora
        if limpo is not None:
            self.salvar(self.treino, "treino", limpo)
            self._t_treino = agora

    def amostra_detectada(self, anotado, r):
        """Salva uma detecção e seu rótulo YOLO para posterior revisão/treino."""
        if self.auto_intervalo <= 0 or not len(r.boxes):
            return None
        agora = time.time()
        if agora - self._t_auto < self.auto_intervalo:
            return None

        base = f"auto_{datetime.now():%Y%m%d_%H%M%S_%f}"
        imagens = self.auto_dataset / "images" / "train"
        rotulos = self.auto_dataset / "labels" / "train"
        imagem = imagens / f"{base}.jpg"
        rotulo = rotulos / f"{base}.txt"

        if not cv2.imwrite(str(imagem), anotado, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"Não foi possível salvar a amostra detectada em {imagem}")

        caixas = r.boxes.xywhn.cpu().tolist()
        linhas = []
        for caixa in caixas:
            x, y, largura, altura = caixa
            linhas.append(f"0 {x:.6f} {y:.6f} {largura:.6f} {altura:.6f}")
        try:
            rotulo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
        except OSError:
            imagem.unlink(missing_ok=True)
            raise

        self._t_auto = agora
        return imagem

    @staticmethod
    def salvar(pasta, prefixo, img, n=None):
        Path(pasta).mkdir(parents=True, exist_ok=True)
        sufixo = f"_n{n}" if n is not None else ""
        caminho = Path(pasta) / f"{prefixo}_{datetime.now():%Y%m%d_%H%M%S}{sufixo}.jpg"
        cv2.imwrite(str(caminho), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return caminho

    def fechar(self):
        try:
            self._gravar(time.time())
        finally:
            self.con.close()