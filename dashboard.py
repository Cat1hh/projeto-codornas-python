#!/usr/bin/env python3
"""
Painel web local das codornas (não precisa instalar nada além do Python).

    python dashboard.py                  # abre em http://localhost:8000
    python dashboard.py --host 0.0.0.0   # também acessível por outros aparelhos da mesma rede

Lê o banco dados.db e as pastas de fotos geradas pelo detectar_codornas.py.
Sem senha: use --host 0.0.0.0 somente em rede de confiança.
"""
import argparse
import json
import mimetypes
import re
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

DB = "dados.db"
PASTAS = {}
EXT = {".jpg", ".jpeg", ".png"}


def consulta(sql, args=()):
    if not Path(DB).exists():
        return []
    con = sqlite3.connect(DB, timeout=5)
    try:
        return con.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


def api_resumo(horas):
    t0 = int(time.time() - horas * 3600)
    quadros, marc, media, pico = (consulta(
        "SELECT SUM(quadros), SUM(marcacoes), AVG(estavel), MAX(estavel) FROM amostras WHERE ts>=?",
        (t0,)) or [(None,) * 4])[0]
    ult = consulta("SELECT ts, agora, estavel FROM amostras ORDER BY ts DESC LIMIT 1")
    bucket = max(60, int(horas * 3600 / 90))
    serie = consulta(
        "SELECT ts/?*?, AVG(estavel), MAX(estavel) FROM amostras WHERE ts>=? GROUP BY 1 ORDER BY 1",
        (bucket, bucket, t0))
    return {"quadros": quadros, "marcacoes": marc, "media": media, "pico": pico,
            "ultima": {"ts": ult[0][0], "agora": ult[0][1], "estavel": ult[0][2]} if ult else None,
            "serie": serie}


def api_mapa(horas, gx=48, gy=27):
    t0 = int(time.time() - horas * 3600)
    grade = [[0] * gx for _ in range(gy)]
    for (c,) in consulta("SELECT centros FROM amostras WHERE ts>=? AND centros!='' "
                         "ORDER BY ts DESC LIMIT 200000", (t0,)):
        for par in c.split(";"):
            try:
                x, y = par.split(",")
                grade[min(gy - 1, max(0, int(float(y) * gy)))][min(gx - 1, max(0, int(float(x) * gx)))] += 1
            except ValueError:
                pass
    return {"gx": gx, "gy": gy, "grade": grade}


def api_fotos(pasta, limite, desloc):
    d = PASTAS.get(pasta)
    if not d or not d.is_dir():
        return {"total": 0, "itens": []}
    arqs = sorted((f for f in d.iterdir() if f.suffix.lower() in EXT),
                  key=lambda f: f.stat().st_mtime, reverse=True)
    itens = []
    for f in arqs[desloc:desloc + limite]:
        m = re.search(r"_n(\d+)", f.stem)
        itens.append({"nome": f.name, "ts": int(f.stat().st_mtime), "n": int(m.group(1)) if m else None})
    return {"total": len(arqs), "itens": itens}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def enviar(self, corpo, tipo, cache="no-store"):
        self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        def num(k, padrao):
            try:
                return float(q[k][0])
            except (KeyError, ValueError):
                return padrao

        try:
            if u.path == "/":
                return self.enviar(HTML.encode(), "text/html; charset=utf-8")
            if u.path == "/api/resumo":
                return self.enviar(json.dumps(api_resumo(num("horas", 24))).encode(), "application/json")
            if u.path == "/api/mapa":
                return self.enviar(json.dumps(api_mapa(num("horas", 24))).encode(), "application/json")
            if u.path == "/api/fotos":
                r = api_fotos(q.get("pasta", ["capturas"])[0], int(num("limite", 24)), int(num("desloc", 0)))
                return self.enviar(json.dumps(r).encode(), "application/json")
            if u.path.startswith("/foto/"):
                _, _, pasta, nome = u.path.split("/", 3)
                d = PASTAS.get(pasta)
                f = d / Path(unquote(nome)).name if d else None
                if f and f.is_file() and f.suffix.lower() in EXT:
                    return self.enviar(f.read_bytes(), mimetypes.guess_type(f.name)[0] or "image/jpeg",
                                       "max-age=3600")
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self.send_error(500, str(e))


HTML = r"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Codornas · Painel</title>
<style>
:root{--bg:#f6f5f1;--card:#fff;--tx:#1d2420;--mu:#6b746d;--ac:#2f7d4f;--ac2:#d98a1c;--bd:#e3e1d9}
@media(prefers-color-scheme:dark){:root{--bg:#141815;--card:#1d231f;--tx:#e9ece8;--mu:#99a39b;--ac:#58c07f;--ac2:#f0a63a;--bd:#2c342e}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.4 system-ui,sans-serif}
header{max-width:1100px;margin:0 auto;display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:18px 20px}
h1{font-size:21px;margin:0}h2{font-size:15px;margin:0 0 10px}
.st{font-size:13px;color:var(--mu)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--mu);margin-right:6px}.on .dot{background:var(--ac)}
.seg button,.btn,select{font:inherit;color:var(--tx);background:var(--card);border:1px solid var(--bd);padding:6px 12px;cursor:pointer}
.seg button{border-radius:0}.seg button:first-child{border-radius:8px 0 0 8px}.seg button:last-child{border-radius:0 8px 8px 0}
.seg .ativo{background:var(--ac);border-color:var(--ac);color:#fff}.btn,select{border-radius:8px}
main{max-width:1100px;margin:0 auto;padding:0 20px 40px;display:grid;gap:16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpi b{display:block;font-size:30px;line-height:1.1}.kpi span{color:var(--mu);font-size:13px}
.g2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
canvas{width:100%;display:block;border-radius:8px}#grafico{height:240px}#mapa{aspect-ratio:16/9}
.leg{font-size:12px;color:var(--mu);margin:8px 0 0}
.top{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}.top h2{margin:0}
.fotos{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
figure{margin:0;cursor:pointer;position:relative}
figure img{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:8px;display:block;background:var(--bd)}
figcaption{font-size:12px;color:var(--mu);margin-top:4px}
.badge{position:absolute;top:6px;left:6px;background:rgba(0,0,0,.65);color:#fff;border-radius:6px;padding:1px 7px;font-size:12px}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}
#lb img{max-width:94vw;max-height:94vh}
</style></head><body>
<header>
 <div><h1>Codornas · Painel</h1><div class="st" id="st"><span class="dot"></span><span id="sttx">Carregando…</span></div></div>
 <div class="seg" id="seg"><button data-h="1">1 h</button><button data-h="6">6 h</button><button data-h="24" class="ativo">24 h</button><button data-h="168">7 dias</button></div>
</header>
<main>
 <section class="kpis">
  <div class="card kpi"><b id="k-agora">–</b><span>codornas agora (contagem estável)</span></div>
  <div class="card kpi"><b id="k-pico">–</b><span>pico no período</span></div>
  <div class="card kpi"><b id="k-media">–</b><span>média no período</span></div>
  <div class="card kpi"><b id="k-marc">–</b><span>marcações feitas pela IA</span></div>
  <div class="card kpi"><b id="k-quadros">–</b><span>quadros analisados</span></div>
 </section>
 <section class="g2">
  <div class="card"><h2>Contagem ao longo do tempo</h2><canvas id="grafico"></canvas>
   <p class="leg">Linha verde: média. Linha laranja: máximo em cada intervalo.</p></div>
  <div class="card"><h2>Onde a IA marcou</h2><canvas id="mapa"></canvas>
   <p class="leg" id="mapaleg"></p></div>
 </section>
 <section class="card">
  <div class="top"><h2>Fotos</h2>
   <select id="pasta"><option value="capturas">Capturas da IA</option><option value="alertas">Alertas</option><option value="treino">Fotos para treino</option></select></div>
  <div class="fotos" id="fotos"></div>
  <p class="leg" id="vazio" style="display:none">Nenhuma foto nesta pasta ainda.</p>
  <p style="text-align:center"><button class="btn" id="mais" style="display:none">Ver mais</button></p>
 </section>
</main>
<div id="lb"><img alt=""></div>
<script>
const $=s=>document.querySelector(s);let horas=24,pasta='capturas',off=0;
const J=u=>fetch(u).then(r=>r.json());
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hm=t=>new Date(t*1000).toLocaleString('pt-BR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
function cv(id){const c=$(id),d=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.scale(d,d);return[x,w,h]}
function linha(s){const[x,w,h]=cv('#grafico');x.clearRect(0,0,w,h);
 if(!s.length){x.fillStyle=css('--mu');x.font='14px system-ui';x.fillText('Sem dados no período',16,30);return}
 const L=34,B=22,T=10,R=8,mx=Math.max(1,...s.map(p=>p[2])),t0=s[0][0],t1=Math.max(s[s.length-1][0],t0+1);
 const X=t=>L+(t-t0)/(t1-t0)*(w-L-R),Y=v=>T+(1-v/mx)*(h-T-B);
 x.strokeStyle=css('--bd');x.fillStyle=css('--mu');x.font='11px system-ui';
 for(let i=0;i<=4;i++){const v=mx*i/4,y=Y(v);x.beginPath();x.moveTo(L,y);x.lineTo(w-R,y);x.stroke();x.fillText(Math.round(v),4,y+4)}
 x.fillText(hm(t0),L,h-6);x.textAlign='right';x.fillText(hm(t1),w-R,h-6);x.textAlign='left';
 const tr=(k,c,lw)=>{x.beginPath();s.forEach((p,i)=>i?x.lineTo(X(p[0]),Y(p[k])):x.moveTo(X(p[0]),Y(p[k])));x.strokeStyle=c;x.lineWidth=lw;x.stroke()};
 tr(2,css('--ac2'),1.2);tr(1,css('--ac'),2.5)}
async function mapa(){const[g,f]=await Promise.all([J('/api/mapa?horas='+horas),J('/api/fotos?pasta=capturas&limite=1')]);
 const c=$('#mapa');let img=null;
 if(f.itens.length){img=new Image();img.src='/foto/capturas/'+f.itens[0].nome;try{await img.decode()}catch(e){img=null}}
 if(img)c.style.aspectRatio=img.naturalWidth+'/'+img.naturalHeight;
 const[x,w,h]=cv('#mapa');
 if(img){x.drawImage(img,0,0,w,h);x.fillStyle='rgba(0,0,0,.35)'}else x.fillStyle=css('--bd');
 x.fillRect(0,0,w,h);
 const mx=Math.max(1,...g.grade.flat()),cw=w/g.gx,ch=h/g.gy;
 g.grade.forEach((r,j)=>r.forEach((v,i)=>{if(v){const t=Math.sqrt(v/mx);x.fillStyle=`hsla(${60-60*t},95%,50%,${.25+.55*t})`;x.fillRect(i*cw,j*ch,cw,ch)}}));
 $('#mapaleg').textContent=g.grade.flat().some(v=>v)?'Quanto mais quente a cor, mais vezes a IA marcou uma codorna naquele ponto da imagem.':'Ainda sem marcações no período.'}
async function carregar(){const r=await J('/api/resumo?horas='+horas),n=v=>(v||0).toLocaleString('pt-BR');
 $('#k-agora').textContent=r.ultima?r.ultima.estavel:'–';$('#k-pico').textContent=r.pico??'–';
 $('#k-media').textContent=r.media!=null?r.media.toFixed(1):'–';$('#k-marc').textContent=n(r.marcacoes);$('#k-quadros').textContent=n(r.quadros);
 const on=r.ultima&&Date.now()/1000-r.ultima.ts<60;$('#st').className='st '+(on?'on':'');
 $('#sttx').textContent=on?'Detector ativo · '+hm(r.ultima.ts):(r.ultima?'Detector parado · último dado '+hm(r.ultima.ts):'Sem dados ainda');
 linha(r.serie);mapa()}
async function fotos(mais){if(!mais)off=0;const f=await J(`/api/fotos?pasta=${pasta}&limite=24&desloc=${off}`),g=$('#fotos');
 if(!mais)g.innerHTML='';
 f.itens.forEach(i=>{const e=document.createElement('figure'),u='/foto/'+pasta+'/'+i.nome;
  e.innerHTML=`<img loading="lazy" src="${u}" alt=""><figcaption>${hm(i.ts)}</figcaption>`+(i.n!=null?`<span class="badge">${i.n} codornas</span>`:'');
  e.onclick=()=>{$('#lb img').src=u;$('#lb').style.display='flex'};g.appendChild(e)});
 off+=f.itens.length;$('#mais').style.display=off<f.total?'inline-block':'none';$('#vazio').style.display=f.total?'none':'block'}
$('#lb').onclick=()=>$('#lb').style.display='none';
$('#mais').onclick=()=>fotos(true);$('#pasta').onchange=e=>{pasta=e.target.value;fotos()};
document.querySelectorAll('#seg button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#seg button').forEach(x=>x.classList.remove('ativo'));b.classList.add('ativo');horas=+b.dataset.h;carregar()});
addEventListener('resize',carregar);carregar();fotos();setInterval(carregar,10000);
</script></body></html>
"""


def main():
    global DB
    p = argparse.ArgumentParser(description="Painel web local das codornas")
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = só este PC; 0.0.0.0 = toda a rede")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--db", default="dados.db")
    p.add_argument("--capturas", default="capturas", help="Pasta das fotos periódicas da IA")
    p.add_argument("--alertas", default="alertas", help="Pasta das fotos de alerta")
    p.add_argument("--treino", default="dataset/images/raw", help="Pasta das fotos para treino")
    a = p.parse_args()
    DB = a.db
    PASTAS.update({"capturas": Path(a.capturas), "alertas": Path(a.alertas), "treino": Path(a.treino)})
    servidor = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[info] Painel em http://localhost:{a.port}  (Ctrl+C para parar)")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] Painel encerrado.")


if __name__ == "__main__":
    main()