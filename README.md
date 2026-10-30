# Detector de Codornas

Sistema de detecção, rastreamento e contagem de codornas usando YOLO e
Ultralytics. O projeto pode consumir uma webcam, uma câmera IP ou um fluxo
RTSP, registrar contagens e exibir um dashboard com o histórico.

## Requisitos

- Python 3.10 ou superior
- Uma câmera ou arquivo de vídeo
- Dependências listadas em `requirements.txt`

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp config.example.YAML config.YAML
```

Edite `config.YAML` e configure `source` para a sua câmera. O arquivo local
fica fora do Git para não expor endereços da rede.

## Uso

Para iniciar a detecção:

```bash
python detectar_codornas.py
```

Para iniciar o dashboard:

```bash
python dashboard.py
```

O treinamento usa o dataset configurado em `codornas.yaml`:

```bash
python treinar.py
```

Capturas, alertas, banco de dados, CSVs e resultados de treinamento são
gerados localmente e não são versionados.
