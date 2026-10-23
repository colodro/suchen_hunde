"""API de inferência — reconhecimento de cães perdidos (TCC).

Serve o modelo de embeddings (TFLite) para o plugin WordPress.
Endpoints:
    GET  /health      verificação de vida (e "acorda" o serviço no plano gratuito do Render)
    POST /embedding   {"foto": "<base64>"}                          -> {"embedding": [...]}
    POST /search      {"foto": "<base64>", "candidatos": [...], "top_k": 10}
                                                                     -> {"embedding_novo": [...], "ranking": [...]}

Variáveis de ambiente (todas opcionais):
    PORT            porta HTTP (o Render define sozinho)
    API_TOKEN       se definido, todas as rotas (exceto /health) exigem "Authorization: Bearer <token>"
    CORS_ORIGINS    origens permitidas, separadas por vírgula (padrão: *)
    MODELO_PATH     caminho do .tflite (padrão: modelo_caes_v1.tflite ao lado deste arquivo)
    MAX_CANDIDATOS  máximo de candidatos por busca (padrão: 5000)
    MAX_UPLOAD_MB   tamanho máximo da requisição (padrão: 8)
    TFLITE_THREADS  threads do interpretador (padrão: 1)
"""
import base64
import binascii
import hmac
import io
import os
import threading

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter  # pesado: use só em testes locais

# ---------------------------------------------------------------- configuração
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # mesmos valores do treino
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PASTA = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.environ.get('MODELO_PATH', os.path.join(PASTA, 'modelo_caes_v1.tflite'))
API_TOKEN = os.environ.get('API_TOKEN', '')
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
MAX_CANDIDATOS = int(os.environ.get('MAX_CANDIDATOS', '5000'))
MAX_UPLOAD_MB = int(os.environ.get('MAX_UPLOAD_MB', '8'))
TFLITE_THREADS = int(os.environ.get('TFLITE_THREADS', '1'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
_origens = '*' if CORS_ORIGINS.strip() == '*' else [o.strip() for o in CORS_ORIGINS.split(',') if o.strip()]
CORS(app, origins=_origens)


# ---------------------------------------------------------------- modelo
# Carregado uma única vez, ao iniciar o processo.
print(f'Carregando modelo de {MODELO_PATH} ...', flush=True)
_interp = Interpreter(model_path=MODELO_PATH, num_threads=TFLITE_THREADS)
_interp.allocate_tensors()
_entrada = _interp.get_input_details()[0]
_saida = _interp.get_output_details()[0]
EMBEDDING_DIM = int(_saida['shape'][-1])
_lock = threading.Lock()   # o interpretador TFLite não é thread-safe


def preprocessar(img):
    """Mesmo pré-processamento do treino (redimensiona + normaliza com média/desvio do ImageNet)."""
    img = ImageOps.exif_transpose(img).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr[None, ...].astype(np.float32)


def calcular_embedding(img):
    arr = preprocessar(img)
    with _lock:
        _interp.set_tensor(_entrada['index'], arr)
        _interp.invoke()
        return _interp.get_tensor(_saida['index'])[0].copy()


# aquecimento: a primeira inferência é sempre a mais lenta
calcular_embedding(Image.new('RGB', (IMG_SIZE, IMG_SIZE)))
print(f'Modelo pronto (embedding de {EMBEDDING_DIM} dimensões).', flush=True)


# ---------------------------------------------------------------- erros e autenticação
class ErroRequisicao(Exception):
    def __init__(self, mensagem, status=400):
        super().__init__(mensagem)
        self.mensagem = mensagem
        self.status = status


@app.errorhandler(ErroRequisicao)
def _tratar_erro_requisicao(e):
    return jsonify({'erro': e.mensagem}), e.status


@app.errorhandler(Exception)
def _tratar_erro_geral(e):
    if isinstance(e, HTTPException):
        return jsonify({'erro': e.description}), e.code
    app.logger.exception('Erro inesperado')
    return jsonify({'erro': 'Erro interno do servidor'}), 500


@app.before_request
def _exigir_token():
    if not API_TOKEN or request.method == 'OPTIONS' or request.path == '/health':
        return None
    recebido = request.headers.get('Authorization', '')
    esperado = f'Bearer {API_TOKEN}'
    if not hmac.compare_digest(recebido.encode('utf-8'), esperado.encode('utf-8')):
        raise ErroRequisicao('Não autorizado', 401)
    return None


# ---------------------------------------------------------------- utilidades de entrada
def _json_obrigatorio():
    dados = request.get_json(silent=True)
    if not isinstance(dados, dict):
        raise ErroRequisicao('Corpo da requisição deve ser um JSON (objeto)')
    return dados


def _foto_para_imagem(dados):
    b64 = dados.get('foto')
    if not isinstance(b64, str) or not b64:
        raise ErroRequisicao('Campo "foto" (base64) é obrigatório')
    if ',' in b64:                       # aceita o prefixo "data:image/jpeg;base64,"
        b64 = b64.split(',', 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64, validate=False)))
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
        raise ErroRequisicao('Não foi possível ler a imagem em "foto"')


# ---------------------------------------------------------------- rotas
@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'modelo': os.path.basename(MODELO_PATH), 'embedding_dim': EMBEDDING_DIM})


@app.post('/embedding')
def endpoint_embedding():
    dados = _json_obrigatorio()
    emb = calcular_embedding(_foto_para_imagem(dados))
    return jsonify({'embedding': emb.tolist()})


@app.post('/search')
def endpoint_search():
    dados = _json_obrigatorio()

    candidatos = dados.get('candidatos')
    if not isinstance(candidatos, list) or not candidatos:
        raise ErroRequisicao('Campo "candidatos" (lista não vazia) é obrigatório')
    if len(candidatos) > MAX_CANDIDATOS:
        raise ErroRequisicao(f'Máximo de {MAX_CANDIDATOS} candidatos por busca', 413)

    try:
        top_k = int(dados.get('top_k', 10))
    except (TypeError, ValueError):
        raise ErroRequisicao('"top_k" deve ser um inteiro')
    top_k = max(1, min(top_k, 100))

    ids, vetores = [], []
    for i, c in enumerate(candidatos):
        if not isinstance(c, dict) or 'id_cao' not in c or 'embedding' not in c:
            raise ErroRequisicao(f'Candidato {i} inválido: use {{"id_cao": ..., "embedding": [...]}}')
        try:
            v = np.asarray(c['embedding'], dtype=np.float32)
        except (TypeError, ValueError):
            raise ErroRequisicao(f'Embedding do candidato {i} não é numérico')
        if v.shape != (EMBEDDING_DIM,):
            raise ErroRequisicao(f'Embedding do candidato {i} deve ter {EMBEDDING_DIM} valores')
        ids.append(c['id_cao'])
        vetores.append(v)

    emb_novo = calcular_embedding(_foto_para_imagem(dados))

    dist = np.linalg.norm(np.stack(vetores) - emb_novo[None, :], axis=1)
    sim = 1.0 / (1.0 + dist)                       # mesma definição usada no Notebook 4
    ordem = np.argsort(dist)[:top_k]

    ranking = [{'id_cao': ids[i], 'similaridade': float(sim[i]), 'distancia': float(dist[i])} for i in ordem]
    return jsonify({'embedding_novo': emb_novo.tolist(), 'ranking': ranking})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
