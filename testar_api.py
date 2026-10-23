"""Teste ponta a ponta da API (só biblioteca padrão do Python).

Uso:
    python testar_api.py https://SEU-SERVICO.onrender.com caminho/da/foto.jpg --token SEU_TOKEN
    python testar_api.py http://localhost:10000 foto.jpg            (API rodando localmente, sem token)
"""
import argparse
import base64
import json
import math
import sys
import time
import urllib.error
import urllib.request


def chamar(url, metodo='GET', corpo=None, token=None, timeout=180):
    cabecalhos = {'Content-Type': 'application/json'}
    if token:
        cabecalhos['Authorization'] = f'Bearer {token}'
    dados = json.dumps(corpo).encode('utf-8') if corpo is not None else None
    req = urllib.request.Request(url, data=dados, headers=cabecalhos, method=metodo)
    inicio = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode('utf-8')), time.time() - inicio
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8') or '{}'), time.time() - inicio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('foto')
    ap.add_argument('--token', default=None)
    a = ap.parse_args()
    base = a.url.rstrip('/')

    with open(a.foto, 'rb') as f:
        foto_b64 = base64.b64encode(f.read()).decode('ascii')

    print('1) /health (se o serviço estava dormindo, esta chamada pode levar ~1 minuto)')
    st, resp, t = chamar(f'{base}/health')
    print(f'   {st} em {t:.1f}s -> {resp}')
    if st != 200:
        sys.exit('Falha no /health')

    print('2) /embedding')
    st, resp, t = chamar(f'{base}/embedding', 'POST', {'foto': foto_b64}, a.token)
    print(f'   {st} em {t:.1f}s')
    if st != 200:
        sys.exit(f'Falha: {resp}')
    emb = resp['embedding']
    norma = math.sqrt(sum(x * x for x in emb))
    print(f'   {len(emb)} dimensões | norma L2 = {norma:.4f} (esperado ~1.0)')

    print('3) /search (candidato 1 = o próprio embedding; candidato 2 = vetor qualquer)')
    qualquer = [1 / math.sqrt(len(emb))] * len(emb)
    corpo = {'foto': foto_b64, 'top_k': 5,
             'candidatos': [{'id_cao': 1, 'embedding': emb}, {'id_cao': 2, 'embedding': qualquer}]}
    st, resp, t = chamar(f'{base}/search', 'POST', corpo, a.token)
    print(f'   {st} em {t:.1f}s')
    if st != 200:
        sys.exit(f'Falha: {resp}')
    for item in resp['ranking']:
        print(f"   id_cao={item['id_cao']}  similaridade={item['similaridade']:.4f}  distancia={item['distancia']:.4f}")
    ok = resp['ranking'][0]['id_cao'] == 1 and resp['ranking'][0]['distancia'] < 0.05
    print('RESULTADO:', 'OK' if ok else 'ATENÇÃO: o candidato idêntico não ficou em 1º com distância ~0')


if __name__ == '__main__':
    main()
