# API de inferência — reconhecimento de cães perdidos

Serviço Flask que extrai embeddings de fotos de cães (modelo TFLite) e ranqueia candidatos por similaridade.
Consumido pelo plugin WordPress.

## Estrutura do repositório

```
app.py                  API
requirements.txt        dependências
Dockerfile              imagem para o Render
.dockerignore
testar_api.py           teste ponta a ponta (opcional)
modelo_caes_v1.tflite   <- gerado pelo notebook (célula "Exportar TFLite"), copie do Drive para cá
```

## 1. Obter o modelo

Rode no Colab a célula **Exportar TFLite** (logo após a exportação do modelo). Ela cria
`MyDrive/tcc_caes/api_render/modelo_caes_v1.tflite`. Baixe o arquivo e coloque na raiz deste repositório
(ao lado do `app.py`). Ele tem ~60 MB: o GitHub avisa acima de 50 MB, mas só bloqueia acima de 100 MB.

## 2. Subir para o GitHub

Crie antes um repositório vazio no GitHub (pode ser privado) e, dentro desta pasta:

```bash
git init
git add .
git commit -m "API de inferência"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

## 3. Criar o serviço no Render

1. **New → Web Service** e conecte o repositório (para repositório privado, autorize o app do Render no GitHub).
2. **Language:** Docker (o `Dockerfile` é detectado na raiz). **Instance Type:** Free.
3. Em **Advanced → Health Check Path:** `/health`.
4. **Environment Variables:**
   - `API_TOKEN` = uma chave longa e aleatória. Para gerar:
     `python -c "import secrets; print(secrets.token_urlsafe(32))"`
   - `CORS_ORIGINS` = (opcional) endereço do seu site, ex.: `https://meusite.com.br`. Como o plugin chama a API pelo
     servidor PHP, o CORS só importa se algum JavaScript do navegador chamar a API diretamente.
5. **Create Web Service**. Ao final do build você recebe uma URL `https://NOME.onrender.com`.

## 4. Testar

```bash
python testar_api.py https://NOME.onrender.com caminho/da/foto.jpg --token SUA_CHAVE
```

O teste chama `/health`, `/embedding` e `/search` (com um candidato idêntico, que deve vir em 1º lugar com distância ~0).
Para testar localmente: `pip install -r requirements.txt` e `python app.py` (porta 10000).

## 5. Contrato da API (para o plugin)

Todas as rotas, exceto `/health`, exigem o cabeçalho `Authorization: Bearer SUA_CHAVE` quando `API_TOKEN` está definido.

| Rota | Entrada (JSON) | Saída (JSON) |
|---|---|---|
| `GET /health` | — | `{"status":"ok","modelo":"...","embedding_dim":128}` |
| `POST /embedding` | `{"foto":"<base64>"}` | `{"embedding":[... 128 valores ...]}` |
| `POST /search` | `{"foto":"<base64>","candidatos":[{"id_cao":1,"embedding":[...]}],"top_k":10}` | `{"embedding_novo":[...],"ranking":[{"id_cao":1,"similaridade":0.91,"distancia":0.10}]}` |

- `similaridade = 1 / (1 + distância)`; a distância é a euclidiana entre embeddings L2-normalizados (varia de 0 a 2).
- O limiar de decisão do plugin deve ser definido a partir da avaliação (distância no EER do conjunto de teste),
  convertido em similaridade com a mesma fórmula.
- Erros retornam `{"erro":"mensagem"}` com status 400 (entrada inválida), 401 (token), 413 (muito grande) ou 500.

Exemplo de chamada no plugin (PHP):

```php
$resposta = wp_remote_post( $base_url . '/search', array(
    'timeout' => 90,   // a 1ª chamada após hibernação leva ~1 min no plano gratuito
    'headers' => array(
        'Content-Type'  => 'application/json',
        'Authorization' => 'Bearer ' . $token,
    ),
    'body'    => wp_json_encode( array(
        'foto'       => $foto_base64,
        'candidatos' => $candidatos,   // [ ['id_cao'=>1,'embedding'=>[...]], ... ]
        'top_k'      => 10,
    ) ),
) );
```

## 6. Limitações do plano gratuito do Render

- O serviço **hiberna após ~15 min sem requisições** e leva cerca de 1 minuto para acordar. Use timeout longo no plugin e,
  se quiser, chame `/health` quando a página de busca for aberta para "acordar" o serviço.
- 512 MB de RAM e CPU compartilhada: a inferência de uma foto pode levar alguns segundos.
- As condições do plano mudam com frequência; confira em https://render.com/docs/free.

## Problemas comuns

- **`pip` não encontra `ai-edge-litert` durante o build:** troque a linha por `tflite-runtime` em `requirements.txt`
  (o `app.py` tenta os dois). Se nenhum funcionar para a sua versão do Python, use `tensorflow-cpu` apenas como último recurso
  (a imagem fica pesada e pode estourar os 512 MB).
- **Build falha em `COPY ... modelo_caes_v1.tflite`:** o arquivo do modelo não está na raiz do repositório.
- **"Out of memory" nos logs do Render:** o plano gratuito não comportou o serviço; considere um plano pago de entrada
  ou uma VM gratuita com mais memória.
