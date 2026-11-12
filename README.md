# Music Lyrics Analyzer

Skill e CLI Python para extrair BPM, batidas, tempo de cada palavra cantada e **força acústica relativa (0–100)** de uma música e sua letra. Executa localmente; não envia o áudio para APIs. Alinhamento automático é feito com [stable-ts](https://github.com/jianfch/stable-ts), quando instalado. **Não funciona por adivinhação:** sem alinhamento, os tempos/forças ficam `null`.

## Requisitos

- Python **3.11 ou 3.12** (recomendado)
- FFmpeg / ffprobe no `PATH`
- Memória e CPU suficientes para modelos Whisper. GPU é opcional. Primeira execução do modelo pode baixar pesos públicos.

No Ubuntu:

```bash
sudo apt update && sudo apt install -y ffmpeg python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[align,test]'
```

No Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[align,test]"
```

Instale o FFmpeg pelo gerenciador de pacotes do sistema e confirme `ffmpeg -version`.

## Análise automática

```bash
python analise.py --music-path "audio/musica.wav" --lyrics-path "lyrics/letra.txt" --language pt --model base --device cpu --lrc
```

Opções: `--model tiny` reduz consumo e tende a sacrificar a precisão; `--model small` pode melhorar acurácia com maior custo. `--device cuda` exige PyTorch/CUDA compatíveis. `--output analysis/lyrics-analysis.json` altera o destino.

Opcionalmente, após instalar Demucs e suas dependências com `pip install -e '.[separation]'`:

```bash
python analise.py --music-path "audio/musica.wav" --lyrics-path "lyrics/letra.txt" --separator demucs --device cpu
```

Demucs é opt-in e o projeto original está arquivado; verifique se a instalação é compatível com a versão do seu PyTorch. A separação pode ser lenta. Sem separação, a força é medida no mix completo e **não representa a intensidade isolada da voz**.

## Testes sem modelos pesados

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python analise.py --help
python analise.py --music-path "audio/teste.wav" --lyrics-path "lyrics/teste.txt" --aligner none
```

Para testar alinhamentos conhecidos, crie `timestamps.json` como uma lista na ordem das ocorrências, por exemplo:

```json
[
  {"text":"Foi","start":0.1,"end":0.4},
  {"text":"só","start":0.5,"end":0.8}
]
```

E execute:

```bash
python analise.py --music-path "audio/teste.wav" --lyrics-path "lyrics/teste.txt" --word-timings "timestamps.json" --lrc
```

**Esses tempos são fornecidos pelo usuário para testes; não são produzidos pela análise.**

## Saídas

- `analysis/lyrics-analysis.json`: BPM e beats; linhas da letra; cada palavra com `start`, `end`, `strength`, características acústicas, método e confiança.
- `analysis/lyrics-analysis.lrc`: arquivo auxiliar com timestamp inicial por palavra, quando `--lrc`.
- Valores nulos significam que o alinhamento ou a medição não puderam ser feitos.

Os valores `strength` são **relativos às palavras alinhadas na faixa**, não representam decibéis ou emoção. O parâmetro `confidence` do alinhador, quando disponível, não é uma calibração garantida para vocais cantados. Repetições, coros, melismas e auto-tune exigem revisão humana. O BPM pode estar na relação metade/dobro.

## Integração com Music Video Director

A skill [music-video-director](https://github.com/luigicollesi/music-video-director) pode executar esta CLI e consumir `analysis/lyrics-analysis.json`, preservando o áudio como fonte temporal e o texto da letra como fonte ortográfica. Não reutilize estimativas como timestamps definitivos sem verificar trechos de baixa confiança.

## Status

V0.1: pipeline local, parser, cálculo de BPM, integração de alinhamento stable-ts, força acústica relativa, opcional Demucs, JSON/LRC, testes automatizados. Não foram concluídos testes de precisão com gravações reais e referência humana; o código não promete alinhamento perfeito.