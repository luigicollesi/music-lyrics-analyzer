## V0.6 experimental — janelas CTC por frases e candidatos auditáveis

A execução híbrida agora oferece **dois modos** de recuperação acústica:
`--ctc-window-mode phrase` (padrão experimental) planeja janelas por linha/frase,
expande até 2,5 s as bordas de seções que começam/terminam com palavras ausentes
sem cruzar a seção vizinha, e gera candidatos CTC **sem modificar a timeline**
durante a inferência. Um resolvedor monotônico seleciona propostas e só então
atribui timestamps a palavras ainda não resolvidas. O modo anterior permanece
disponível via `--ctc-window-mode gap` para comparação.

```bash
python analise.py --music-path audio/Inferno.wav --lyrics-path lyrics/Inferno.md \
  --aligner hybrid --strategy hierarchical --ctc-window-mode phrase \
  --ctc-score-window 2 --ctc-threshold -2.5 --ctc-device cpu \
  --device cpu --resume --report --lrc \
  --output analysis/lyrics-analysis-v06.json
```

O modo por frases cria `lyrics-analysis-v06-ctc-candidates.json` ao lado do
resultado principal, com todas as hipóteses do decodificador que chegaram ao
estágio de scoring, inclusive as rejeitadas (e respectivos scores, janelas,
IDs e razões), além das decisões do resolvedor. A análise contém
`diagnostics.timeline_issues` para regredir e verificar ordenação antes de
sincronizar palavras ao vídeo. `--lrc` regenera o arquivo a partir da mesma
timeline; caso não solicitado e exista um LRC anterior, um aviso é emitido.

`--ctc-score-window` permite testar valores 1–100 frames; alterar esse valor
muda o significado de `--ctc-threshold`, que não é uma probabilidade
calibrada. Teste diversas configurações contra palavras com intervalos
anotados manualmente. O modelo CTC não prova que a letra foi cantada.

`--ctc-device cpu` (padrão) permite executar Whisper com `--device cuda`
sem carregar também o MMS 1B em uma GPU pequena. Use
`--ctc-device inherit` ou `--ctc-device cuda` apenas após validar a
memória disponível. O processo descarrega o Whisper antes de inicializar MMS.

**Limitações atuais:** o resolvedor preserva timestamps antigos e não refaz
a posição de âncoras automáticas potencialmente incorretas; o decoder ainda
deriva intervalos de palavras de `prepare_token_list`, não de uma
reconstrução fonética independente. Nenhuma meta de cobertura ou exatidão
foi validada nesta branch com áudio real. Não mescle à main sem benchmark.

## V0.5 experimental — alinhamento acústico CTC (MMS)

A nova opção `--aligner hybrid` mantém a localização e alinhamento hierárquico `stable-ts`, mas tenta preencher **lacunas** usando um modelo acústico CTC multilíngue, diretamente com a letra original. O modelo é carregado apenas quando solicitado e usa os tokens do tokenizer do idioma `por` (português). Esta primeira implementação é *grafêmica* (tokens do MMS), não um alinhador IPA/fonêmico independente.

Instale as dependências opcionais:

```bash
python -m pip install -e '.[align,ctc,test]'
```

Execute:

```bash
python analise.py --music-path audio/song.wav --lyrics-path lyrics/song.md --aligner hybrid --strategy hierarchical --ctc-language por --device cpu --report --resume
```

A análise `ctc` pura está disponível **somente** quando há um `--section-timings` com limites de seções comprovados; isso evita encaixar refrões em ocorrências incorretas. Cada tentativa CTC deve ocorrer em um intervalo de no máximo 20 s, sendo rejeitadas sequências maiores que 40 palavras. Os timestamps CTC são `tentative`, porque o alinhamento forçado não verifica se a letra foi realmente cantada. As demais palavras não são alteradas. Os métodos são identificados por `alignment.method=ctc-mms` e cada tentativa é registrada no JSON de diagnóstico. O limite de score `--ctc-threshold` usa log-probabilidade *não calibrada*; não o interprete como confiança.

O checkpoint padrão `facebook/mms-1b-all` é grande e sua licença CC-BY-NC restringe usos comerciais. O modelo não é incluído no repositório e precisa ser baixado na primeira utilização. Considere recursos de RAM e a licença antes de usar; `--aligner stable-ts` permanece o comportamento padrão.

A V0.5 ainda requer validação com áudio real: alta cobertura CTC não garante timestamps corretos. Não faça merge baseado apenas em testes unitários.

### Recuperação acústica de lacunas (V0.4 candidata)

Antes da segunda passagem de ASR, o pipeline agora tenta reexecutar `model.align` apenas nas sequências de palavras ainda sem horário. Os limites do recorte são obtidos da palavra já alinhada anterior e posterior, ou da janela local já localizada. Não se geram tempos por interpolação. Lacunas maiores que 12 segundos ou com mais de 18 palavras permanecem pendentes desta etapa. O resultado registra `diagnostics.attempts[].acoustic_gap_recovery`.

O JSON também inclui `coverage.timed`, `coverage.non_tentative`, `coverage.tentative` e `coverage.unresolved`. O número de palavras com horário, por si só, não mede a precisão dos tempos ou a certeza de que a palavra foi cantada.

## V0.4 (candidata): recuperação contextual de palavras

Com `--strategy hierarchical --recovery auto` (padrão), as palavras que permanecerem sem timestamps após o alinhamento forçado são novamente investigadas em janelas curtas. O sistema solicita timestamps **por palavra** ao ASR e compara a transcrição ruidosa com a **letra original** por ordem de ocorrência, similaridade de caracteres e âncoras exatas.

- `alignment.method=word-asr-exact`: palavra reconhecida exatamente na segunda passagem.
- `word-asr-fuzzy`: semelhança textual alta.
- `word-asr-contextual`: substituição recuperada entre duas âncoras confiáveis.
- `word-asr-contextual-edge`: substituição próxima de borda, com duas âncoras consecutivas e semelhança parcial.
- `alignment.status=tentative`: correspondência contextual inferida, **não confirmada acusticamente**. Não usar para animações frame-perfect sem revisão.

O sistema não reparte artificialmente os tempos de segmentos por palavra, não altera horários previamente alinhados e não inventa palavras ausentes do reconhecimento. A cobertura pode aumentar, mas precisão em músicas reais ainda precisa ser medida por comparação com anotações humanas. Use `--recovery none` para desabilitar a segunda passagem.

## V0.3 — Entonação vocal por palavra

A V0.3 adiciona estimativas acústicas opcionais de **pitch (F0)**, contorno melódico, intensidade RMS e evolução da intensidade para cada palavra alinhada. O processamento utiliza `librosa.pyin` em grupos de palavras com timestamps válidos, sem inventar dados para as palavras ausentes.

```bash
python analise.py --music-path audio/song.wav --lyrics-path lyrics/song.md --strategy hierarchical --prosody pyin --pitch-contour --vocals-path audio/vocals.wav --report --resume
```

Para experimentos, `--word-timings verified.json` permite calcular prosódia sobre horários conferidos manualmente, evitando executar ASR. Se não houver vocais separados, o resultado terá `reliability: low`, pois instrumentos podem dominar a frequência fundamental detectada.

Cada palavra recebe `vocal.pitch` (mediana F0, faixa 10º/90º percentil, tendência de altura, fração de frames sonoros, pontos de pitch opcionais com timestamps absolutos), `vocal.intensity` (RMS e pico em dBFS, ataque e tendência de energia) e `vocal.delivery` com características não verificadas deixadas como `null` (por exemplo, gritado, soproso, vibrato). O campo `vocal` inteiro é `null` quando não há alinhamento válido. O resultado também inclui `prosody` com cobertura de palavras analisadas.

Flags: `--prosody none|pyin` (padrão none); `--pitch-min-hz` (75); `--pitch-max-hz` (1000); `--pitch-hop-length` (256); `--pitch-contour` exporta a curva por frame. O esquema JSON passa a versão 2 quando a prosódia está habilitada; versão 1 sem prosódia.

**Avisos:** a análise mede propriedades acústicas, não emoções. Octavas, vozeamento e timestamps podem estar errados quando o canto é polifônico, quando há silêncio ou quando o alinhamento da letra é impreciso. Confira a gravação antes de efeitos sincronizados por frame. A música original e o stem de voz devem manter o mesmo tempo inicial e duração.

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

## Letras em Markdown (.md) e seções da música

A entrada recomendada é um arquivo **Markdown (.md)** com título opcional, observações do autor e seções em títulos de segundo nível (`##`). O analisador distingue *Chorus*, *Verse*, *Final Chorus*, *Bridge*, *Intro*, *Outro*, *Instrumental* e outros nomes comuns. Descrições entre parênteses são preservadas, e cada ocorrência de refrão recebe um ID diferente.

```markdown
# Inferno — Divina Comédia

**Letra definitiva fornecida pelo autor.** Essa nota não é cantada.

## Chorus (abertura)

Foi só uma vez, só uma vez,
Eu disse isso da última vez.

## Verse 1

Espinhos rasgam minha pele, mal consigo enxergar.

## Chorus (após Verse 1)

Foi só uma vez, só uma vez,

## Final Chorus

Foi mais de uma vez, mais de uma vez.
```

Executar:

```bash
python analise.py --music-path "audio/Inferno.wav" --lyrics-path "lyrics/Inferno.md" --language pt --model base --device cpu --lrc
```

O terminal lista as seções antes da inferência e, ao terminar, apresenta cada seção com sua categoria, início e fim estimados e quantidade de palavras alinhadas. Exemplo meramente ilustrativo:

```text
Letra: 4 seções, 6 linhas, 40 palavras
  01. [chorus] Chorus (abertura): ...
  02. [verse] Verse 1: ...
  03. [chorus] Chorus (após Verse 1): ...
  04. [final-chorus] Final Chorus: ...

Seções identificadas (tempos baseados nas palavras alinhadas):
  [chorus] Chorus (abertura): 00.00s–12.40s (...)
```

**Observação:** no JSON os objetos `sections[]`, `lines[]` e `words[]` são ligados por `section_id`; `sections[].type` indica a categoria e `sections[].detail` preserva o contexto como `abertura` ou `após Verse 1`. Os tempos da seção são derivados das palavras efetivamente alinhadas; se nenhuma palavra for alinhada, os tempos ficam `null`.

As linhas de título (`# ...`), etiquetas de seção (`## ...`) e anotações antes do primeiro verso não são enviadas para o modelo e não contam como palavras cantadas. O formato antigo com `[Chorus]` e `[Verse]` continua aceito, assim como letras simples em `.txt`. Em Markdown, coloque versos apenas sob títulos de seção reconhecidos; blocos sob títulos como `## Instruções` serão ignorados.

## V0.2: alinhamento hierárquico de canções

A estratégia padrão agora é **hierarchical**: reconhecimento auxiliar em janelas de 30 segundos, seleção monotônica de âncoras para identificar refrões repetidos e alinhamento de cada seção em sua própria região. Se não houver âncora, nenhuma palavra da seção recebe horário inventado.

```bash
python analise.py --music-path "audio/Inferno.wav" --lyrics-path "lyrics/Inferno.md" \
  --strategy hierarchical --language pt --device cpu --model base --resume --report --lrc
```

**Comparação:** `--strategy legacy` mantém o alinhamento global da V0.1.

**Teste com reconhecimento externo:** `--asr-segments recognized.json` recebe uma lista de segmentos `[{"text":"...", "start":12.0, "end":15.0}]` realmente observados.

**Âncoras verificadas manualmente:** `--section-timings sections.json` pode conter:

```json
{"section_0001":[12.0,23.5],"section_0002":[23.5,78.5]}
```

Cada intervalo delimita uma seção, sem deslocar os offsets do áudio original. Se uma seção não estiver fornecida, continuará pendente. Para usar um vocal já separado com a mesma linha de tempo do master, passe `--vocals-path audio/vocals.wav`.

**Cache:** `--resume` grava o reconhecimento auxiliar na pasta `analysis/.cache`, indexado por hash do vocal, idioma, modelo e tamanho das janelas. `--report` escreve `analysis/lyrics-analysis-report.json` com cobertura, âncoras e erros de consistência.

**Limitações:** a localização automática depende da qualidade do ASR no canto; uma frase não reconhecida pode permanecer sem horário. O valor de intensidade é um proxy acústico relativo, não emoção. A estratégia nova precisa de validação manual em áudio real; não existe ainda fallback WhisperX nem medição de precisão para a música completa.

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