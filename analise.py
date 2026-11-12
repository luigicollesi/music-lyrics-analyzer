#!/usr/bin/env python3
"""Executável: python analise.py --music-path ... --lyrics-path ..."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from music_lyrics_analyzer.cli import main
if __name__=='__main__':
    raise SystemExit(main())
