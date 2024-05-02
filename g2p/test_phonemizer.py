# -*- coding: utf-8 -*-
from phonemizer import phonemize
from phonemizer.separator import Separator

text = "Ah! N'importe, vieux farceur! Tu ne sens pas bon!"

phn = phonemize(
    text,
    language='fr-fr',
    separator=Separator(phone='#', word="@", syllable=None),
    strip=False,
    preserve_punctuation=True)

print(phn.replace('#', ' ').replace('@',' _ ').split())