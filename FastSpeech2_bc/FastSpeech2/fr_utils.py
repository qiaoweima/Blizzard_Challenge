import os

def get_phone_dict():
    phones_dict = set()
    with open("fr_lexicon_cleaned.dict") as f:
        lines = f.readlines()
        for line in lines:
            _, phones = line.split("\t")
            phones = phones.strip('\n').split(' ')
            for p in phones:
                if p != '':
                    phones_dict.add(p)
    return list(phones_dict)

def get_graphme_dict():
    graphme_dict = set()
    with open("fr_lexicon_cleaned.dict") as f:
        lines = f.readlines()
        for line in lines:
            text, _ = line.split("\t")
            graphmes = [c for c in text]
            for g in graphmes:
                if g != '':
                    graphme_dict.add(g)
    return list(graphme_dict)

_phones = ['ɔ̱', 's', 'ɛ̃̃', 'n̩', 'oˑ', 'l', 't', 'à', 'ε', 'õ', 'O', 'ø͜', 't͡', 
          'ɜ̃', 'ɵ', 'ʊ˞', 'ɔ̯', 'ɐ', 'ʰ', 'ə͜', 'h', 'sˁ', 'ɥ', 'ã', 'i̯', 'øː', 
          'ô', 'ɟ', 'ɔ', '̃', 'ɛ̃ː', 'ĩ', 'ʁ', 'ɛ̃', 'ɛ', 'u̯', 'o̞', 'ç', 'ᴣ', 'M', 
          '∫', 'aː', 'ū', 'ŋ̥', 'aʰ', 'v', 'ā', 'ɨ', 'u', 'ˈ', 'œː', 'n˨˩', 'ʏ', 
          'ɘ', 'm̭', '̭', 'ɔ́', 's̩ː', 'y', 'd͡', 'z', 'è', 'n̥', 'ðˁ', '2', 'ł', 'a˥', 
          'ɴ', 'ɒ', 'ɔ̃', 'ñ', 'χ', 'ʀ', 'ɾ', 'm̃', 'ɬ', 'ʎ', 'ɳ', 'ᵊ', 'ɑ̃ː', 'f',
          'n', 'eː', 'ɑː', 'sː', 'uː', 'ḫ', 'ɡ', 'o', 'ʕ', 'd', 'iː', 'w', 'Y', 
          'yː', 'p͡', 'β', 'æ', 'ɑ̃', 'ᴚ', 'ɹ', 'āʾ', 'ɬ͡', 'ḱ', 'ɒː', 'ə', 'ᴐ', 'ʢ', 
          'ʊ', 'l̩', 'θ', 'a', 'ԑ', 'R', '?', 'ê', 'ø', 'sʲ', 'ɫ', 'i˩', 'əː', 'ɔː',
          'lˠ', 'a͜', 'z̩ː', 'ɛː', 'k͡', 'j', 'ʔ', 'œ̃', 'ɪ̯', 'q', 'e', 'dˁ', 'ʁ̥', 'ɑ',
          'ð', '1', 'm', 'ʌ', 'ö', 'ǝ', 'ο', 'B', 'a˩', 'é', 'p', 'ɛ˞', 'ɛʲ', '⊘',
          'ŋ', 'r', 'ʊ̯', 'ɲ', 'k̃', 'Ԑ', 'ɶ', 'ə̃', 'ʁ̃', 'ï', 'x', 'ɞ̃', 'S', 'o͜',
          'ɪ', 'ħ', 'ᵐ', 'ʒ', 'i', 'ɑ̃̃', 'g', 'ʲ', 'ø̃', 'Ʒ', 'Z', 'ỹ', 'ʃ', 'ẽ',
          'c', 'ʊː', 'ɑ́', 'u͜', 'k', 'F', 'I', 'b', 'ɣ', 'z̃', 'ɪː', 'œ', 'ɜ', 'ũ', 'l̥', 'ɛ͂', 'oː', 'ɔ˞', 'ᴇ̃', 'ʁ̌', 'e͜']

_silence = ['__','_']

_punctuation = [',','.','?','!',';']

symbols = ['_pad_', '<GO>', '<END>'] + _phones + _silence + _punctuation + ['UNK']

graphemes = ['_pad_'] + ['L', 'ź', '(', 'H', 'n', 'U', 'û', 'j', 't', 'Ò', 'ö', 'P', 'ə', 'ț', 
             'ʾ', 'ʔ', 'Å', 'N', 'q', '×', 'á', 'β', 'γ', '−', 'ü', 'î', 'È', 'σ',
               'Î', 'r', 'R', 'ḷ', 'Q', 'é', 'D', 'ý', 'ó', "'", 'a', 'ŋ', 'ï', '+',
                 'ɩ', '̓', 'ɂ', 'G', 'b', 'ğ', '7', 'μ', 'º', 'S', 'f', ')', 'ɣ', 'Á',
                   'š', 'Ž', '²', 'z', 'ɛ', '&', 'Ł', 'ʼ', '2', 'ã', '́', 'E', ':', 'ō',
                     'ç', '̃', 'd', '̐', 'O', '°', 'M', 'ṃ', 'K', 'ṭ', '0', 'É', 'e', 'Â',
                       '8', '–', 'ī', 'ġ', 'ò', '9', '.', '′', 'œ', '″', 'å', '̠', '€', '5',
                         '*', 'C', 'W', 'X', 'Ä', 'Ü', '4', 'ê', '-', 'ɔ', 'ǘ', 's', 'è', 'A',
                           'ÿ', 'â', 'č', 'Y', 'F', 'ë', 'v', 'J', 'ä', '͟', 'ñ', 'w', 'Æ', 'x',
                             'ū', 'õ', '’', 'ʻ', 'Œ', '₂', '3', 'm', 'o', 'h', 'ā', 'ʋ', 'g', 'ǀ',
                               'ù', 'ŝ', 'α', 'à', 'l', 'I', 'ú', '=', 'ž', 'y', '/', 'Z', 'Ô', 'ì',
                                 '1', 'ɨ', 'k', '6', 'c', 'i', 'ø', 'ʿ', 'ṣ', 'Ê', 'æ', 'u', 'í', 'V', 
                                 'p', '̱', 'B', 'T', 'ô', '%'] + ['UNK']

def get_phone_idx(phone):
    idx = len(symbols)-1 #unseen phone
    if phone in symbols:
        idx = symbols.index(phone)
    return idx

def get_grapheme_idx(c):
    idx = len(graphemes)-1 #unseen phone
    if c in graphemes:
        idx = graphemes.index(c)
    return idx

def text2grapheme(text):
    char_idxs = [get_grapheme_idx(c) for c in text]
    return char_idxs

if __name__ == '__main__':
    print(get_graphme_dict())
