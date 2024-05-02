""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_pad_"
# _special = ['«','»', '¬', '~','"','"','(', ')','[', ']', '-']
_punctuation=list(",;.!?")
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# _ipa = ['a', 'ã', 'b', 'd', 'dʒ', 'e', 'ẽ', 'f', 'g', 'h', 'i', 'iː', 'j', 'k', 'l', 'l̩', 'm', 'm̩', 'n', 'n̩', 'o', 'oː', 'p', 'pf', 'pː', 'r', 's', 't', 'ts', 'tʃ', 'u', 'uː', 'v', 'w', 'x', 'y', 'z', 'æ', 'ç', 'ð', 'ø', 'ŋ', 'œ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɔ', 'ɔː', 'ɔ̃', 'ə', 'ɚ', 'ɛ', 'ɜ', 'ɜː', 'ɟ', 'ɡ', 'ɣ', 'ɪ', 'ɫ', 'ɲ', 'ɳ', 'ɹ', 'ɾ', 'ʀ', 'ʃ', 'ʊ', 'ʌ', 'ʎ', 'ʏ', 'ʒ', 'ʔ', 'ʝ', 'ˈa', 'ˈe', 'ˈẽ', 'ˈi', 'ˈiː', 'ˈo', 'ˈoː', 'ˈu', 'ˈuː', 'ˈy', 'ˈæ', 'ˈø', 'ˈœ', 'ˈɐ', 'ˈɑː', 'ˈɒ', 'ˈɔ', 'ˈɔː', 'ˈɔ̃', 'ˈɚ', 'ˈɛ', 'ˈɜ', 'ˈɜː', 'ˈɪ', 'ˈʊ', 'ˈʌ', 'ˈʏ', 'ˌa', 'ˌe', 'ˌi', 'ˌiː', 'ˌn̩', 'ˌo', 'ˌoː', 'ˌu', 'ˌuː', 'ˌæ', 'ˌɐ', 'ˌɑː', 'ˌɒ', 'ˌɔ', 'ˌɔː', 'ˌɔ̃', 'ˌɛ', 'ˌɜ', 'ˌɜː', 'ˌɪ', 'ˌʊ', 'ˌʌ', 'β', 'θ', 'ᵻ']
# _lexicon_phones_dict={
# 'a':'a', 'e':'e', 'e^':'ɛ', 'x':'œ', 'x^':'ø', 
# 'i':'i', 'y':'y', 'u':'u', 'o':'o', 'o^':'ɔ', 
# 'q':'ʒ', 'a~':'ɑ̃', 'e~':'ɛ̃', 'x~':'œ̃', 'o~':'ɔ̃',
# 'h':'ɥ', 'w':'w','j':'j','p':'p', 't':'t', 'k':'k', 
# 'b':'b', 'd':'d', 'g':'ɡ', 'f':'f', 's':'s' , 
# 's^':'ʃ', 'v':'v', 'z':'z', 'z^':'ʒ', 'r':'ʁ',
# 'l':'l', 'm':'m', 'n':'n', 'n~':'ɲ', 'ng':'ŋ'
# }
mfa_french_phoneme = ['a', 'b', 'c', 'd', 'dʒ', 'e', 'f', 'i', 'j', 'k', 'l',
                        'm', 'mʲ', 'n', 'o', 'p', 's', 't', 'ts', 'tʃ' ,'u' ,'v', 'w', 'y' ,
                        'z' ,'ø' ,'ŋ' ,'œ', 'ɑ', 'ɑ̃', 'ɔ', 'ɔ̃', 'ə', 'ɛ' ,'ɛ̃', 'ɟ', 'ɡ', 'ɥ', 'ɲ' ,'ʁ' ,'ʃ', 'ʎ', 'ʒ']
_phones = [ '@' + s for s in mfa_french_phoneme]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ["@" + s for s in cmudict.valid_symbols]
# _pinyin = ["@" + s for s in pinyin.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + _silences
    # + list(_special)
    # + list(_punctuation)
    # + list(_letters)
    # + _arpabet
    # + _pinyin
    + _phones

)
