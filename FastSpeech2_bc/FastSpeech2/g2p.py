from text import text_to_sequence
import numpy as np

from text import *

# import ptvsd

# ptvsd.enable_attach(("0.0.0.0",5688))

from transformers import FlaubertTokenizer


def preprocess_french(text):
    # lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    from phonemizer import phonemize
    from phonemizer.separator import Separator
    from fr_utils import _phones, _punctuation

    phn = phonemize(
        text,
        language='fr-fr',
        separator=Separator(phone='#', word="@", syllable=None),
        strip=False,
        preserve_punctuation=True)

    phns = phn.replace('#', ' ').replace('@',' _ ').split()
    
    print(phns)
    phones = ""
    for p in phns:
        if p in _phones:
            phones += p + " "
        elif p in _punctuation:
            phones += 'sil '
        else:
            phones += 'sp '

    phones = '{' + phones[:-1] + '}'
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, []
        )
    )

    return np.array(sequence)

def get_batch_embedding(model, tokenizer,sentence):
    tokenized_text = tokenizer._tokenize(sentence)
    print(tokenizer._tokenize(sentence))
    token_ids = torch.tensor([tokenizer.encode(sentence)])
    print(token_ids)

    last_layer = model(token_ids)[0][:,1:-1,:]
    per_word_embedding = []
    temp = []
    curr = 0
    # per_word_embedding.append(last_layer[:,0:1,:])#Go frame
    for i, token_text in enumerate(tokenized_text):
        temp.append(last_layer[:,i,:])
        if '</w>' in token_text:
            per_word_embedding.append(torch.mean(torch.stack(temp,dim=1),dim=1))
            temp.clear()
    # per_word_embedding.append(last_layer[:,-1:,:])#End frame

    embeddings = torch.stack(per_word_embedding, dim =1)
    print(embeddings.shape)
    return embeddings

if __name__ == '__main__':
    # ptvsd.wait_for_attach()
    text = ',vous les aimerez.'
    
    # preprocess_french(text)

    # from sentence_transformers import SentenceTransformer

    # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    # embeddings = model.encode([text])
    # print(embeddings.shape)

    from transformers import AutoTokenizer, AutoModel 
    import torch
    from transformers import FlaubertModel, FlaubertTokenizer
    from speechbrain.wordemb.transformer import TransformerWordEmbeddings
    modelname = "flaubert/flaubert_base_uncased" 

    # Load pretrained model and tokenizer
    # flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
    flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=True)
    # do_lowercase=False if using cased models, True if using uncased ones

    
    sentence = "L'aéronef"

    print(flaubert_tokenizer._tokenize(sentence))

    # tokenized_text = flaubert_tokenizer._tokenize(sentence)
    # print(flaubert_tokenizer._tokenize(sentence))
    # token_ids = torch.tensor([flaubert_tokenizer.encode(sentence)])
    # print(token_ids)

    # last_layer = flaubert(token_ids)[0][:,1:-1,:]
    # per_word_embedding = []
    # temp = []
    # curr = 0
    # # per_word_embedding.append(last_layer[:,0:1,:])#Go frame
    # for i, token_text in enumerate(tokenized_text):
    #     temp.append(last_layer[:,i,:])
    #     if '</w>' in token_text:
    #         per_word_embedding.append(torch.mean(torch.stack(temp,dim=1),dim=1))
    #         temp.clear()
    # # per_word_embedding.append(last_layer[:,-1:,:])#End frame

    # embeddings = torch.stack(per_word_embedding, dim =1)
    # print(embeddings.shape)
    # print(get_batch_embedding(flaubert, flaubert_tokenizer, sentence).shape)
    
    # torch.Size([1, 8, 768])  -> (batch size x number of tokens x embedding dimension)

