import abc
import torch
import numpy as np
import pickle as pkl
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class Featurizer(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def featurize(self):
        pass

class BERTFeaturizer(Featurizer):
    def __init__(self):
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def featurize(self, docs):
        self.model.eval()
        dataset = BERTFeatureDataset(docs, self.tokenizer)
        feature_loader = DataLoader(
            dataset, shuffle=False, batch_size=30, collate_fn=self.__pad_collate)
        doc_embeds = []
        # Predict hidden states features for each layer.
        for input_tok_ids, input_mask, input_seq_ids in feature_loader:
            all_encoder_layers, _ = self.model(
                input_tok_ids, token_type_ids=input_seq_ids, attention_mask=input_mask)
            doc_embeds.append(all_encoder_layers[11][:, 0, :].detach())

        return torch.cat(doc_embeds)

    def __pad_collate(self, batch):
        (token_ids, segment_ids, mask_ids) = zip(*batch)
        token_ids_pad = pad_sequence(token_ids, batch_first=True)
        mask_ids_pad = pad_sequence(mask_ids, batch_first=True)
        segment_ids_pad = pad_sequence(segment_ids, batch_first=True)
        return token_ids_pad, mask_ids_pad, segment_ids_pad

class BERTFeatureDataset(Dataset):
    '''
    Torch dataset class that pads sentences for vector processing by BERT
    '''

    def __init__(self, docs, tokenizer):
        self.docs = docs
        self.tokenizer = tokenizer
        self.all_tokens, self.all_token_ids, self.all_token_masks = self.tokenize_data(
            docs, self.tokenizer)
    
    def __len__(self) :
        return len(self.docs)

    def __getitem__(self, idx) :
        return self.docs[idx]

    def tokenize_data(self, docs, tokenizer):

        all_tokens = []
        all_token_ids = []
        all_token_masks = []

        for doc in docs:
            doc = "[CLS] " + doc + " [SEP]"
            token_vector = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(doc))
            all_tokens.append(torch.tensor(token_vector))
            all_token_ids.append(torch.tensor(
                [0] * len(token_vector)))   # BERT sentence ID
            all_token_masks.append(torch.tensor(
                [1] * len(token_vector)))  # BERT Masking

        return all_tokens, all_token_ids, all_token_masks

class FeaturizerFactory:
    def featurize(self, doc_list, feature_type):
        featurizer = get_featurizer(feature_type)
        return featurizer.featurize(doc_list)
        
def get_featurizer(feature_type):
    if feature_type == 'BERT':
        return BERTFeaturizer()
    else:
        raise Exception("Invalid feature type")