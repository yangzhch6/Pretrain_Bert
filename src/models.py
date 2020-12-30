import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from transformers import BertTokenizer, BertConfig, BertModel


def _gelu_python(x):
    """Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    

def ACT2FN(hidden_act):
    if hidden_act == "gelu":
        if torch.__version__ < "1.4.0":
            gelu = _gelu_python
        else:
            gelu = F.gelu
        return gelu
    else:
        print("activation function wrong!!!")
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert_path):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.cls = BertOnlyMLMHead(self.config)

        # weight = torch.tensor([1]*self.config.vocab_size, dtype=torch.float)
        # weight[1] = 60
        # self.loss_fct = nn.CrossEntropyLoss(weight=weight)  # -100 index = padding token
        self.loss_fct = nn.CrossEntropyLoss() 

    def forward(self, input_ids, token_type_ids, attention_mask, lm_labels):
        x, _ = self.bert(input_ids, token_type_ids, attention_mask)
        scores = self.cls(x)
        masked_lm_loss = self.loss_fct(scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
        return masked_lm_loss   
    
    def savebert(self, save_path):
        torch.save(self.bert.state_dict(), save_path)


# class MaskedLanguageModel(nn.Module):
#     """
#     predicting origin token from masked input sequence
#     n-class classification problem, n-class = vocab_size
#     """

#     def __init__(self, hidden, vocab_size):
#         """
#         :param hidden: output size of BERT model
#         :param vocab_size: total vocab size
#         """
#         super().__init__()
#         self.linear = nn.Linear(hidden, vocab_size)
#         self.softmax = nn.LogSoftmax(dim=-1)

#     def forward(self, x):
#         return self.softmax(self.linear(x))