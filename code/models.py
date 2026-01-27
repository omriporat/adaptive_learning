from collections import OrderedDict

import torch
from plm_base import load_model

# This is kinda deprecated at this point, better using plmEmbeddingModel instead
class plmTrunkModel(torch.nn.Module):    
    def __init__(self, 
                 plm_name,
                 hidden_layers=[1024],
                 activation="relu",
                 opmode="pos",
                 emb_only=True,
                 logits_only=False,
                 layer_norm=True,
                 use_bias=True,
                 activation_on_last_layer=False,
                 tok_dropout=True,
                 specific_pos=None,
                 kernel_size=20,
                 stride=5,
                 trunk_classes=2,
                 device=torch.device("cpu"),                 
                 dtype=torch.double):
        super().__init__()
        
        
        # plm = load_model(plm_name)
        # #plm, plm_tokenizer = load_esm2_model_and_alphabet(plm_name)
        # V, plm_d_model = plm.embed_tokens.weight.size()
        
        plm_obj = load_model(plm_name)
        plm = plm_obj.get_model()
        plm_tokenizer = plm_obj.get_tokenizer()
        vocab, plm_d_model = plm_obj.get_token_vocab_dim()
        V = len(vocab)
        
        self.tokenizer = plm_tokenizer
        self.plm = plm.to(device)
        self.last_layer = plm_obj.get_n_layers()
        self.forward_func = plm_obj.get_forward()
        self.internal_encode = plm_obj.get_encode()
        self.specific_pos = specific_pos
        self.vocab = vocab
        
        #### IMPORTANT -> this is a bunch of legacy code that I got too scared to delete - I will delete it soon

        # if (type(plm) == esm2.model.esm2.ESM2):
        #     self.last_layer = plm_obj.get_n_layers()
            
        # def plm_forward_presentation(x):
        #     forward = self.plm.forward(x, repr_layers=[self.last_layer])
        #     hh = forward["representations"][self.last_layer]
        #     return(hh)
            
        # self.forward_func = plm_forward_presentation                
        # self.opmode = opmode
        
        # possible_opmodes = ["mean", "class", "avgpool", "pos"]
        
        # if opmode not in possible_opmodes:
        #     raise Exception("Unable to support opmode %s for trunk model, allowed opmodes are: %s" % (opmode, ", ".join(possible_opmodes)))
                        
        # if opmode == "mean":            
        #     if specific_pos is not None:
        #         # Average across specific positions
        #         self.specific_pos = torch.tensor(specific_pos, dtype=torch.int64) - 1 # PDB INDEX!!!!!! (1-based)
                
        #         def emb_pool_func(hh):                
        #             return(hh[:,self.specific_pos,:].mean(dim=1))
        #     else:
        #         def emb_pool_func(hh):                
        #             return(hh.mean(dim=1))
            
        # elif opmode == "class":
        #     class_token = torch.tensor(self.tokenizer.encode("<unk>"), dtype=torch.int64)
            
        #     def emb_pool_func(hh):
        #         return(hh[:,0,:])
            
        # elif opmode == "avgpool":
        #     self.conv1d = torch.nn.AvgPool1d(kernel_size=kernel_size,stride=stride)
                
        #     def emb_pool_func(hh):
        #         return(self.conv1d(einops.rearrange(hh,"B S D->B D S")).mean(dim=2))   
        
        # elif opmode == "pos":
        #     self.specific_pos = torch.tensor(specific_pos, dtype=torch.int64) - 1 # PDB INDEX!!!!!! (1-based)
            
        #     def emb_pool_func(hh):
        #         return(hh[:,self.specific_pos,:].flatten(1,2))
            
            
        # trunk_d_in_factor = 1 if opmode != "pos" else len(self.specific_pos)
        trunk_d_in_factor = 1
            
            
        # self.emb_func = emb_pool_func
        self.epinnet_trunk = EpiNNet(d_in=plm_d_model * trunk_d_in_factor,
                                     d_out=trunk_classes,                 
                                     hidden_layers=hidden_layers,
                                     activation=activation,
                                     layer_norm=layer_norm,
                                     use_bias=use_bias,
                                     activation_on_last_layer=activation_on_last_layer,
                                     device=device,                 
                                     dtype=dtype).to(device)

        if emb_only:
            self.final_forward = self._emb_only_forward            
        elif logits_only:
            self.final_forward = self._logits_only_forward 
        else:   
            self.final_forward = self._forward
        
    def encode(self, seq):            
        enc_seq = ""
        if self.opmode == "class":
            enc_seq = "<unk>"
                
        enc_seq = enc_seq + "<cls>" + seq + "<eos>"
                
        return self.tokenizer.encode(enc_seq)
            
    def _logits_only_forward(self, x):
        return self.forward_func(x)[0]

    def _emb_only_forward(self, x):
        return self.forward_func(x)[1]

    def _forward(self, x):                
        hh = self._emb_only_forward(x)

        emb = torch.nn.functional.normalize(hh[:,torch.tensor(self.specific_pos),:], dim=1).mean(dim=1)
        emb = torch.nn.functional.normalize(emb, dim=1)
            
        return emb, hh, self.epinnet_trunk(emb)
    
    def forward(self, x):
        return self.final_forward(x)
      
class plmEmbeddingModel(torch.nn.Module):
    def __init__(self, 
                 plm_name,
                 emb_only=True,
                 logits_only=False,
                 tok_dropout=True,
                 device=torch.device("cpu"),                 
                 dtype=torch.double):
        super().__init__()
        plm_obj = load_model(plm_name)
        plm = plm_obj.get_model()
        plm_tokenizer = plm_obj.get_tokenizer()
        vocab, plm_d_model = plm_obj.get_token_vocab_dim()
        V = len(vocab)

        self.plm_name = plm_name
        self.tokenizer = plm_tokenizer
        self.plm = plm.to(device)
        self.last_layer = plm_obj.get_n_layers()
        self.forward_func = plm_obj.get_forward()
        self.internal_encode = plm_obj.get_encode()
        self.vocab = vocab
        
        if emb_only:
            self.final_forward = self._emb_only_forward            
        elif logits_only:
            self.final_forward = self._logits_only_forward

    def encode(self, seq):
        # enc_seq = ""
        # enc_seq = enc_seq + "<cls>" + seq + "<eos>"
        # return self.tokenizer.encode(enc_seq)
        return self.internal_encode(seq)



    def _logits_only_forward(self, x, **kwargs):
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            return self.forward_func(x, attention_mask=attention_mask)[0]
        else:
            return self.forward_func(x)[0]

    def _emb_only_forward(self, x, **kwargs):
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            return self.forward_func(x, attention_mask=attention_mask)[1]
        else:
            return self.forward_func(x)[1]
    
    def forward(self, x, **kwargs):
        return self.final_forward(x, **kwargs)

class abPlmEmbeddingModel(plmEmbeddingModel):
    def esm_encode(self, seq):
        h_l_split = seq.split("#L#")
        h_seq = h_l_split[0]
        l_seq = h_l_split[1]
        h_seq = h_seq.split("#H#")[1]
        final_seq = h_seq + l_seq
        final_seq = final_seq.replace("-", "<pad>")
        return self.tokenizer.encode(final_seq)

    def encode_heavy_or_light_only(self, seq):
        seq = " ".join([aa for aa in seq])
        final_seq = seq.replace("-", "[PAD]")
        return self.tokenizer.encode(final_seq)

    def encode(self, seq):
        h_l_split = seq.split("#L#")
        h_seq = h_l_split[0]
        l_seq = h_l_split[1]
        h_seq = h_seq.split("#H#")[1]
        h_seq = " ".join([aa for aa in h_seq])
        l_seq = " ".join([aa for aa in l_seq])
        final_seq = (h_seq + " [SEP] " + l_seq).replace("-", "[PAD]")
        return self.tokenizer.encode(final_seq)

class EpiNNet(torch.nn.Module):
    def __init__(self, 
                 d_in,
                 d_out,                 
                 hidden_layers=[1024],
                 activation="sigmoid",
                 layer_norm=True,
                 use_bias=True,
                 activation_on_last_layer=False,
                 device=torch.device("cpu"),                 
                 dtype=torch.double,
                 **kwargs):
        super().__init__()
        
        sequence_list = []
        
        activation_dict = {'relu': torch.nn.ReLU(),
                           'gelu': torch.nn.GELU(),
                           'sigmoid': torch.nn.Sigmoid()}
        
        if activation not in activation_dict.keys():
            activation = 'sigmoid'
            
        activation_func = activation_dict[activation]
        
        layers = [d_in] + hidden_layers + [d_out]
        
        N_layers = len(layers) - 1
        for layer_idx in range(0, N_layers):                        
            l_in = layers[layer_idx]
            l_out = layers[layer_idx + 1]
            
            if layer_norm:
                sequence_list += [('l%d_norm' % layer_idx, torch.nn.LayerNorm(l_in))]
            
            sequence_list += [('l%d_linear' % layer_idx, torch.nn.Linear(l_in, l_out, use_bias))]
            
            # last layer
            if layer_idx != (N_layers - 1) or activation_on_last_layer:            
                    sequence_list += [('l%d_activation' % layer_idx, activation_func)]
            
            
        self.sequential = torch.nn.Sequential(OrderedDict(sequence_list)).to(device)
    
    def forward(self, x):
        return self.sequential(x)
            
class seqMLP(torch.nn.Module):
    def __init__(self, 
                 encoding_type,
                 encoding_size,
                 encoding_func,
                 plm_name=None,                         
                 hidden_layers=[1024],
                 activation="sigmoid",
                 opmode="pos",
                 layer_norm=True,
                 use_bias=True,
                 activation_on_last_layer=False,
                 tok_dropout=True,
                 device=torch.device("cpu"),                 
                 dtype=torch.double):
        super().__init__()
                 
        possible_encodings = ["onehot", "plm_embedding"]
         
        if encoding_type not in possible_encodings:
            raise Exception("Unable to support encoding type %s for trunk model, allowed encoding types are: %s" % (encoding_type, ", ".join(possible_encodings)))
        
        self.encoding_type = encoding_type
        self.encoding_size = encoding_size
        
        
        if encoding_type == "plm_embedding":
            plm_obj = load_model(plm_name)
            vocab, plm_d_model = plm_obj.get_token_vocab_dim()
            V = len(vocab)
            #plm, plm_tokenizer = load_esm2_model_and_alphabet(plm_name)
            #V, plm_d_model = plm.embed_tokens.weight.size()
                    
            self.tokenizer = plm_obj.get_tokenizer()
            self.encoding_func = encoding_func # Should return just requested positiosn working on
            
            def encode(seq):                
                selected_seq = self.encoding_func(seq)
                return self.tokenizer.encode("".join(selected_seq))
            
            
            self.embedding = torch.nn.Embedding(V, plm_d_model)
            
            def forward(self, x):                
                return self.epinnet_trunk(x)
            
            d_in = plm_d_model * self.encoding_size  # should be num of working positions * d_model
           
        elif encoding_type == "onehot":
            self.encoding_func = encoding_func # Should return one hot encoding
            
            def encode(self, seq):
                return self.encoding_fun(seq)
            
            
            def forward(x):                
                return self.epinnet_trunk(x)
            
            d_in = self.encoding_size # Should be overall dimension of onehot
            
        self.encode_int = encode
        self.epinnet_trunk = EpiNNet(d_in=d_in,
                                     d_out=1,                 
                                     hidden_layers=hidden_layers,
                                     activation=activation,
                                     layer_norm=layer_norm,
                                     use_bias=use_bias,
                                     activation_on_last_layer=activation_on_last_layer,
                                     device=device,                 
                                     dtype=dtype).to(device)

    def encode(self, *args):
        return self.encode_int(*args)
    