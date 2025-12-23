from collections import OrderedDict

import torch
from plm_base import load_model


# POSITION IS IN PDB (1-based idx!!!!!!)
class plmTrunkModel(torch.nn.Module):    
    def __init__(self, 
                 plm_name,
                 hidden_layers=[1024],
                 activation="relu",
                 opmode="mean",
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

        plm_obj = load_model(plm_name)
        plm = plm_obj.get_model()
        plm_tokenizer = plm_obj.get_tokenizer()
        vocab, plm_d_model = plm_obj.get_token_vocab_dim()
        V = len(vocab)

        self.tokenizer = plm_tokenizer
        self.plm = plm.to(device)
        self.last_layer = plm_obj.get_n_layers()
        self.forward_func = plm_obj.get_forward()
        self.specific_pos = specific_pos
        self.opmode = opmode
        self.vocab = vocab

        trunk_d_in_factor = 1

        # self.emb_func = emb_pool_func
        self.epinnet_trunk = EpiNNet(
            d_in=plm_d_model * trunk_d_in_factor,
            d_out=trunk_classes,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_norm=layer_norm,
            use_bias=use_bias,
            activation_on_last_layer=activation_on_last_layer,
            device=device,
            dtype=dtype,
        ).to(device)

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

        emb = torch.nn.functional.normalize(
            hh[:, torch.tensor(self.specific_pos), :], dim=1
        ).mean(dim=1)
        emb = torch.nn.functional.normalize(emb, dim=1)

        return emb, hh, self.epinnet_trunk(emb)

    def forward(self, x):
        return self.final_forward(x)


class EpiNNet(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        hidden_layers=[1024],
        activation="sigmoid",
        layer_norm=True,
        use_bias=True,
        activation_on_last_layer=False,
        device=torch.device("cpu"),
        dtype=torch.double,
    ):
        super().__init__()

        sequence_list = []

        activation_dict = {
            "relu": torch.nn.ReLU(),
            "gelu": torch.nn.GELU(),
            "sigmoid": torch.nn.Sigmoid(),
        }

        if activation not in activation_dict.keys():
            activation = "sigmoid"

        activation_func = activation_dict[activation]

        layers = [d_in] + hidden_layers + [d_out]

        N_layers = len(layers) - 1
        for layer_idx in range(0, N_layers):
            l_in = layers[layer_idx]
            l_out = layers[layer_idx + 1]

            if layer_norm:
                sequence_list += [("l%d_norm" % layer_idx, torch.nn.LayerNorm(l_in))]

            sequence_list += [
                ("l%d_linear" % layer_idx, torch.nn.Linear(l_in, l_out, use_bias))
            ]

            # last layer
            if layer_idx != (N_layers - 1) or activation_on_last_layer:
                sequence_list += [("l%d_activation" % layer_idx, activation_func)]

        self.sequential = torch.nn.Sequential(OrderedDict(sequence_list)).to(device)

    def forward(self, x):
        return self.sequential(x)


class SeqMLP(torch.nn.Module):
    def __init__(
        self,
        encoding_type,
        encoding_size,
        encoding_func,
        plm_name=None,
        hidden_layers=[1024],
        activation="sigmoid",
        opmode="mean",
        layer_norm=True,
        use_bias=True,
        activation_on_last_layer=False,
        tok_dropout=True,
        device=torch.device("cpu"),
        dtype=torch.double,
    ):
        super().__init__()

        possible_encodings = ["onehot", "plm_embedding"]

        if encoding_type not in possible_encodings:
            raise Exception(
                "Unable to support opmode %s for trunk model, allowed opmodes are: %s"
                % (opmode, ", ".join(possible_encodings))
            )

        self.encoding_type = encoding_type
        self.encoding_size = encoding_size

        if encoding_type == "plm_embedding":
            plm_obj = load_model(plm_name)
            vocab, plm_d_model = plm_obj.get_token_vocab_dim()
            V = len(vocab)
            # plm, plm_tokenizer = load_esm2_model_and_alphabet(plm_name)
            # V, plm_d_model = plm.embed_tokens.weight.size()

            self.tokenizer = plm_obj.get_tokenizer()
            self.encoding_func = (
                encoding_func  # Should return just requested positiosn working on
            )

            def encode(seq):
                selected_seq = self.encoding_func(seq)
                return self.tokenizer.encode("".join(selected_seq))

            self.embedding = torch.nn.Embedding(V, plm_d_model)

            def forward(self, x):
                # TODO: Check where did emb come from
                # return self.epinnet_trunk(emb)
                return self.epinnet_trunk(x)
            

            d_in = (
                plm_d_model * self.encoding_size
            )  # should be num of working positions * d_model

        elif encoding_type == "onehot":
            self.encoding_func = encoding_func  # Should return one hot encoding

            def encode(self, seq):
                return self.encoding_fun(seq)

            def forward(x):
                # TODO: Check where did emb come from
                # return self.epinnet_trunk(emb)
                return self.epinnet_trunk(x)

            d_in = self.encoding_size  # Should be overall dimension of onehot

        self.encode_int = encode
        self.epinnet_trunk = EpiNNet(
            d_in=d_in,
            d_out=1,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_norm=layer_norm,
            use_bias=use_bias,
            activation_on_last_layer=activation_on_last_layer,
            device=device,
            dtype=dtype,
        ).to(device)

    def encode(self, *args):
        return self.encode_int(*args)