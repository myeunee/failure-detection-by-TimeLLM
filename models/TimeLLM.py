from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            try:
                self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            except Exception as e:
                print(f"Failed to load config, attempting download: {e}")
                self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b', local_files_only=False)
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except (EnvironmentError, OSError, AttributeError) as e:  # downloads model from HF if not already done
                print(f"Local model files not found ({type(e).__name__}: {e}). Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except (EnvironmentError, OSError, AttributeError) as e:  # downloads the tokenizer from HF if not already done
                print(f"Local tokenizer files not found ({type(e).__name__}: {e}). Attempting to download...")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            try:
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            except Exception as e:
                print(f"Failed to load config, attempting download: {e}")
                self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2', local_files_only=False)

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except (EnvironmentError, OSError, AttributeError) as e:  # downloads model from HF if not already done
                print(f"Local model files not found ({type(e).__name__}: {e}). Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except (EnvironmentError, OSError, AttributeError, TypeError) as e:  # downloads the tokenizer from HF if not already done
                print(f"Local tokenizer files not found ({type(e).__name__}: {e}). Attempting to download...")
                import time
                time.sleep(1)  # 다운로드 완료 대기
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        # override llm dim with actual embedding size to avoid mismatch across models (e.g., GPT2=768, LLaMA=4096)
        self.d_llm = int(self.word_embeddings.shape[1])
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        # Set extra_head early (before using it)
        self.extra_head = getattr(configs, 'extra_head', 'none')

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        # optional extra modules for ablation (green boxes in diagram)
        # Note: self.extra_head is already set above (before mlp_lstm initialization)
        if self.extra_head == 'mlp':
            self.extra_module = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.ReLU(),
                nn.Linear(configs.d_model, configs.d_model)
            )
        elif self.extra_head == 'lstm':
            self.extra_module = nn.LSTM(
                input_size=configs.d_model,
                hidden_size=configs.d_model,
                num_layers=1,
                batch_first=True
            )
        elif self.extra_head == 'mlp_lstm':
            # LSTM: takes patching output (before patch embedding)
            # patch_len is the size of each patch
            self.extra_lstm = nn.LSTM(
                input_size=self.patch_len,
                hidden_size=configs.d_model,
                num_layers=1,
                batch_first=True,
                bidirectional=False,  # 단방향
                dropout=0.0
            )
            # MLP: takes ReprogrammingLayer output + LSTM output (d_llm + d_model)
            mlp_input_dim = self.d_llm + configs.d_model
            self.extra_norm = nn.LayerNorm(mlp_input_dim)
            self.extra_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, self.d_llm),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.d_llm, self.d_llm),
                nn.Dropout(0.2)
            )
            self.extra_module = 'mlp_lstm'
        else:
            self.extra_module = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        # compute prompt embeddings on the same device as LLM embeddings (usually CPU), then move to x_enc device
        emb_device = self.word_embeddings.device
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(emb_device))
        prompt_embeddings = prompt_embeddings.to(x_enc.device)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        source_embeddings = source_embeddings.to(x_enc.device)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # MPS는 bfloat16 미지원이므로 dtype을 동적으로 선택
        desired_dtype = torch.bfloat16 if (x_enc.device.type == 'cuda') else torch.float32
        
        # For mlp_lstm: save patched data before embedding
        lstm_skip = None
        if self.extra_head == 'mlp_lstm':
            # Do patching manually to get raw patches for LSTM
            B_orig, N_orig, T_orig = x_enc.shape
            x_patched = self.patch_embedding.padding_patch_layer(x_enc.to(torch.float32))
            x_patched = x_patched.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            # x_patched shape: (B, N, num_patches, patch_len)
            # Reshape for LSTM: (B*N, num_patches, patch_len)
            x_patched = torch.reshape(x_patched, (x_patched.shape[0] * x_patched.shape[1], x_patched.shape[2], x_patched.shape[3]))
            # Apply LSTM (단방향)
            lstm_skip, _ = self.extra_lstm(x_patched)  # (B*N, num_patches, d_model)
        
        enc_out, n_vars = self.patch_embedding(x_enc.to(desired_dtype))

        # apply optional extra head (non-mlp_lstm cases)
        if self.extra_module is not None and self.extra_head != 'mlp_lstm':
            original_dtype = enc_out.dtype
            enc_out = enc_out.to(torch.float32)
            if self.extra_head == 'lstm':
                enc_out, _ = self.extra_module(enc_out)
            else:
                enc_out = self.extra_module(enc_out)
            enc_out = enc_out.to(original_dtype)
        
        # ReprogrammingLayer
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # For mlp_lstm: combine ReprogrammingLayer output + LSTM output, then apply MLP
        if self.extra_head == 'mlp_lstm' and lstm_skip is not None:
            # enc_out: (B*N, num_patches, d_llm)
            # lstm_skip: (B*N, num_patches, d_model)
            combined = torch.cat([enc_out, lstm_skip], dim=-1)  # (B*N, num_patches, d_llm + d_model)
            combined = self.extra_norm(combined)
            enc_out = self.extra_mlp(combined)  # (B*N, num_patches, d_llm)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        # MPS에서 torch.fft가 미지원이므로 FFT만 CPU로 우회
        original_device = x_enc.device
        if original_device.type == 'mps':
            x_cpu = x_enc.detach().to('cpu', dtype=torch.float32)
            q_fft = torch.fft.rfft(x_cpu.permute(0, 2, 1).contiguous(), dim=-1)
            k_fft = torch.fft.rfft(x_cpu.permute(0, 2, 1).contiguous(), dim=-1)
        else:
            q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
            k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags.to(original_device)


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
