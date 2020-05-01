import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, mode):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost
        self._mode = mode

    def forward(self, inputs):
        # convert inputs from BHT -> BTH
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances (BT, num_embeddings)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        if self._mode == 'attn-vqvae':
            encodings = torch.matmul(flat_input, self._embedding.weight.t())
            encodings = F.softmax(encodings,dim=1)
            quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        elif self._mode == 'multi-vqvae':
            RANK = 3
            encoding_sort = torch.argsort(distances,dim=1)
            encodings = torch.zeros(encoding_sort.shape[0], self._num_embeddings).to(device) #2048 x 512
            for k in range(RANK):
                encoding_indices = encoding_sort[:,k].unsqueeze(1)
                encodings.scatter_(1, encoding_indices, 1)
            quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape) / RANK

        elif self._mode == 'vqvae':
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # 2048x1
            a = 27
            b = 36
            encoding_indices_small = encoding_indices[a:b,:]
            # Random
            encoding_indices_small = torch.randint(low=0, high=self._num_embeddings, size=encoding_indices_small.size()).cuda()
            encoding_indices[a:b,:] = encoding_indices_small               

            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device) #2048 x 512
            encodings.scatter_(1, encoding_indices, 1)
            quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))


        
        # convert quantized from BTH -> BHT
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encoding_indices



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        self._vq_vae = VectorQuantizer(num_embeddings=64, embedding_dim=d_v, commitment_cost=0.25, mode='vqvae')
    
    def attention_result(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        loss, v, perplexity, indices = self._vq_vae(v.permute(0,2,1))
        v = v.permute(0,2,1)

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        return output, loss, perplexity, indices, attn

    def forward(self, q, k, v):
        sz_b, len_q, _ = q.size()
        output, loss, perplexity, indices, attn = self.attention_result(q,k,v)
        # output[0,:,:,:] = torch.zeros((sz_b, len_q, d_v))
        # output[1,:,:,:] = torch.zeros((sz_b, len_q, d_v))
        # output[2,:,:,:] = torch.zeros((sz_b, len_q, d_v))
        # output[3,:,:,:] = torch.zeros((sz_b, len_q, d_v))
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output)

        return output, attn, loss, perplexity, indices

    def inference(self, x, x_p):
        sz_b, len_q, _ = x.size()
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        output,loss, perplexity, indices, attn= self.attention_result(x,x,x)
        output_p,_,_,_,_ = self.attention_result(x_p,x_p,x_p)
        # n, b, lq, dv
        # output[0,:,:,:] = output_p[0,:,:,:]
        # output[1,:,:,:] = torch.zeros((sz_b, len_q, d_v))
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output)
        return output, attn, loss, perplexity, indices



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        # attnmax = torch.argmax(attn,dim=2)
        # for i in range(attnmax.size(0)):
        #     print(attnmax[i,:])
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn