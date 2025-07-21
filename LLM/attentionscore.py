## simple self attention score
##简单自注意力机制

import torch

inputs = torch.tensor(
    [[0.43,0.15,0.89],
     [0.55,0.87,0.66],
     [0.57,0.85,0.64],
     [0.22,0.58,0.33],
     [0.77,0.25,0.10],
     [0.05,0.80,0.55]
    ]
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])

for i,x_i in enumerate(inputs):
    print(i,x_i)
    attn_scores_2[i] =torch.dot(x_i, query)
print(attn_scores_2) #计算中间注意力分数


attn_weights2= torch.softmax(attn_scores_2, dim = 0)
print("Attention weights:",attn_weights2)


##计算上下文向量

context_vec_2 = torch.zeros(query.shape)

for i,x_i in enumerate(inputs):
    print(i,x_i)
    context_vec_2+= attn_weights2[i] * x_i
print(context_vec_2)


attn_scores = torch.empty(inputs.shape[0],inputs.shape[0])
for i, x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
        attn_scores[i,j]= torch.dot(x_i, x_j)
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim = -1)
print(attn_weights)


all_context_vecs = attn_weights @ inputs
print(all_context_vecs)