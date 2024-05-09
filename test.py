import torch
import torch.nn as nn

source = torch.tensor([[0.8, 0.08, 0.11], [0.07, 0.85, 0.07], [0.15, 0.17, 0.7]]) # 3x3 matrix
target = torch.tensor([0, 1, 2]).view(-1, 1) # 3x1 matrix

print(source)
print(target)
result = source.gather(1, target)
print(result)

# manual calculus of the entropy
manual_loss = torch.mean(-torch.log(result))
print(f"manual loss of {manual_loss.item()}")

source2 = torch.tensor([[0.53,2.5,0.1268],[0.613,-1.145,-1.234]])
print(source2)

manual_softmax = source2.exp()/source2.exp().sum(dim=1).view(-1,1)
print(source2.exp()) # exponentiel de chaque élément du tensor
print(source2.exp().sum(dim=1)) # somme de tous les exposants

print(manual_softmax)

true_softmax = source2.softmax(1)
print(true_softmax)

# how embedding works :

token_embedding_table = nn.Embedding(10,10)
source3 = torch.tensor([[1,9,3],[6,1,9],[1,9,4]])
logits = token_embedding_table(source3)
print("------")
print("logits")
print(logits)
print("logits size :")
print(logits.size())

