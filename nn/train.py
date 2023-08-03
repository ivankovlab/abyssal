import torch as T

def train(model, iterator, optimizer, criterion, device='cuda'):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
                
        output = model(batch['embedding_base'],
                        batch['embedding_mutation'])
        output = output.squeeze()
        loss = criterion(output, batch['ddg'])

        loss.backward()
        optimizer.step()
        
        epoch_loss +=loss.item()
            
    return epoch_loss / (i+1)

def evaluate(model, iterator, criterion, device='cuda'):
    model.eval()
    epoch_loss = 0

    with T.no_grad():
        for i, batch in enumerate(iterator):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch['embedding_base'],
                            batch['embedding_mutation'])
            loss = criterion(output.squeeze(), batch['ddg'])

            epoch_loss += loss.item()
        return epoch_loss / (i+1)
