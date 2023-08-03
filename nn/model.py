import torch as T
from torch import nn
from pytorch_lightning import LightningModule


class AbyssalModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.attention = cfg.model.attention
        self.feature_convolution = nn.Conv1d(in_channels=cfg.model.embeddings_dim, out_channels=cfg.model.embeddings_dim, 
                                             kernel_size=cfg.model.kernel_size, stride=1,
                                             padding=cfg.model.kernel_size // 2)
        self.attention_convolution = nn.Conv1d(in_channels=cfg.model.embeddings_dim, out_channels=cfg.model.embeddings_dim, 
                                               kernel_size=cfg.model.kernel_size, stride=1,
                                               padding=cfg.model.kernel_size // 2)
        
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(cfg.model.conv_dropout)
        
        self.first_layer = nn.Linear(2560, 2048)
        self.bn_first = nn.BatchNorm1d(2048)
        
        self.second_layer = nn.Linear(2048, 1024)
        self.bn_second = nn.BatchNorm1d(1024)

        self.output = nn.Linear(1024,1)     

    def forward(self, base_vector, mutant_vector):
        
        if self.attention:
            b_repr = self.reweighting_inputs(base_vector)
            m_repr = self.reweighting_inputs(mutant_vector)
        else:
            b_repr = base_vector
            m_repr = mutant_vector
        
        # сливаем предобработанные сетью эмбеддинги для оригинальной аминокислоты и мутированной
        # conc имеет размерность [batchsize, 2*embeddings_dim]
        # замечу, что так как вектора конкатенируются в определенном порядке то это должно по идее 
        # помочь с антисимметрией, так как b_repr⊕m_repr и m_repr⊕b_repr разные вектора, но это 
        # надо еще доказать
        
        conc = T.cat((b_repr, m_repr), dim=-1)
        layer1 = self.dropout(self.relu(self.bn_first(self.first_layer(conc))))
        layer2 = self.dropout(self.relu(self.bn_second(self.second_layer(layer1))))
        return self.output(layer2)
    
    def reweighting_inputs(self, input_vector):
        # исходно этот прием называется Light attention
        # см : https://www.biorxiv.org/content/10.1101/2021.04.25.441334v1
        
        # input_vector shape - [batchsize, embeddings_dim]

        # это артефактное преобразование - можно убрать
        input_vector = T.unsqueeze(input_vector, dim=2)

        # одномерная свертка для получения "фичей" - [batchsize, embeddings_dim, 1]
        # нужна для того чтобы у сети была возможность преобразовать входной вектор 
        # в что-то для себя удобное

        o = self.feature_convolution(input_vector)  
        o = self.dropout(o)

        # одномерная свертка для получения весов для ренормализации "фичей" - [batchsize, embeddings_dim, 1]
        attention = self.attention_convolution(input_vector)

        # приводим коэффициенты attention к диапазону [0.. 1] и домнажаем покомпонентно на "фичи"
        # o * self.softmax(attention) - дает размерность [batchsize, embeddings_dim, 1]
        # суммирование нужно для приведения к виду [batchsize, embeddings_dim] - вектора при этом не меняются
        o = T.sum(o * self.softmax(attention), dim=-1) 
            
        return o
    
    
    def configure_optimizers(self):
        return T.optim.AdamW(self.parameters(), lr=5e-4)
    
    
    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch['embedding_base'], batch['embedding_mutation']).squeeze()
        y_true = batch['ddg']
        loss = nn.functional.mse_loss(y_hat, y_true)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch['embedding_base'], batch['embedding_mutation'])
    
#     def test_step(self, batch, batch_idx, dataloader_idx=0):
#         with T.no_grad():
#             return self.forward(batch['embedding_base'], batch['embedding_mutation'])
    
    def predict(self, loader):
        y_pred = []
        y_true = []
        
        for batch_idx, batch in enumerate(loader):
            y_pred.append(self.predict_step(batch, batch_idx).detach().cpu())
        
        return T.cat(y_pred, dim=0)
