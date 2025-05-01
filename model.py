import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.3):
        super(NeuralNet, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
        self.hidden_layer2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size // 4)

        self.output_layer = nn.Linear(hidden_size // 4, num_classes)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        self.residual_connection = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.batch_norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        residual = self.residual_connection(out)

        out = self.hidden_layer1(out)
        out = self.batch_norm2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out += residual  # Add residual connection
        
        out = self.hidden_layer2(out)
        out = self.batch_norm3(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.output_layer(out)
        return out
