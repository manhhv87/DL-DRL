import torch
import json
from problems.op.problem_op import OP
from nets.attention_model import AttentionModel
from utils.functions import _load_model_file

## trained model cannot load in other file with different structure
## use it to change the model

load_model = torch.load('./outputs/op_20/run_20211001T224409/epoch-99.pt', map_location=lambda storage, loc: storage)
with open('./outputs/op_20/run_20211001T224409/args.json', 'r') as f:
    args = json.load(f)
model = AttentionModel(
        args['embedding_dim'],
        args['hidden_dim'],
        OP,
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None)
    )
model.load_state_dict({**model.state_dict(), **load_model.get('model', {})})
model, *_ = _load_model_file('./outputs/op_20/run_20211001T224409/epoch-99.pt', model)
torch.save(model.state_dict(), 'epoch-99.pt')
print(1)
