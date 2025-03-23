import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(log_dir, adapt_embedding=True):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    # Find latest checkpoint
    files = os.listdir(log_dir)
    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"): ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    if not iters:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    latest_iter = sorted(iters)[-1]

    # Load checkpoint
    checkpoint = torch.load(os.path.join(log_dir, f"step_{latest_iter}.t7"), map_location='cpu')
    state_dict = checkpoint['net']
    
    # Process state dict
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove `module.` prefix
        name = k[7:] if k.startswith('module.') else k
        
        # Remove `encoder.` prefix if present
        if name.startswith('encoder.'):
            name = name[8:]
        
        new_state_dict[name] = v
    
    # Handle embedding adaptation if needed
    if adapt_embedding:
        # Find embedding layer weights in state dict
        embed_keys = [k for k in new_state_dict.keys() if 'embedding' in k.lower() and 'weight' in k.lower()]
        
        for embed_key in embed_keys:
            saved_embed = new_state_dict[embed_key]
            
            try:
                # Get corresponding model parameter
                model_embed = None
                model_key = embed_key
                
                # Get model embedding parameters
                try:
                    model_embed = bert.state_dict()[model_key]
                except KeyError:
                    print(f"Warning: Could not find matching key for {embed_key} in model state dict")
                    continue
                
                # Check for dimension mismatch
                if model_embed is not None and saved_embed.size(0) != model_embed.size(0):
                    original_n_symbols = saved_embed.size(0)
                    current_n_symbols = model_embed.size(0)
                    embed_dim = saved_embed.size(1)
                    
                    print(f"Adapting embedding layer '{embed_key}' from size {original_n_symbols} to {current_n_symbols}")
                    
                    # Create new embedding tensor with expanded size
                    new_embedding = torch.zeros(current_n_symbols, embed_dim, device=saved_embed.device)
                    
                    # Copy original weights to the first part of the new tensor
                    copy_size = min(original_n_symbols, current_n_symbols)
                    new_embedding[:copy_size] = saved_embed[:copy_size]
                    
                    # Initialize new embeddings if the vocabulary is larger
                    if current_n_symbols > original_n_symbols:
                        with torch.no_grad():
                            # Use similar initialization as the original embeddings
                            mean = saved_embed.mean().item()
                            std = saved_embed.std().item()
                            torch.nn.init.normal_(new_embedding[original_n_symbols:], mean=mean, std=std)
                    
                    # Replace in state dict
                    new_state_dict[embed_key] = new_embedding
            except Exception as e:
                print(f"Error adapting embedding layer '{embed_key}': {e}")
    
    # Remove position_ids if present
    if "embeddings.position_ids" in new_state_dict:
        del new_state_dict["embeddings.position_ids"]
    
    # Load state dict
    try:
        missing, unexpected = bert.load_state_dict(new_state_dict, strict=False)
        
        if missing:
            print(f"Warning: Missing keys when loading PLBERT: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys when loading PLBERT: {unexpected}")
            
        print(f"Successfully loaded PLBERT model from checkpoint at iteration {latest_iter}")
        print('--------------------------------')
    except Exception as e:
        print(f"Error loading PLBERT model: {e}")
    
    return bert
