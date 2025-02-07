#%%
from sparsify import Sae

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import plotly.express as px
import numpy as np



#%%
sae = Sae.load_from_hub("EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k", hookpoint="layers.10.mlp")



#%%
in_text = "What is the capital of Frace?"
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
inputs = tokenizer(in_text, return_tensors="pt")

with torch.inference_mode():
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    outputs = model(**inputs, output_hidden_states=True)

    latent_acts = [] 
    for hidden_state in outputs.hidden_states:
        latent_acts.append(sae.encode(hidden_state))


del model
del latent_acts
del outputs
torch.cuda.empty_cache()




#%%
# Get the output tokens and their probabilities
logits = outputs.logits[0]  # Shape: [sequence_length, vocab_size]
probs = torch.softmax(logits, dim=-1)
predicted_tokens = torch.argmax(logits, dim=-1)

# Convert tokens to text
output_text = tokenizer.decode(predicted_tokens)
print("Model output:", output_text)

# Get token-by-token probabilities
top_k = 5  # Number of top predictions to show
top_probs, top_tokens = torch.topk(probs, k=top_k, dim=-1)

# Convert to readable format
for position in range(len(predicted_tokens)):
    for prob, token in zip(top_probs[position], top_tokens[position]):
        token_text = tokenizer.decode([token])
        print(f"  {token_text}: {prob:.3f}")




#%%
# Convert latent activations to numpy arrays
layer_acts = latent_acts[10]  # Looking at layer 10 (same as SAE layer)
acts_np = layer_acts.top_acts.detach().cpu().numpy()  # Access the top_acts attribute

# Create heatmap of activations
fig = px.imshow(acts_np[0],  # First batch item
                title="SAE Top-K Latent Activations",
                labels=dict(x="Feature Index", y="Sequence Position"),
                color_continuous_scale="RdBu")
fig.show()

# Plot activation statistics
acts_mean = np.mean(acts_np[0], axis=0)
fig = px.bar(x=range(len(acts_mean)), 
             y=acts_mean,
             title="Mean Activation per Top-K Feature",
             labels=dict(x="Feature Index", y="Mean Activation"))
fig.show()


# Visualize which features were selected
indices_np = layer_acts.top_indices.detach().cpu().numpy()
flat_indices = np.ravel(indices_np[0])
# Optionally, ensure it is of integer type:
flat_indices = flat_indices.astype(np.int64)
selected_features = np.bincount(flat_indices, minlength=sae.num_latents)

fig = px.bar(x=range(len(selected_features)), 
             y=selected_features,
             title="Feature Selection Frequency",
             labels=dict(x="Feature Index", y="Times Selected"))
fig.show()

# %%
del model
del latent_acts
del outputs
torch.cuda.empty_cache()
