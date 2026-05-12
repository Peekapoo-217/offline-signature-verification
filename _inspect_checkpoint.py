"""Inspect best_model.pth — xem cấu trúc state_dict."""
import torch

ckpt = torch.load("best_model.pth", map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)

lines = []
lines.append(f"Checkpoint keys: {list(ckpt.keys())}")
lines.append(f"Epoch: {ckpt.get('epoch')}")
lines.append(f"Best val loss: {ckpt.get('best_val_loss')}")
lines.append(f"Total params keys: {len(sd)}")
lines.append("")

for k, v in sd.items():
    lines.append(f"{k} -> {list(v.shape)}")

output = "\n".join(lines)
print(output)

with open("_inspect.txt", "w") as f:
    f.write(output)

print("\n--- Written to _inspect.txt ---")
