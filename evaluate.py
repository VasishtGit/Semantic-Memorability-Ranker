from models.memorability_ranker import NeuroDapt

model = NeuroDapt(
    unfreeze_last_n_layers=2,
)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total:,}")
print(f"Trainable Parameters: {trainable:,}")