import os
from datasets import load_from_disk, concatenate_datasets
from data.processors import get_tokenizer, get_image_processor
from data.dataset import FlickrDataset, CauldronDataset
from data.collator import VQACollator
from models.config import VLMConfig, TrainConfig

# Load configs
cfg = VLMConfig(load_backbone_weights=False)
train_cfg = TrainConfig(dataset_type="cauldron")

# Load tokenizer and image processor
tokenizer = get_tokenizer(cfg.lm.tokenizer, cfg.image_token)
image_processor = get_image_processor(cfg.vit.img_size)

# Load from disk (same logic as train.py)
print(f"Loading dataset from: {train_cfg.dataset_local_path}")

if train_cfg.dataset_type == "flickr":
    raw = load_from_disk(train_cfg.dataset_local_path)
    ds = raw["train"] if "train" in raw else raw
    dataset = FlickrDataset(ds, tokenizer, image_processor, cfg)
else:
    splits = []
    for subset in train_cfg.dataset_subsets:
        subset_path = os.path.join(train_cfg.dataset_local_path, subset)
        if not os.path.exists(subset_path):
            print(f"  [skip] {subset} not found at {subset_path}")
            continue
        print(f"  Loading {subset}...")
        raw = load_from_disk(subset_path)
        ds = raw["train"] if "train" in raw else raw
        splits.append(ds)
    if not splits:
        raise ValueError(f"No subsets found under {train_cfg.dataset_local_path}")
    ds = concatenate_datasets(splits)
    print(f"Concatenated {len(splits)} subsets → {len(ds)} samples")
    dataset = CauldronDataset(ds, tokenizer, image_processor, cfg)

# Fetch one sample
print("Fetching one sample...")
sample = next(iter(dataset))

print(f"input_ids shape:      {sample['input_ids'].shape}")
print(f"attention_mask shape: {sample['attention_mask'].shape}")
print(f"labels shape:         {sample['labels'].shape}")
print(f"pixel_values shape:   {sample['pixel_values'].shape}")

# Test collator
collator = VQACollator(tokenizer, max_length=train_cfg.max_length)
batch = collator([sample])
print(f"\nCollated batch:")
print(f"  input_ids:      {batch['input_ids'].shape}")
print(f"  attention_mask: {batch['attention_mask'].shape}")
print(f"  labels:         {batch['labels'].shape}")
print(f"  pixel_values:   {batch['pixel_values'].shape}")

# Decode to verify text
decoded = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
print(f"\nDecoded text (first 300 chars):")
print(decoded[:300])
print("\nPipeline OK!")