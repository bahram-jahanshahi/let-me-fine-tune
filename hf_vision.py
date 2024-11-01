from transformers import pipeline
import time

# Specify the device as 0 to use the first GPU
vision_classifier = pipeline(model="google/vit-base-patch16-224", device=0)

start_time = time.time()
preds = vision_classifier(images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
end_time = time.time()

# Calculate elapsed time in milliseconds
elapsed_time_ms = (end_time - start_time) * 1000
print(f"Time taken: {elapsed_time_ms:.2f} ms")

# Sort predictions by their scores in descending order
sorted_preds = sorted(preds, key=lambda x: x['score'], reverse=True)

# Print predictions with their scores
for pred in sorted_preds:
    print(f"Label: {pred['label']}, Score: {pred['score']}")