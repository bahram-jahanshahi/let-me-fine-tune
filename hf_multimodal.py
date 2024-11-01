from transformers import pipeline

# Specify the device as 0 to use the first GPU
vqa = pipeline(model="impira/layoutlm-document-qa", device=0)

output = vqa(
    image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    question="What is the invoice number?",
)

print(output)
print(output[0]["answer"])