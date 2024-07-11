
# StarCoder Model Fine-Tuning and Evaluation

This project involves fine-tuning the StarCoder model for competitive coding problems like LeetCode and Codeforces and evaluating its performance. The code demonstrates loading datasets, training the model, and generating solutions for given coding prompts.

## Installation

Ensure you have the required libraries by installing them using the following command:

```sh
pip install -r requirements.txt
```

## Google Colab Setup

1. Mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Install additional optional packages, if required:

```sh
!pip install transformers --upgrade
!pip install accelerate
!pip install -i https://pypi.org/simple/ bitsandbytes
```

## Loading and Preparing Data

### Loading Datasets

- The datasets are loaded from JSON files and processed to extract problem statements and solutions.
- Two data formats are handled by `load_format_1` and `load_format_2` functions.

```python
file_path1 = '/content/drive/MyDrive/StarCoder/tigerbot-kaggle-leetcodesolutions-en-2k.json'
problems = load_format_1(file_path1)

file_path2 = '/content/drive/MyDrive/StarCoder/train.json'
new_problems = load_format_2(file_path2)

file_path3 = '/content/drive/MyDrive/StarCoder/evaluation.json'
new_problems1 = load_format_2(file_path3)
```

### Combining Datasets

- Combine problems from multiple sources to form a comprehensive dataset.

```python
all_problems = problems + new_problems + new_problems1
```

## Model Setup and Training

### Tokenizer and Model Initialization

- Load the tokenizer and model from the specified checkpoint.

```python
checkpoint = "bigcode/starcoder2-3b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config).to(device)
```

### Training Loop

- Train the model using the `DataLoader` for efficient batching.
- Use `AdamW` optimizer and a linear learning rate scheduler.

```python
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(data_loader) * num_epochs)

model.train()
for epoch in range(num_epochs):
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch['input_ids'])
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
```

### Saving the Model

- Save the fine-tuned model to Google Drive.

```python
model.save_pretrained('/content/drive/MyDrive/StarCoder')
```

## Model Evaluation

### Loading the Fine-Tuned Model

- Load the fine-tuned model and tokenizer for evaluation.

```python
model_path = '/content/drive/MyDrive/StarCoder'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
```

### Generating Solutions

- Define a function to generate text based on a given prompt.

```python
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)[0]

    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return generated_text

prompt = "Define a function in Python that calculates the sum of two numbers:"
generated_solution = generate_text(prompt)

print("Generated Solution:")
print(generated_solution)
```

## Explanation of the Content

### Installation

The `requirements.txt` file lists all the necessary Python packages required for this project. Running the command `pip install -r requirements.txt` installs these packages.

### Loading and Preparing Data

Save the .JSON files in your Google Drive account. Copy the path of the saved files and paste them appropriately, in the locations.

### Model Evaluation

The evaluation section shows how to load the fine-tuned model and generate solutions for given prompts. The `generate_text` function takes a prompt as input and uses the model to generate corresponding text.

### Conclusion

The project demonstrates a comprehensive pipeline for fine-tuning the StarCoder model on custom datasets, making it capable of generating solutions for coding problems. The setup is optimized for use in Google Colab with GPU acceleration.
