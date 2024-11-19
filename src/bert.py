import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline

from src.data_split import x_train, x_dev, x_test, y_train, y_dev, y_test

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
train_encodings = tokenizer(x_train, truncation=True, padding=True)
val_encodings = tokenizer(x_dev, truncation=True, padding=True)
test_encodings = tokenizer(x_test, truncation=True, padding=True)

train_dataset = IMDbDataset(train_encodings, y_train)
val_dataset = IMDbDataset(val_encodings, y_dev)
test_dataset = IMDbDataset(test_encodings, y_test)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
)

# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained('./model')

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

# trainer.train()
# trainer.save_model("./model")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device = 0)

for document in x_test:
    print(classifier(document))