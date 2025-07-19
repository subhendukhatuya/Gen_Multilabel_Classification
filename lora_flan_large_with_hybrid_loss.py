import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, LoraConfig, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os, copy
import pandas as pd
from tqdm import tqdm

class FlanT5_AllMiniLMv6(nn.Module):
    def __init__(self, t5_model_name='google/flan-t5-large', embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super(FlanT5_AllMiniLMv6, self).__init__()

        # Load the Flan-T5 model and tokenizer
        # self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name, cache_dir="/NS/ssdecl/work")
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name, cache_dir='/NS/ssdecl/work')
        # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, cache_dir="/NS/ssdecl/work")
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name, cache_dir='/NS/ssdecl/work')

        lora_config = LoraConfig(r=2, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
        self.t5_model = get_peft_model(self.t5_model, lora_config)
        # Load the all-MiniLMv6 model for embedding generation
        self.embedding_model = SentenceTransformer(embedding_model_name, cache_folder="/NS/ssdecl/work")

        # Hyperparameters for the loss function balance
        # self.alpha = 0.5  # Weight for generation loss
        # self.beta = 0.5   # Weight for embedding loss
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, input_ids, target_ids):
        device = next(self.parameters()).device  # Get the current device of the model

        # Generate descriptions using Flan-T5
        outputs = self.t5_model(input_ids=input_ids, labels=target_ids)
        gen_loss = outputs.loss

        # Get the logits (not decoded text) and process the token outputs directly
        generated_ids = outputs.logits.argmax(dim=-1).to(device)  # Move to the correct device

        # Decode the generated token IDs into text for embedding loss calculation
        generated_ids = torch.clamp(generated_ids, min=0, max=self.tokenizer.vocab_size - 1)
        generated_descriptions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # print(generated_descriptions)

        # Compute embedding loss
        target_descriptions = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
        emb_loss = self.embedding_loss(generated_descriptions, target_descriptions)

        # Total loss
        # alpha = torch.clamp(self.alpha, min=0)
        beta = 1 - self.alpha
        total_loss = self.alpha * gen_loss + beta * emb_loss
        return total_loss
    
    def generate(self, input_ids, target_ids):
        """Generate descriptions given the input text."""
        device = next(self.parameters()).device  # Get the current device of the model
        # Generate output using the Flan-T5 model
        output = self.t5_model(input_ids = input_ids, labels = target_ids)

        # Decode the generated tokens into text
        # generated_descriptions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_descriptions = self.tokenizer.batch_decode(torch.argmax(output.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        return generated_descriptions

    def embedding_loss(self, generated_descriptions, target_descriptions):
        # Compute embeddings for both generated and target descriptions
        generated_embeddings = self.embedding_model.encode(generated_descriptions, convert_to_tensor=True)
        target_embeddings = self.embedding_model.encode(target_descriptions, convert_to_tensor=True)

        # Cosine similarity loss (1 - cosine similarity to maximize similarity)
        cos_sim = F.cosine_similarity(generated_embeddings, target_embeddings)
        loss = 1 - cos_sim.mean()
        return loss

class LabelDescriptionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, predefined_label_descriptions):
        self.data = pd.read_csv(csv_file)
        self.data['labels'] = self.data['labels'].fillna("neutral")
        self.input_texts = self.data['input'].tolist()
        self.target_labels_list = [labels.split() for labels in self.data['labels'].tolist()]
        self.tokenizer = tokenizer
        self.predefined_label_descriptions = predefined_label_descriptions

        # Tokenize inputs and targets
        self.tokenized_inputs = self.tokenizer(self.input_texts, padding=True, max_length = 210, truncation=True, return_tensors='pt')
        self.target = ["".join([self.predefined_label_descriptions[label] for label in labels]) for labels in self.target_labels_list]
        self.tokenized_targets = self.tokenizer(self.target,
            padding=True,
            max_length = 150, 
            truncation=True,
            return_tensors='pt'
        )
        self.tokenized_targets[self.tokenized_targets== tokenizer.pad_token_id] = -100


    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        return self.tokenized_inputs['input_ids'][idx], self.tokenized_targets['input_ids'][idx]

def validate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0

    with torch.no_grad():  # Disable gradient computation for validation
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss = model(input_ids, target_ids)
            total_val_loss += loss.item()

    return total_val_loss / len(dataloader)

def setup(rank, world_size):
    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()

def save_checkpoint(model, optimizer, epoch, loss, rank, filename='checkpoint.pth'):
    if rank == 0:  # Save only from the main process
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, filename)
        torch.save(model.module.embedding_model, "embedding_model_len_50.pth")

def train_model(rank, world_size, model, optimizer, train_loader, val_loader, epochs=5):
    device = f"cuda:{rank}"
    model.to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    best_val_loss = float('inf')
    best_val_model = copy.deepcopy(model)
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for input_ids, target_ids in tqdm(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()

            # Forward pass
            loss = model(input_ids, target_ids)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # print(f"Rank {rank}, Epoch {epoch + 1}, Loss: {loss.item()}", flush=True)

        # Compute validation loss
        val_loss = validate_model(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, total_loss / len(train_loader), rank, filename='best_checkpoint_len_50.pth')
            best_val_model = copy.deepcopy(model)
        if rank == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader)}, Validation Loss: {val_loss}")
    return best_val_model
def main():
    world_size = 1
    batch_size = 16  # Define the batch size

    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large', cache_dir="/NS/ssdecl/work")
    
    #Label descriptions will be chnaged based on the dataset

    predefined_label_descriptions = {
    'cmp-lg': 'Computational Linguistics - This field involves the use of computer algorithms and models to understand and process human language.',
    'cond-mat.dis-nn': 'Condensed Matter - Disordered Systems and Neural Networks - This subfield of condensed matter physics focuses on disordered materials and their properties, as well as neural networks.',
    'cond-mat.stat-mech': 'Condensed Matter - Statistical Mechanics - Statistical mechanics studies the behavior of a large number of particles in condensed matter systems.',
    'cs.ai': 'Computer Science - Artificial Intelligence - This field focuses on developing algorithms and systems that can perform tasks typically requiring human intelligence, such as learning and problem-solving.',
    'cs.cc': 'Computer Science - Computational Complexity - Computational complexity theory studies the resources required to solve computational problems.',
    'cs.ce': 'Computer Science - Computational Engineering - This field applies computational methods to engineering problems.',
    'cs.cg': 'Computer Science - Computer Graphics - Computer graphics deals with the creation, manipulation, and rendering of visual images using computers.',
    'cs.cl': 'Computer Science - Computation and Language - This subfield explores the intersection of computation and natural language.',
    'cs.cr': 'Computer Science - Cryptography and Security - Cryptography focuses on secure communication and data protection.',
    'cs.cv': 'Computer Science - Computer Vision - Computer vision is concerned with teaching computers to interpret visual information from the world.',
    'cs.cy': 'Computer Science - Cybersecurity - Cybersecurity focuses on protecting computer systems and networks from attacks and unauthorized access.',
    'cs.db': 'Computer Science - Databases - This field deals with the design, storage, and retrieval of data in databases.',
    'cs.dc': 'Computer Science - Distributed, Parallel, and Cluster Computing - This area explores the use of multiple computers to solve complex problems.',
    'cs.dl': 'Computer Science - Digital Libraries - Digital libraries involve the organization and access to large collections of digital information.',
    'cs.dm': 'Computer Science - Discrete Mathematics - Discrete mathematics deals with countable, distinct, and separate objects and structures.',
    'cs.ds': 'Computer Science - Data Structures - Data structures are fundamental components for organizing and storing data efficiently.',
    'cs.fl': 'Computer Science - Formal Languages and Automata Theory - This field studies the mathematical properties of formal languages and automata.',
    'cs.gt': 'Computer Science - General Literature - General literature might refer to publications related to computer science that don\'t fit into specific subfields.',
    'cs.hc': 'Computer Science - Human-Computer Interaction - HCI focuses on improving the interaction between humans and computers.',
    'cs.ir': 'Computer Science - Information Retrieval - Information retrieval involves the retrieval of relevant information from large datasets.',
    'cs.it': 'Computer Science - Information Theory - Information theory studies the quantification of information.',
    'cs.lg': 'Computer Science - Machine Learning - Machine learning is the study of algorithms and statistical models that enable computers to improve their performance on a specific task through experience.',
    'cs.lo': 'Computer Science - Logic in Computer Science - Logic is used in computer science for reasoning and problem-solving.',
    'cs.ma': 'Computer Science - Multiagent Systems - Multiagent systems involve multiple agents (computational entities) interacting in a shared environment.',
    'cs.mm': 'Computer Science - Multimedia - Multimedia involves the integration of various forms of media, such as text, audio, and video.',
    'cs.ms': 'Computer Science - Mathematical Software - This field focuses on the development of software for mathematical applications.',
    'cs.na': 'Computer Science - Numerical Analysis - Numerical analysis deals with numerical approximations of mathematical problems.',
    'cs.ne': 'Computer Science - Neural and Evolutionary Computing - This field explores computational models inspired by neural networks and evolution.',
    'cs.ni': 'Computer Science - Networking and Internet Architecture - Networking focuses on the design and implementation of computer networks.',
    'cs.pf': 'Computer Science - Performance - This field addresses the performance analysis and optimization of computer systems.',
    'cs.pl': 'Computer Science - Programming Languages - Programming languages are formal systems for coding instructions that computers can execute.',
    'cs.ro': 'Computer Science - Robotics - Robotics involves the design, construction, and operation of robots.',
    'cs.sc': 'Computer Science - Symbolic Computation - Symbolic computation involves manipulating mathematical expressions symbolically.',
    'cs.se': 'Computer Science - Software Engineering - Software engineering focuses on systematic approaches to software development.',
    'cs.si': 'Computer Science - Social and Information Networks - This field explores the analysis of social and information networks.',
    'cs.sy': 'Computer Science - Systems and Control - Systems and control engineering involves the control of dynamic systems.',
    'math.co': 'Mathematics - Combinatorics - Combinatorics studies combinations and arrangements of objects.',
    'math.it': 'Mathematics - Information Theory - Information theory in mathematics deals with quantifying information.',
    'math.lo': 'Mathematics - Logic - Logic is the study of reasoning and inference.',
    'math.na': 'Mathematics - Numerical Analysis - Numerical analysis involves approximating solutions to mathematical problems.',
    'math.nt': 'Mathematics - Number Theory - Number theory focuses on the properties and relationships of integers.',
    'math.oc': 'Mathematics - Optimization and Control - Optimization involves finding the best solution among a set of possible solutions.',
    'math.pr': 'Mathematics - Probability - Probability theory deals with uncertainty and randomness.',
    'math.st': 'Mathematics - Statistics - Statistics involves collecting, analyzing, and interpreting data.',
    'nlin.ao': 'Nonlinear Sciences - Adaptation and Self-Organizing Systems - This field explores complex, adaptive, and self-organizing systems.',
    'physics.data-an': 'Physics - Data Analysis, Statistics, and Probability - This area involves statistical and probabilistic methods in data analysis within physics.',
    'physics.soc-ph': 'Physics - Social and Behavioral Physics - This subfield applies physics principles to social and behavioral phenomena.',
    'q-bio.nc': 'Quantitative Biology - Neurons and Cognition - Quantitative biology involves the application of mathematical and computational techniques to biological research.', 'q-bio.qm': 'Quantitative Biology - Quantitative Methods - This subfield focuses on quantitative approaches in biology.',
    'quant-ph': 'Quantum Physics - Quantum Mechanics and Quantum Information - Quantum physics deals with the behavior of particles at the quantum level.',
    'stat.ap': 'Statistics - Applications - This field involves the practical application of statistical methods.',
    'stat.me': 'Statistics - Methodology - Methodology in statistics focuses on the development of statistical techniques.',
    'stat.ml': 'Statistics - Machine Learning - Machine learning is applied to statistical problems and data analysis.',
    'stat.th': 'Statistics - Theory - This subfield involves the theoretical foundations of statistics.'}


    train_dataset = LabelDescriptionDataset('train.csv', tokenizer, predefined_label_descriptions)
    val_dataset = LabelDescriptionDataset('test.csv', tokenizer, predefined_label_descriptions)
    print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '24000'

    torch.multiprocessing.spawn(train_ddp,
                                args=(world_size, train_loader, val_loader, val_dataset),
                                nprocs=world_size,
                                join=True)

def train_ddp(rank, world_size, train_loader, val_loader, val_dataset):
    setup(rank, world_size)

    model = FlanT5_AllMiniLMv6()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    best_val_model = train_model(rank, world_size, model, optimizer, train_loader, val_loader, epochs=5)

    # Assuming `tokenizer`, `model`, `device`, `eval_dataloader`, and `dataset_test` are already defined

    eval_preds = []  # Store predictions

    # Perform inference on validation data
    for input_id, target_id in (tqdm(val_loader)):

        # Run through the FlanT5_AllMiniLMv6 model
        # generated_descriptions = []
        # for input_id, target_id in zip(input_ids, target_ids):
            # Get predictions using the forward pass of FlanT5_AllMiniLMv6
        best_val_model.eval()  # Set model to evaluation mode
        device = f"cuda:{rank}"
        input_id = input_id.to(device)
        target_id = target_id.to(device)
        with torch.no_grad():
            generated_descriptions = best_val_model.module.generate(input_id, target_id)  # Custom method in your FlanT5_AllMiniLMv6 model
            # print(generated_descriptions)
        # Append the decoded descriptions to eval_preds
        eval_preds.extend(generated_descriptions)

    # Calculate accuracy and save predictions to a text file
    correct = 0
    total = 0

    # Open the file to store predictions
    with open('./flanT5_miniL6_validation_output_len_50.txt', 'w') as f1:
        for pred, true in zip(eval_preds, val_dataset.target):  # Assuming true labels are in dataset_test["text_label"]
            # Write the true and predicted labels to the file
            f1.write(f'True: {true} Pred: {pred}\n')

            # Check if the prediction is correct (strip whitespaces for comparison)
            if pred.strip() == true.strip():
                correct += 1
            total += 1

    # Calculate accuracy
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy}% on the evaluation dataset")

    print("---Completed The Task -----")


    cleanup()

if __name__ == "__main__":
    main()
