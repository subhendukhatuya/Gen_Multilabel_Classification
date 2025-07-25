# -*- coding: utf-8 -*-
"""aiso-testing (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18R_xLvDnDccTr8LzV_VRJt0LeqHeJHq-
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

# Load the pre-trained model

model = SentenceTransformer('all-MiniLM-L6-v2')
# Your 'questions' dictionary
questions = {
    'A': 'Anatomy - Pertains to the structure and organization of living organisms.',
    'B': 'Organisms - Relates to specific living beings, such as animals, plants, and microorganisms.',
    'C': 'Diseases - Encompasses research on various diseases, their causes, symptoms, and treatments.',
    'D': 'Chemicals and Drugs - Involves the study of chemicals, pharmaceuticals, and drugs.',
    'E': 'Analytical, Diagnostic, and Therapeutic Techniques, and Equipment - Covers medical and diagnostic techniques and equipment.',
    'F': 'Psychiatric and Psychology - Addresses topics related to mental health, psychology, and psychiatry.',
    'G': 'Phenomena and Processes - Includes discussions of various phenomena and processes in science and medicine.',
    'H': 'Disciplines and Occupations - Relates to academic disciplines and occupational fields.',
    'I': 'Anthropology, Education, Sociology, and Social Phenomena - Encompasses social sciences, education, and sociology.',
    'J': 'Technology, Industry, and Agriculture - Covers topics related to technology, industry, and agriculture.',
    'L': 'Information Science - Includes articles on information science, data analysis, and related subjects.',
    'M': 'Named Groups - Relates to specific named groups, populations, ethnic groups, and organizations.',
    'N': 'Health Care - Pertains to healthcare, medical practices, and healthcare systems.',
    'Z': 'Geographicals - Addresses geographical locations, regions, and their significance.'
}
'''
questions = {
    'a': 'Anatomy - Pertains to the structure and organization of living organisms.',
    'b': 'Organisms - Relates to specific living beings, such as animals, plants, and microorganisms.',
    'c': 'Diseases - Encompasses research on various diseases, their causes, symptoms, and treatments.',
    'd': 'Chemicals and Drugs - Involves the study of chemicals, pharmaceuticals, and drugs.',
    'e': 'Analytical, Diagnostic, and Therapeutic Techniques, and Equipment - Covers medical and diagnostic techniques and equipment.',
    'f': 'Psychiatric and Psychology - Addresses topics related to mental health, psychology, and psychiatry.',
    'g': 'Phenomena and Processes - Includes discussions of various phenomena and processes in science and medicine.',
    'h': 'Disciplines and Occupations - Relates to academic disciplines and occupational fields.',
    'i': 'Anthropology, Education, Sociology, and Social Phenomena - Encompasses social sciences, education, and sociology.',
    'j': 'Technology, Industry, and Agriculture - Covers topics related to technology, industry, and agriculture.',
    'l': 'Information Science - Includes articles on information science, data analysis, and related subjects.',
    'm': 'Named Groups - Relates to specific named groups, populations, ethnic groups, and organizations.',
    'n': 'Health Care - Pertains to healthcare, medical practices, and healthcare systems.',
    'z': 'Geographicals - Addresses geographical locations, regions, and their significance.'
}

'''





# Generate sentence embeddings for the values in the 'questions' dictionary
question_embeddings = {key: model.encode(value, convert_to_tensor=True, show_progress_bar=False) for key, value in questions.items()}

# Initialize a dictionary to store the top matching labels for each sentence
top_matching_labels = {}

list_labels = []
k = 0
# Read sentences from a text file (replace 'your_text_file.txt' with the actual file path)
with open('./lora_flan_large_prediction_pubmed.txt', 'r') as file:
    current_label = None
    for line in file:
        k = k + 1;
        if(k % 1000 == 0):
            print(f"{k}th iteration")
        line = line.strip()
        temp_list = []
        # Check if "Pred:" appears in the line
        if "Pred:" in line:
            # Split the line at "Pred:" and take the part after it
            current_text = line.split("Pred:", 1)[1].strip()

            # Set the current label
            current_label = current_text
            # Split the line into sentences using full stops (periods) as separators
            sentences = current_label.split('.')
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            #print(sentences)

            # Find the top matching key for each sentence
            for i, sentence in enumerate(sentences):
                #print('sentence', sentence)
                words = sentence.split()

                # Define a set of articles and the word 'no'
                #articles_and_no = set(['a', 'an', 'the', 'no'])

                # Check if all words in the sentence are in the set of articles and 'no'
                #if all(word.lower() in articles_and_no for word in words):
                 #   continue
                #if 'none' not in words and len(words) < 5:
                 #   continue
                similarities = {}
                embedding = model.encode(sentence, convert_to_tensor=True, show_progress_bar=False)
                for key, category_embedding in question_embeddings.items():
                    cos_sim = util.pytorch_cos_sim(embedding, category_embedding)
                    #print('key', cos_sim)
                    similarities[key] = np.mean(cos_sim.cpu().numpy())
                

                #print('Similarities', similarities)
                #x =1/0
                top_matching_key = max(similarities, key=similarities.get)
                #print(top_matching_key)
                temp_list.append(top_matching_key)
                #print(set(temp_list))
        #print(set(temp_list))
        list_labels.append(list(set(temp_list)))
#print(list_labels)
flattened_data = [' '.join(map(str, sublist)) for sublist in list_labels]
#print(flattened_data)
# Convert it to a pandas DataFrame with a single column
df = pd.DataFrame(flattened_data, columns=['Combined_Column'])

# Join the elements within each sublist with space separation
df['Combined_Column'] = df['Combined_Column'].apply(lambda x: ''.join(x))
#df.to_csv('./predicted.csv')
#print(df['Combined_Column'])
#x =1/0


val_aiso = pd.read_csv('test.csv')
val_aiso['Predicted'] = df['Combined_Column']
val_aiso.to_csv('predicted_test_pubmed_flan_t5_t5xxl.csv')


from sklearn.metrics import f1_score, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
ground_truth_labels = val_aiso['labels'].str.split()  # Assuming labels are separated by spaces
predicted_labels = val_aiso['Predicted'].str.split()        # Assuming labels are separated by spaces

#print(predicted_labels) #Initialize the MultiLabelBinarizer to convert labels into binary format
mlb = MultiLabelBinarizer()


# Transform the ground truth and predicted labels into binary format
ground_truth_binary = mlb.fit_transform(ground_truth_labels)
predicted_binary = mlb.transform(predicted_labels)

#print('gt labels', ground_truth_labels)
#print('pred labels', predicted_labels)
#print('gt binary',ground_truth_binary)

#print('pred binary', predicted_binary)

# Calculate the F1 macro score
f1_macro = f1_score(ground_truth_binary, predicted_binary, average='macro')

f1_micro = f1_score(ground_truth_binary, predicted_binary, average='micro')

# Print the F1 macro score
print("F1 Macro Score:", f1_macro)

print("F1 Micro Score:", f1_micro)


from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(ground_truth_binary, predicted_binary)
print('accuracy score', accuracy_score)

weight = f1_score(ground_truth_binary, predicted_binary, average="weighted", zero_division=0)
jacc = jaccard_score(ground_truth_binary, predicted_binary, average="samples", zero_division=0)

print('weighted', weight, 'jacc', jacc)
