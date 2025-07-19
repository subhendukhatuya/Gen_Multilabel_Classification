
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

# Load the pre-trained model

model = SentenceTransformer('all-MiniLM-L6-v2')
# Your 'questions' dictionary

questions = {
    'anger': 'Anger, which can also encompass annoyance and rage, is a powerful emotion that arises when one feels slighted or wronged.',
    'anticipation': 'Anticipation, which also includes feelings of interest and vigilance, is the eager and watchful excitement that precedes an event or outcome.',
    'disgust': 'Disgust, which can involve disinterest, dislike, and even loathing, is the strong aversion or revulsion towards something unpleasant or offensive.',
    'fear': 'Fear, encompassing apprehension, anxiety, and terror, is the primal emotion that surfaces in response to perceived threats or danger.',
    'joy': 'Joy, which can also be experienced as serenity and ecstasy, is the intense happiness and delight that fills our hearts in moments of great positivity.',
    'love': 'Love, including affection, is a deep and affectionate emotional attachment and care for someone or something.',
    'optimism': 'Optimism, along with hopefulness and confidence, is the positive outlook and belief in favorable outcomes even in challenging situations.',
    'pessimism': 'Pessimism, which includes cynicism and a lack of confidence, is the inclination to expect negative or unfavorable outcomes.',
    'sadness': 'Sadness, encompassing pensiveness and grief, is the feeling of sorrow and melancholy often triggered by loss or unfortunate events.',
    'surprise': 'Surprise, which can also encompass distraction and amazement, is the sudden and unexpected reaction to something extraordinary or unforeseen.',
    'trust': 'Trust, including acceptance, liking, and admiration, is the firm belief in the reliability and honesty of someone or something, leading to a sense of comfort and security.',
    'neutral or no emotion': 'In contrast to the other emotions, neutral or no emotion refers to a state of emotional neutrality or absence of any particular emotional response.'
}







# Generate sentence embeddings for the values in the 'questions' dictionary
question_embeddings = {key: model.encode(value, convert_to_tensor=True, show_progress_bar=False) for key, value in questions.items()}

# Initialize a dictionary to store the top matching labels for each sentence
top_matching_labels = {}

list_labels = []
k = 0
# Read sentences from a text file (replace 'your_text_file.txt' with the actual file path)
with open('./lora_flan_large_prediction_semeval.txt', 'r') as file:
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
                articles_and_no = set(['a', 'an', 'the', 'no'])

                # Check if all words in the sentence are in the set of articles and 'no'
                if all(word.lower() in articles_and_no for word in words):
                    continue
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
                temp_list.append(top_matching_key.lower())
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
val_aiso.to_csv('predicted_test_semeval_flan_t5_large_t5xxl.csv')


from sklearn.metrics import f1_score, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
ground_truth_labels = val_aiso['labels'].str.split()  # Assuming labels are separated by spaces
predicted_labels = val_aiso['Predicted'].str.split()        # Assuming labels are separated by spaces

#print(predicted_labels) #Initialize the MultiLabelBinarizer to convert labels into binary format
mlb = MultiLabelBinarizer()


print(ground_truth_labels)
import math

ground_truth_labels_modified = []
predicted_labels_modified = []
count  = 0
for gt_label, pred_label in zip(ground_truth_labels, predicted_labels):
    try:
        if math.isnan(gt_label):
            count = count +1


    except:
        ground_truth_labels_modified.append(gt_label)
        predicted_labels_modified.append(pred_label)

print('Try Count', count)
    

print(len(ground_truth_labels_modified))
    

# Transform the ground truth and predicted labels into binary format
ground_truth_binary = mlb.fit_transform(ground_truth_labels_modified)
predicted_binary = mlb.transform(predicted_labels_modified)

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
