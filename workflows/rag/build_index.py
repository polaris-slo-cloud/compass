import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

print("Loading SQuAD 2.0...")
with open('dev-v2.0.json', 'r') as f:
    squad_data = json.load(f)

passages = []
passage_to_id = {}
questions = []

passage_id = 0

for article in tqdm(squad_data['data'], desc = 'Processing Articles'):
    for paragraph in article['paragraphs']:
        context = paragraph['context']

        if context not in passage_to_id:
            passage_to_id[context] = str(passage_id)
            passages.append({
                'id': str(passage_id),
                'text': context,
            })
            passage_id += 1

        for qa in paragraph['qas']:
            question_text = qa['question']
            answers = []
            if not qa['is_impossible']:
                answers = [ans['text'] for ans in qa['answers']]

            questions.append({
                'id': qa['id'],
                'question': question_text,
                'answers': answers,
                'passage_id': passage_to_id[context],
                'is_impossible': qa['is_impossible'],
            })

print(f"\nExtracted {len(passages)} passages")
print(f"\nExtracted {len(questions)} questions")

print("\nBuilding FAISS index...")
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

passage_texts = [p['text'] for p in passages]
embeddings = encoder.encode(passage_texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

print(f"Index built with {index.ntotal} vectors")

print("\nSaving...")
faiss.write_index(index, 'squad_passages.faiss')

with open('squad_passages.pkl', 'wb') as f:
    pickle.dump(passages, f)

with open('squad_passages_metadata.pkl', 'wb') as f:
    pickle.dump({
        'passages': [p['text'] for p in passages],
        'passage_ids': [p['id'] for p in passages]
    }, f)

with open('squad_questions.json', 'w') as f:
    json.dump(questions, f, indent=2)

print("\nDone!")
print("Files created:")
print("  - squad_passages.faiss")
print("  - squad_passages.pkl")
print("  - squad_questions.json")