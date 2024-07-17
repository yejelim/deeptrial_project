import pdfplumber
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# PDF 텍스트 추출 함수
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# 데이터 전처리 및 토큰화
def preprocess_and_tokenize(texts, labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = []
    for text in texts:
        tokenized_text = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='tf')
        tokenized_texts.append(tokenized_text)
    return tokenized_texts, labels

# BERT 모델 정의
def create_model():
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# PDF 파일 목록 (프로젝트 폴더에 있는 파일)
pdf_paths = ['./example_1.pdf']

# 텍스트 데이터 추출
texts = [extract_text_from_pdf(path) for path in pdf_paths]
labels = [0]  # 예시 라벨 데이터, 실제 데이터에 맞게 변경 필요

# 데이터 전처리 및 토큰화
tokenized_texts, labels = preprocess_and_tokenize(texts, labels)

# 데이터셋 준비
def create_dataset(tokenized_texts, labels):
    input_ids = np.array([token['input_ids'].numpy()[0] for token in tokenized_texts])
    attention_masks = np.array([token['attention_mask'].numpy()[0] for token in tokenized_texts])
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))
    dataset = dataset.shuffle(len(tokenized_texts)).batch(16)
    return dataset

train_dataset = create_dataset(tokenized_texts, labels)

# 모델 생성 및 학습
model = create_model()
model.fit(train_dataset, epochs=3)

# 예측
def predict(model, tokenized_texts):
    predictions = []
    for tokenized_text in tokenized_texts:
        prediction = model(tokenized_text)
        predictions.append(tf.nn.softmax(prediction.logits, axis=-1).numpy())
    return predictions

predictions = predict(model, tokenized_texts)
print(predictions)
