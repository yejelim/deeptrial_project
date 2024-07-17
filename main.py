import requests
from bs4 import BeautifulSoup
import os
import pdfplumber
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import time
import re

# PDF 다운로드 함수
def download_pdf(url, folder_path):
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join(folder_path, url.split('/')[-1])
        with open(file_path, 'wb') as f:
            f.write(response.content)

# 웹 스크래핑 함수; 검색 키워드 선정 필요
def scrape_pdfs(search_query, folder_path, max_pdfs=15, delay=1):
    pdf_count = 0
    search_url = f"https://www.google.com/search?q={search_query}+filetype:pdf"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for link in soup.find_all('a'):
        if pdf_count >= max_pdfs:
            break
        href = link.get('href')
        if href and 'url?q=' in href:
            pdf_url = href.split('url?q=')[1].split('&')[0]
            if pdf_url.endswith('.pdf'):
                download_pdf(pdf_url, folder_path)
                pdf_count += 1
                time.sleep(delay)  # 설정된 딜레이 적용
    print(f"Downloaded {pdf_count} PDF files for query: {search_query}")

# PDF 파일 텍스트 추출 함수
def extract_text_from_pdfs(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            with pdfplumber.open(os.path.join(folder_path, filename)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                if text:
                    texts.append(text)
    return texts

# 한국어 텍스트 전처리 함수
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # 공백 문자 제거
    text = re.sub(r'[^\w\s]', ' ', text)  # 특수 문자 제거
    return text.strip()

# 데이터 전처리 및 토큰화
def preprocess_and_tokenize(texts, labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenized_texts = []
    for text in texts:
        cleaned_text = clean_text(text)  # 텍스트 전처리
        tokenized_text = tokenizer(cleaned_text, padding='max_length', truncation=True, max_length=512, return_tensors='tf')
        tokenized_texts.append(tokenized_text)
    return tokenized_texts, labels

# BERT 모델 정의
def create_model(num_labels):
    model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# 데이터셋 준비
def create_dataset(tokenized_texts, labels):
    input_ids = np.array([token['input_ids'].numpy()[0] for token in tokenized_texts])
    attention_masks = np.array([token['attention_mask'].numpy()[0] for token in tokenized_texts])
    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))
    dataset = dataset.shuffle(len(tokenized_texts)).batch(16)
    return dataset

# PDF 파일 저장할 폴더
folder_path = "./pdfs"
if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(folder_path, exist_ok=True)

# 검색어 목록
search_queries = [
    "심평원 심의사례집",
    "진료비 심사",
    "요양급여 판정"
]

# PDF 파일 수집
for query in search_queries:
    scrape_pdfs(query, folder_path, max_pdfs=15, delay=1)

# 텍스트 데이터 추출
texts = extract_text_from_pdfs(folder_path)
if not texts:
    raise ValueError("텍스트 데이터가 없습니다. PDF 파일을 확인하세요.")
labels = [0] * len(texts)  # 예시 라벨 데이터, 실제 데이터에 맞게 변경 필요

# 데이터 전처리 및 토큰화
tokenized_texts, labels = preprocess_and_tokenize(texts, labels)

# 데이터셋 준비
train_dataset = create_dataset(tokenized_texts, labels)

# 모델 생성 및 학습
model = create_model(num_labels=len(set(labels)))
model.fit(train_dataset, epochs=3)

# 예측 함수
def predict(model, tokenized_texts):
    predictions = []
    for tokenized_text in tokenized_texts:
        prediction = model(tokenized_text)
        predictions.append(tf.nn.softmax(prediction.logits, axis=-1).numpy())
    return predictions

# 예측 수행
predictions = predict(model, tokenized_texts)
print(predictions)
