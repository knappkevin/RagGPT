# Imports
from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
from pdfminer.high_level import extract_text
import base64
from io import BytesIO
import os
import concurrent
from tqdm import tqdm
from openai import OpenAI
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from rich import print
from ast import literal_eval


import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.uic import loadUi

class Design(QMainWindow):
    def __init__(self):
        super(Design, self).__init__()
        loadUi("Design.ui", self)
        self.BrowseButton.clicked.connect(self.browse_file)
        self.ParseButton.clicked.connect(self.parse_pdfs)
        self.StartButton.clicked.connect(self.send_prompts)

    def browse_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Select one or more files', '/home', "PDF files (*.pdf)")
        if files:
            filesString = "|".join(files)
            self.PDFPath.setText(filesString)

    def parse_pdfs(self):
        global client
        OpenAI.api_key = self.APIKey.text()
        client = OpenAI(api_key=OpenAI.api_key)

        # key_path = f'open_api_key.txt'
        # with open(key_path, 'r') as f:
        #     OpenAI.api_key = f.read().strip()
        # client = OpenAI(api_key=OpenAI.api_key)


        docs = []
        files = self.PDFPath.text().split("|")

        for file_path in files:
            print(f"Processing file: {file_path}")
            self.parse_pdf(docs, file_path)

        # Saving result to file for later
        # json_path = f"parsed_pdf_docs.json"
        # with open(json_path, 'w') as f:
        #     json.dump(docs, f)

        # Chunking content by page and merging together slides text & description if applicable
        content = []
        for doc in docs:
            # Removing first slide as well
            text = doc['text'].split('\f')[1:]
            description = doc['pages_description']
            description_indexes = []
            for i in range(len(text)):
                slide_content = text[i] + '\n'
                # Trying to find matching slide description
                slide_title = text[i].split('\n')[0]
                for j in range(len(description)):
                    description_title = description[j].split('\n')[0]
                    if slide_title.lower() == description_title.lower():
                        slide_content += description[j].replace(description_title, '')
                        # Keeping track of the descriptions added
                        description_indexes.append(j)
                # Adding the slide content + matching slide description to the content pieces
                content.append(slide_content)
            # Adding the slides descriptions that weren't used
            for j in range(len(description)):
                if j not in description_indexes:
                    content.append(description[j])
        
        # Cleaning up content
        # Removing trailing spaces, additional line breaks, page numbers and references to the content being a slide
        clean_content = []
        for c in content:
            text = c.replace(' \n', '').replace('\n\n', '\n').replace('\n\n\n', '\n').strip()
            text = re.sub(r"(?<=\n)\d{1,2}", "", text)
            text = re.sub(r"\b(?:the|this)\s*slide\s*\w+\b", "", text, flags=re.IGNORECASE)
            clean_content.append(text)


        # Creating the embeddings
        # We'll save to a csv file here for testing purposes but this is where you should load content in your vectorDB.
        global df
        df = pd.DataFrame(clean_content, columns=['content'])
        df.head()
        
        df['embeddings'] = df['content'].apply(lambda x: self.get_embeddings(x))
        df.head()

        # Saving locally for later
        data_path = f"parsed_pdf_docs_with_embeddings.csv"
        df.to_csv(data_path, index=False)

    def get_embeddings(self, text):
        global client
        embeddings = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return embeddings.data[0].embedding
            
    def parse_pdf(self, docs, file_path):
        file_name = os.path.basename(file_path)
        doc = {
            "filename": file_name
        }
        text = extract_text(file_path)
        doc['text'] = text
        imgs = convert_from_path(file_path, poppler_path="./poppler-24.02.0/Library/bin")
        pages_description = []

        print(f"Analyzing pages for doc {file_name}")

        # Concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            
            futures = [
                executor.submit(self.analyze_doc_image, img)
                for img in imgs
            ]

            with tqdm(total=len(imgs)-1) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)

            for f in futures:
                res = f.result()
                pages_description.append(res)

        doc['pages_description'] = pages_description
        docs.append(doc)

    # a parsing function
    def analyze_doc_image(self, img):
        # Converting images to base64 encoded images in a data URI format to use with the ChatCompletions API
        buffer = BytesIO()
        img.save(buffer, format="jpeg")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_uri = f"data:image/jpeg;base64,{base64_image}"

        # analyze_image
        system_prompt_path = f'system_prompt.txt'
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()

        global client
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_uri},
                        },
                    ],
                }
            ],
            max_tokens=300,
            top_p=0.1
        )

        return response.choices[0].message.content

    def send_prompts(self):
        system_prompt = '''
            You will be provided with an input prompt and content as context that can be used to reply to the prompt.

            You will do 2 things:

            1. First, you will internally assess whether the content provided is relevant to reply to the input prompt.

            2a. If that is the case, answer directly using this content. If the content is relevant, use elements found in the content to craft a reply to the input prompt.

            2b. If the content is not relevant, use your own knowledge to reply or say that you don't know how to respond if your knowledge is not sufficient to answer.

            Stay concise with your answer, replying specifically to the input prompt without mentioning additional information provided in the context content.
        '''

        model="gpt-4-turbo"

        def search_content(df, input_text, top_k):
            embedded_value = self.get_embeddings(input_text)
            df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(np.array(x).reshape(1,-1), np.array(embedded_value).reshape(1, -1)))
            res = df.sort_values('similarity', ascending=False).head(top_k)
            return res

        def get_similarity(row):
            similarity_score = row['similarity']
            if isinstance(similarity_score, np.ndarray):
                similarity_score = similarity_score[0][0]
            return similarity_score

        def generate_output(input_prompt, similar_content, threshold = 0.5):

            content = similar_content.iloc[0]['content']

            # Adding more matching content if the similarity is above threshold
            if len(similar_content) > 1:
                for i, row in similar_content.iterrows():
                    similarity_score = get_similarity(row)
                    if similarity_score > threshold:
                        content += f"\n\n{row['content']}"

            prompt = f"INPUT PROMPT:\n{input_prompt}\n-------\nCONTENT:\n{content}"

            global client
            completion = client.chat.completions.create(
                model= model,
                temperature=0.5,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return completion.choices[0].message.content

        # Running the RAG pipeline
        global df
        prompt = self.PromptField.toPlainText()
        
        print(f"[deep_pink4][bold]QUERY:[/bold] {prompt}[/deep_pink4]\n\n")
        matching_content = search_content(df, prompt, 3)
        print(f"[grey37][b]Matching content:[/b][/grey37]\n")
        for i, match in matching_content.iterrows():
            print(f"[grey37][i]Similarity: {get_similarity(match):.2f}[/i][/grey37]")
            print(f"[grey37]{match['content'][:100]}{'...' if len(match['content']) > 100 else ''}[/[grey37]]\n\n")
        reply = generate_output(prompt, matching_content)
        print(f"[turquoise4][b]REPLY:[/b][/turquoise4]\n\n[spring_green4]{reply}[/spring_green4]\n\n--------------\n\n")
        
        self.ResponseField.setText(reply)



app = QApplication(sys.argv)
mainwindow = Design()
mainwindow.show()  
sys.exit(app.exec_())

