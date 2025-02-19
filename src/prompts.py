

prompts = {
    "memorize": """You are provided with a long article. Read the article carefully. After reading, you will be asked to perform specific tasks based on the content of the article.

Now, the article begins:
- **Article Content:** {context}

The article ends here.

Next, follow the instructions provided to complete the tasks.""",
    "narrativeqa": "You are given a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "2wikimqa": "Answer the question. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "general_qa": """You are given a question related to the article. Your task is to answer the question directly.

### Question: {input}
### Instructions:
Provide a direct answer to the question based on the article's content. Do not include any additional text beyond the answer.""",
"multifieldqa_en": "Answer the following question based on the above long article, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
"musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",}
