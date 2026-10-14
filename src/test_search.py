
from semantic_search import search

questions = [
    "What is agroecology?",
    "How does agroecology combine traditional and industrial agriculture?",
    "What are the benefits of crop diversification?",
    "How can farmers improve soil fertility?",
    "What challenges can farmers face when adopting agroecological practices?",
]

for question in questions:
    search(question, top_k=3)
