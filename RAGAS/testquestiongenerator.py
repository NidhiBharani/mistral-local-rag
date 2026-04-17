#imports
# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

loader = DirectoryLoader("/home/nidhi/code/mistral-demo/data")
documents = loader.load()

for document in documents:
    document.metadata['file_name'] = document.metadata['source']

# generator with openai models
generator = TestsetGenerator.with_openai()

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

eval_df = testset.to_pandas()
print(eval_df.head())