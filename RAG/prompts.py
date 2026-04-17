from langchain.prompts import PromptTemplate


def basic_prompt():
    # Define our prompt
    prompt_template = """You are a medical AI assistant for answering patient questions about vision correction through refractive surgery.
    You are given the following extracted parts of long documents and a question. Provide a conversational answer. If the question is vague, assume it is regarding eye surgery.
    Use the context as a source of information, but be sure to answer the question directly with as much statistical information as possible. 
    You're job is to provide the user a helpful summary of the information in the context if it applies to the question.
    If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.

    Question: {question}
    =========
    {context}
    =========
    Answer in Markdown:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
