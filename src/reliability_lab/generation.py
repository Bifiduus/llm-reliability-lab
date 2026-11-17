from ollama import Client


class RAGGenerator:
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.client = Client(host=host)

    def generate(self, question: str, chunks: list) -> str:
        context = "\n\n".join(
            f"[Document: {chunk.document}, Page: {chunk.page}]\n{chunk.text}"
            for chunk in chunks
        )

        prompt = f"""Answer the question using only the provided context.

        If the context does not contain enough information to answer,
        respond exactly: I don't know.

        Context:
        {context}

        Question:
        {question}

        Answer:"""

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": 0,
                "seed": 42,
            },
        )

        return response.response
