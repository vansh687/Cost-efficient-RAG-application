from typing import List, Dict, Any, Tuple
import tiktoken
from app.config import settings

class LLMGenerator:
    def __init__(self, provider: str = None):
        self.provider = provider or settings.LLM_PROVIDER
        
        # Setup Gemini
        if self.provider == "gemini":
            import google.generativeai as genai
            api_key = settings.GEMINI_API_KEY
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set in environment variables")
            genai.configure(api_key=api_key)
            self.model_name = "gemini-1.5-flash"
            self.client = genai.GenerativeModel(self.model_name)
        # Setup OpenAI
        elif self.provider == "openai":
            from openai import OpenAI
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment variables")
            self.client = OpenAI(api_key=api_key)
            self.model_name = "gpt-3.5-turbo"
        # Setup Mockup
        else:
            self.provider = "mockup"
            self.model_name = "mockup-llm-model"

    def get_token_count(self, text: str) -> int:
        """Returns rough token count using tiktoken (cl100k_base)."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback character length approximation
            return len(text) // 4

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Formulates the prompt, instructs LLM to answer using context only, 
        and extracts/returns citations and token metrics.
        """
        if not context_chunks:
            return "No relevant information found", [], {"prompt_tokens": 0, "completion_tokens": 0}
            
        # Build Context String with IDs for citation referencing
        context_str = ""
        for idx, chunk in enumerate(context_chunks):
            source_info = f"Source: {chunk['metadata'].get('source', 'Unknown')}"
            if "page" in chunk["metadata"]:
                source_info += f", Page: {chunk['metadata']['page']}"
            context_str += f"--- [Doc ID: {idx + 1}] ---\n{source_info}\nContent: {chunk['text']}\n\n"
            
        system_prompt = (
            "You are a helpful assistant. You must answer the user's question based strictly on the provided context.\n"
            "If the provided context does not contain sufficient information to answer the question, output exactly: 'No relevant information found'.\n"
            "Do not make assumptions, expand, or use outside knowledge.\n"
            "Provide inline citations referencing the [Doc ID] e.g., 'This is a fact [Doc ID: 1].'"
        )
        
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        prompt_tokens = self.get_token_count(full_prompt)
        
        if self.provider == "gemini":
            response = self.client.generate_content(
                f"{system_prompt}\n\n{user_prompt}"
            )
            answer_text = response.text.strip()
            completion_tokens = self.get_token_count(answer_text)
            
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            answer_text = response.choices[0].message.content.strip()
            completion_tokens = self.get_token_count(answer_text)
            
        else:
            # Mockup answer formulation
            answer_text = f"Based on the provided document, here is the answer to your query: '{query}'. [Doc ID: 1]."
            completion_tokens = self.get_token_count(answer_text)
            
        # Extract matching citations based on mentions of [Doc ID: X] in the answer text
        citations = []
        for idx, chunk in enumerate(context_chunks):
            citation_marker = f"[Doc ID: {idx + 1}]"
            if citation_marker in answer_text:
                citations.append({
                    "id": chunk["id"],
                    "source": chunk["metadata"].get("source", "Unknown"),
                    "page": chunk["metadata"].get("page"),
                    "snippet": chunk["text"][:150] + "..."
                })
                
        # Handle strict 'No relevant information found' if LLM outputs something else without referencing context
        if "no relevant information found" in answer_text.lower():
            answer_text = "No relevant information found"
            citations = []
            
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
        
        return answer_text, citations, token_usage
