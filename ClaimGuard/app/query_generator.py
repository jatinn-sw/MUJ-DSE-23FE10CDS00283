import json
from typing import List

from app.config import get_config
from app.utils import load_prompt, safe_json_parse, get_llm_client
from app.claim_extractor import Claim


class QueryGenerator:
    def __init__(self):
        self.config = get_config()
        try:
            self.client = get_llm_client()
        except Exception:
            self.client = None

    def generate(self, claim: Claim, context: str = "") -> List[str]:
        prompt = load_prompt("query_generation")
        
        if not self.client or not hasattr(self.client, 'chat'):
            return self._fallback_queries(claim)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Claim: {claim.text}\nCategory: {claim.category}\nContext: {context}"}
                ],
                response_format={"type": "json_object"}
            )
            
            result = safe_json_parse(response.choices[0].message.content)
            if isinstance(result, list):
                return result
            return result.get("queries", [])
        except Exception as e:
            print(f"Query generation failed: {e}")
            return self._fallback_queries(claim)

    def _fallback_queries(self, claim: Claim) -> List[str]:
        words = claim.text.split()
        key_terms = [w for w in words if len(w) > 4 and w.isalpha()][:5]
        base = " ".join(key_terms)
        
        return [
            base,
            f"{base} research",
            f"{base} study",
            f"{base} evidence",
            f"{base} analysis",
        ][:5]


def generate_queries(claim: Claim, context: str = "") -> List[str]:
    generator = QueryGenerator()
    return generator.generate(claim, context)