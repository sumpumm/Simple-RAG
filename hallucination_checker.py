from deepeval.models import DeepEvalBaseLLM
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
import ollama

class OllamaDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model="mistral"):
        self.model = model

    def call(self, prompt: str) -> str:
        """Calls the Ollama model and returns a response."""
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

    def generate(self, prompt: str) -> str:
        """Required by DeepEval: Calls the Ollama model synchronously."""
        return self.call(prompt)

    async def a_generate(self, prompt: str) -> str:
        """Required by DeepEval: Async version of generate()."""
        return self.call(prompt)

    def get_model_name(self) -> str:
        """Returns the model name."""
        return self.model

    def load_model(self):
        """DeepEval requires this, but Ollama loads models internally, so we pass."""
        pass

# Initialize the DeepEval-compatible Ollama LLM
ollama_llm = OllamaDeepEvalLLM(model="mistral")


def evaluator(question,context_list,response):
    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        context=context_list
    )
    metric = HallucinationMetric(threshold=0.5,model=ollama_llm)

    print("_______________________________________________")
    print("_____________Hallucibation test________________")
    
    metric.measure(test_case)
    print("Score : ",metric.score)
    print(metric.reason)