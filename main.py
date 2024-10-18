from abc import ABC, abstractmethod
from typing import Dict, Any, List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
from AgencyEvaluation import evaluate_agency, interpret_results


class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, max_length: int = 100, num_return_sequences: int = 1) -> List[str]:
        pass


class HuggingFaceModelInterface(LLMInterface):
    def __init__(self, model_name: str):
        self.generator = pipeline('text-generation', model=model_name)

    def generate_response(self, prompt: str, max_length: int = 100, num_return_sequences: int = 1) -> List[str]:
        responses = self.generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        return [response['generated_text'] for response in responses]


class LocalModelInterface(LLMInterface):
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise ValueError(f"Model file path does not exist: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)

    def generate_response(self, prompt: str, max_length: int = 100, num_return_sequences: int = 1) -> List[str]:
        responses = self.generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
        return [response['generated_text'] for response in responses]


class ModelRegistry:
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, model_class: type, init_params: Dict[str, Any] = None):
        cls._registry[name] = {
            'class': model_class,
            'init_params': init_params or {}
        }

    @classmethod
    def create(cls, name: str, **kwargs) -> LLMInterface:
        if name not in cls._registry:
            raise ValueError(f"Unknown model type: {name}")

        model_info = cls._registry[name]
        model_class = model_info['class']
        init_params = {**model_info['init_params'], **kwargs}

        return model_class(**init_params)

    @classmethod
    def list_registered_models(cls):
        return list(cls._registry.keys())


# Register the models: hugging face pipeline
ModelRegistry.register(
    "huggingface_gpt2",
    HuggingFaceModelInterface,
    {"model_name": "gpt2"}
)

# Register the models: hugging face local model
ModelRegistry.register(
    "local_model",
    LocalModelInterface,
    {"model_path": r"C:\Users\samue\OneDrive\Desktop\Trustworthy_LLMs\local_gpt2_model"}
)


def assess_model(model_name: str, prompts: Dict[str, List[str]], max_length: int = 100, num_return_sequences: int = 1):
    model = ModelRegistry.create(model_name)

    for category, category_prompts in prompts.items():
        print(f"\n{'=' * 50}")
        print(f"Testing category: {category}")
        print(f"{'=' * 50}\n")

        for prompt in category_prompts:
            responses = model.generate_response(prompt, max_length=max_length,
                                                num_return_sequences=num_return_sequences)

            print(f"Prompt: {prompt}")
            print(f"{'-' * 30}")

            for i, response in enumerate(responses):
                # Remove the prompt from the response if it's included
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()

                print(f"Response {i + 1}:")
                print(response)
                print(f"{'-' * 30}")

                # Evaluate agency for each response
                agency_results = evaluate_agency(response)
                interpretation = interpret_results(agency_results)

                print("Agency Evaluation:")
                print(interpretation)
                print(f"{'-' * 30}")

            print(f"\n{'=' * 50}\n")


def parse_prompts_file(file_path):
    categories = {}
    current_category = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # If the line is not empty
                if current_category is None:
                    current_category = line
                    categories[current_category] = []
                else:
                    categories[current_category].append(line)
            else:  # If the line is empty, reset the current category
                current_category = None

    return categories


# Example usage
if __name__ == "__main__":
    prompts = parse_prompts_file('agencyPrompts.txt')

    # Using the Hugging Face model
    print("Using Hugging Face model:")
    assess_model("huggingface_gpt2", prompts, max_length=100, num_return_sequences=1)

    print("\n" + "=" * 50 + "\n")

    # Using the local model
    print("Using local model:")
    try:
        assess_model("local_model", prompts, max_length=100, num_return_sequences=1)
    except ValueError as e:
        print(f"Error loading local model: {e}")
        print("Please ensure the local model path is correctly set in the ModelRegistry.register() call.")