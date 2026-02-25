import os
import time
from collections import defaultdict

import openai
import torch
# import vertexai
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
# from google.cloud.aiplatform_v1beta1.types.content import SafetySetting
from sentence_transformers import SentenceTransformer, util


# from vertexai.generative_models import GenerationConfig, GenerativeModel, HarmBlockThreshold, HarmCategory

from rllm.globals import GCP_LOCATION, GCP_PROJECT_ID, GEMINI_MODEL, OAI_RM_MODEL

# exit()
def compute_pass_at_k(results):
    import hashlib
    import json

    # Create a map to store correct answers per problem
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    # Count correct answers for each problem
    for trajectory in results:
        task = trajectory.task

        # Generate hash of problem dict/string
        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        is_correct = 1 if trajectory.reward > 0 else 0

        problem_correct_map[problem_hash] += is_correct
        problem_total_map[problem_hash] += 1

    # Calculate pass@1 and pass@16
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems

    print("Total unique problems:", total_problems)
    print("Average Pass@1 Accuracy:", pass_at_1)
    print("Average Pass@k Accuracy:", pass_at_k)


def call_oai_rm_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    model_id: str = OAI_RM_MODEL,
    retry_count: int = int(1e9),
) -> list[str]:
    client = openai.OpenAI()

    backoff = 1
    retry_count = int(retry_count)

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                n=n,
            )
            break
        except Exception as e:
            if "429" in str(e):
                print("Retry due to rate limit: ", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            else:
                print("Exception: ", e)
                return []

    if n == 1:
        content = response.choices[0].message.content
        return [content] if content is not None else []
    return [choice.message.content for choice in response.choices if choice.message.content is not None]


def call_gemini_llm(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    project_id: str = GCP_PROJECT_ID,
    location: str = GCP_LOCATION,
    model_id: str = GEMINI_MODEL,
    retry_count: int = int(1e9),
) -> list[str]:
    """
    Calls a Gemini LLM on Vertex AI to generate n responses at a given temperature.

    Args:
        prompt (str): The text prompt to send to the LLM.
        system_prompt (str): System instruction or system prompt to send to the model.
        n (int): Number of responses to generate.
        temperature (float): Sampling temperature.
        project_id (str): Your GCP project ID.
        location (str): The region to use (e.g., us-central1).
        model_id (str): The specific Gemini model resource name.
        retry_count (int): Number of times to retry on rate-limit errors.

    Returns:
        List[str]: A list of response texts from the Gemini model.
    """
    return []
    # # Initialize the Vertex AI environment
    # vertexai.init(project=project_id, location=location)

    # # Define which harm categories to allow (or set thresholds).
    # HARM_CATEGORIES = [
    #     HarmCategory.HARM_CATEGORY_UNSPECIFIED,
    #     HarmCategory.HARM_CATEGORY_HARASSMENT,
    #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    #     HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    # ]

    # # Instantiate the GenerativeModel
    # model = GenerativeModel(
    #     model_name=model_id,
    #     system_instruction=[system_prompt],
    # )

    # # Add an exponential backoff for rate limit errors
    # backoff = 1
    # retry_count = int(retry_count)
    # generation_config = GenerationConfig(
    #     temperature=temperature,
    #     candidate_count=n,
    # )

    # for attempt in range(retry_count):
    #     try:
    #         # Request multiple candidates by specifying n (candidate_count)
    #         response = model.generate_content([prompt], generation_config=generation_config, safety_settings=[SafetySetting(category=h, threshold=HarmBlockThreshold.BLOCK_NONE) for h in HARM_CATEGORIES])
    #         # Once successful, break out of the retry loop
    #         break
    #     except Exception as e:
    #         # Retry if there's a rate-limit error (HTTP 429)
    #         if "429" in str(e):
    #             print("Retry due to rate limit: ", e)
    #             time.sleep(backoff)
    #             backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
    #             continue
    #         elif "403" in str(e):
    #             print("NO ACCESS TO ENDPOINT", e)
    #             raise NotImplementedError from None
    #         else:
    #             print("Exception: ", e)
    #             return []  # or raise an exception if desired

    # # Collect the texts from all returned candidates
    # # Depending on the library version, this might need to be adjusted
    # # if the `response` shape is different

    # try:
    #     # Keep this to check for errors in indexing.
    #     [candidate.text for candidate in response.candidates]
    #     if len(response.candidates) == 1:
    #         return response.candidates[0].text
    #     return [candidate.text for candidate in response.candidates]
    # except Exception as e:
    #     print("Error extracting text from response:", e)
    #     return []


class RAG:
    def __init__(self, docs: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            docs (List[str]): A list of documents to encode.
            model (str): The SentenceTransformer model to use.
        """
        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model)
        self.docs = docs
        # Compute embeddings
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)

    def top_k(self, query, k=1):
        # Create embedding for the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity [1 x N]
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        # Extract top_k indices
        top_results = torch.topk(cos_scores, k=k)

        # Prepare a list of (score, problem_text)
        results = []
        for score, idx in zip(top_results.values, top_results.indices, strict=False):
            results.append(
                {
                    "score": score,
                    "text": self.docs[int(idx)],
                    "idx": int(idx),
                }
            )
        return results


def save_trajectories(results, save_dir="./trajectories", filename="trajectories.pt"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(results, save_path)
    print(f"Trajectories saved to {save_path}")
    return save_path
