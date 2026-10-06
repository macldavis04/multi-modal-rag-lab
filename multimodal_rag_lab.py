import json

import boto3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


json_data = load_json('multimodal_knowledge_base.json')
data = pd.DataFrame(json_data)

print("Columns in the dataset:", data.columns)
print(data.head())

persona = {
    "description": "A history enthusiast interested in landmarks.",
    "interests": ["history", "landmarks", "historic sites", "monuments"],
}

query = " ".join(persona["interests"])
sentence_query = f"What descriptions would interest {persona['description']}?"

print("\nQuery:", query)

captions = data['caption_reference_description']

vectorizer = TfidfVectorizer()
caption_vectors = vectorizer.fit_transform(captions)


def top_matches(query_text, k=5):
    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, caption_vectors)
    top_indices = similarities.argsort()[0][-k:][::-1]
    results = data.iloc[top_indices].copy()
    results["similarity"] = similarities[0, top_indices]
    return results


top_recommendations = top_matches(query)

pd.set_option("display.max_colwidth", 90)
print("\nTop Recommendations (keyword query):")
print(top_recommendations[["caption_reference_description", "similarity"]])

print(f"\nComparison - full sentence query: {sentence_query}")
print(top_matches(sentence_query)[["caption_reference_description", "similarity"]])

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")


def get_model_id() -> str:
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return f"arn:aws:bedrock:us-east-1:{account_id}:inference-profile/us.anthropic.claude-sonnet-4-6"


def ask_claude(prompt, system=None, temperature=1.0, max_tokens=500):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }
    if system:
        body["system"] = system

    response = bedrock.invoke_model(modelId=get_model_id(), body=json.dumps(body))
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


recommendations_text = "\n".join(
    [f"- {row['caption_reference_description']} (image: {row['image_url']})" for _, row in top_recommendations.iterrows()]
)

prompt = (
    f"Given the persona '{persona['description']}', explain why each of the following "
    f"descriptions is a suitable recommendation:\n\n{recommendations_text}"
)

explanation = ask_claude(
    prompt,
    system="You are an expert recommender system.",
    temperature=0.7,
    max_tokens=800
)

print("\nGenerated Explanation:", explanation)