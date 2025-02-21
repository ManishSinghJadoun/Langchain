import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from tkinter import filedialog
import chardet

# Ensure output directory exists
OUTPUT_FOLDER = "raggraph"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)  # Read a sample
        result = chardet.detect(raw_data)
        return result["encoding"]

# Load and extract text from PDF or TXT file
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])  # Combine all pages
    else:
        file_encoding = detect_encoding(file_path)
        with open(file_path, "r", encoding=file_encoding, errors="replace") as file:
            text = file.read()
    return text

# Generate knowledge graph triples using LLM
def generate_knowledge_graph(text):
    llm = Ollama(model="mistral")  # Load Mistral model

    prompt = f"""Extract key knowledge graph triples (subject, predicate, object) 
                 from the following text and format them as a JSON list:
                 Example: 
                 [["Artificial Intelligence", "is a field of", "Computer Science"],
                  ["Machine Learning", "is a subset of", "Artificial Intelligence"]]

                 Text: {text[:5000]}"""  # Limit to 5000 chars
    
    response = llm.invoke(prompt)  # Call the LLM
    print("\nRaw LLM Response:\n", response)  # Debugging Output

    # Parse response into a list of triples
    try:
        triples = json.loads(response)  # Convert response to a Python list
    except json.JSONDecodeError:
        print("Error parsing JSON. Please check the model output.")
        triples = []
    return triples

# Save knowledge graph as JSON
def save_json(triples, filename="knowledge_graph.json"):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=4)
    print(f"Knowledge graph saved as JSON: {file_path}")

# Build and visualize the knowledge graph
def visualize_knowledge_graph(triples, filename="knowledge_graph.png"):
    G = nx.DiGraph()  # Create a directed graph

    for subject, predicate, obj in triples:
        G.add_edge(subject, obj, label=predicate)

    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G)  # Layout positioning

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=10, font_weight="bold")
    
    # Draw edge labels
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Knowledge Graph")

    # Save graph as image
    image_path = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(image_path)
    print(f"Knowledge graph saved as Image: {image_path}")
    plt.show()

# Main Execution
if __name__ == "__main__":
    file_path = "D:\\22118010_manish_singh_jadoun_mt_bsbe.pdf"

    if not file_path:
        print("❌ No file selected. Exiting...")
    else:
        print("📂 Loading document...")
        text = load_document(file_path)

        print("🧠 Generating knowledge graph...")
        triples = generate_knowledge_graph(text)

        print("\n📌 Extracted Triples:\n", triples)

        print("💾 Saving knowledge graph as JSON...")
        save_json(triples)

        print("📊 Visualizing and saving knowledge graph...")
        visualize_knowledge_graph(triples)

        print("\n✅ Process completed successfully. Outputs saved in 'raggraph/' folder.")
