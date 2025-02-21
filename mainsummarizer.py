from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from fpdf import FPDF

# Load the PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text = "\n".join([doc.page_content for doc in documents])  # Combine all pages
    return text

# Split text into chunks
def split_text(text, chunk_size=4000, chunk_overlap=400):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Summarize chunks using Mistral
def summarize_chunks(chunks):
    llm = Ollama(model="mistral")  # Load Mistral model
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing Chunk {i+1}/{len(chunks)}...")  
        prompt = f"Summarize the following text:\n\n{chunk}"
        summary = llm.invoke(prompt)  # Use LangChain's invoke method
        summaries.append(summary)
        # if i == 0:  # Stop after 2 iterations (index 0 and 1)
        #     break  
    return "\n\n".join(summaries)

# Save the summary as a PDF
def save_summary_to_pdf(summary, output_pdf_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Use Arial Unicode font
    pdf.add_font("Arial", "", "Arial.ttf", uni=True)
    pdf.set_font("Arial", size=11)

    # Add summary text to the PDF
    pdf.multi_cell(0, 10, summary)

    pdf.output(output_pdf_path, "F")
    print(f"\n✅ Summary saved successfully as: {output_pdf_path}")


# Main Execution
if __name__ == "__main__":
    file_path = "D:\\pcbprojecta\\An_AOI_algorithm_for_PCB_based_on_feature_extraction.pdf" # Change this to your PDF path
    output_pdf_path = "D:\\summarized_output.pdf"  # Output PDF path

    print("Loading PDF...")
    text = load_pdf(file_path)

    print("Splitting text into chunks...")
    chunks = split_text(text)

    print("Summarizing document...")
    summary = summarize_chunks(chunks)

    print("\n--- FINAL SUMMARY ---\n")
    print(summary)  # Print summarized content

    print("\nSaving summary to PDF...")
    save_summary_to_pdf(summary, output_pdf_path)
