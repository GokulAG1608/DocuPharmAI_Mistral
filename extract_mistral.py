import fitz  # PyMuPDF
import json
import csv
import re
from mistralai import Mistral

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Function to retrieve specific content using the Mistral Model API
def query_mistral_model(api_key, model, text, user_prompt):
    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that processes medical research text and categorizes it into structured JSON format."
            },
            {"role": "user", "content": f"{user_prompt}\n\n{text}"},
        ],
    )
    response_content = chat_response.choices[0].message.content.strip()
    
    # Extract JSON content using regex (handles cases where model adds extra text)
    match = re.search(r'\{.*\}', response_content, re.DOTALL)
    if not match:
        print("Error: Model did not return valid JSON.")
        print("Raw Response:", response_content)
        return None

    json_text = match.group(0)
    
    # Ensure response is valid JSON
    try:
        json.loads(json_text)
    except json.JSONDecodeError:
        print("Error: Received invalid JSON response from the model.")
        print("Raw Response:", response_content)
        return None

    return json_text

# Function to save extracted JSON content into a CSV file with categories as column headers
def save_json_to_csv(json_data, csv_path):
    if not json_data:
        print("Error: No valid JSON data to save.")
        return
        
    try:
        data = json.loads(json_data)
        
        # Ensure data is a dictionary
        if not isinstance(data, dict):
            raise ValueError("Extracted content is not a valid JSON object.")
        
        with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            
            # Write headers (keys from JSON)
            writer.writerow(data.keys())

            # Write values in a single row
            writer.writerow(data.values())
        print(f"Data successfully saved to {csv_path}")
    except Exception as e:
        print(f"Error saving JSON to CSV: {e}")

# Example usage
if __name__ == "__main__":
    pdf_path = "uploads//sgac006.pdf"  # Replace with your PDF file path
    api_key = "MISTRAL_API_KEY"  # Replace with your actual API key
    model = "pixtral-12b-2409"
    csv_path = "uploads//Extracted_sgac006(1).csv"  # Path for the CSV output

    user_prompt = ''' I have extracted text from a PDF containing medical research data. Your task is to extract, categorize, and structure the summarized text into the following format with strict adherence to the provided categories:
                            1. TITLE  
                            2. EDMS NO  
                            3. PRODUCT  
                            4. PRODUCT CODE  
                            5. PROTOCOL NO
                            6. GENERIC NAME  
                            7. REGULATORY AGENCY  
                            8. PREGNANCY OR LACTATION  
                            9. STUDY START DATE (ACTUAL OR PROJECTED)  
                            10. STUDY END DATE (ACTUAL OR PROJECTED FINAL REPORT DELIVERY)  
                            11. FIRST INTERIM REPORT  
                            12. STUDY DESIGN  
                            13. SAMPLE SIZE (EXPECTED)  
                            14. DATA SOURCE TYPE  
                            15. DATA SOURCE NAME  
                            16. EXPOSURE  
                            17. COMPARATOR  
                            18. PRIMARY OUTCOME  
                            19. SECONDARY OUTCOME  
                            20. DATA VALIDITY  
                            21. INFANT FOLLOW-UP DURATION  
                            22. DATA ANALYSIS  
                            23. COUNTRY  
                            24. AGE RANGE  
                            25. LIMITATION  

                            ### Extraction Rules & Formatting Guidelines:  
                            - **Accurate Mapping:** Extract content from the PDF and assign it to the corresponding category based on its meaning.  
                            - **Original Terminology:** Do **not** modify, rephrase, or interpret the extracted content—keep it as found in the document.  
                            - **Preserve Numerical Values & Dates:** Maintain the format of dates, regulatory codes, and numerical values exactly as they appear in the document.  
                            - **Empty Fields:** If a category is missing from the text, return it as an empty string (`""`). Do **not** omit any fields.  
                            - **Strict JSON Compliance:** The output **must** be formatted as valid JSON.  
                            - **No Additional Text:** Do **not** include explanations, introductions, or comments in the output—return **only** the JSON response.  

                            ### Output Format (Strictly JSON)  
                            
                            ```json
                            {
                                "TITLE": "...",
                                "EDMS NO": "...",
                                "PRODUCT": "...",
                                "PRODUCT CODE": "...",
                                "PROTOCOL NO": "...",
                                "GENERIC NAME": "...",
                                "REGULATORY AGENCY": "...",
                                "PREGNANCY OR LACTATION": "...",
                                "STUDY START DATE (ACTUAL OR PROJECTED)": "...",
                                "STUDY END DATE (ACTUAL OR PROJECTED FINAL REPORT DELIVERY)": "...",
                                "FIRST INTERIM REPORT": "...",
                                "STUDY DESIGN": "...",
                                "SAMPLE SIZE (EXPECTED)": "...",
                                "DATA SOURCE TYPE": "...",
                                "DATA SOURCE NAME": "...",
                                "EXPOSURE": "...",
                                "COMPARATOR": "...",
                                "PRIMARY OUTCOME": "...",
                                "SECONDARY OUTCOME": "...",
                                "DATA VALIDITY": "...",
                                "INFANT FOLLOW-UP DURATION": "...",
                                "DATA ANALYSIS": "...",
                                "COUNTRY": "...",
                                "AGE RANGE": "...",
                                "LIMITATION": "..."
                            }
    '''

    text = extract_text_from_pdf(pdf_path)
    response = query_mistral_model(api_key, model, text, user_prompt)
    save_json_to_csv(response, csv_path)


