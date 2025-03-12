## AI-powered Resume Screening and Ranking System 

This application ranks resumes based on their relevance to a given job description. The system uses a combination of natural language processing (NLP) techniques, including TF-IDF vectorization and cosine similarity, to compare resumes and rank them in order of best fit.

## Requirements

To run this application, you need to have the following packages installed:

- `streamlit`: For creating an interactive web interface.
- `PyPDF2`: For reading and extracting text from PDF files.
- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For performing vectorization and similarity calculations.

You can install all the required packages by running the following command:

bash
pip install streamlit PyPDF2 pandas scikit-learn


## How It Works

1. **PDF Upload**: The user uploads resumes in PDF format through a simple web interface built with Streamlit.
2. **Text Extraction**: The application uses `PyPDF2` to extract the text content from the PDF files.
3. **TF-IDF Vectorization**: The extracted resume texts and the job description are vectorized using `TfidfVectorizer` from `scikit-learn` to convert them into numerical features.
4. **Cosine Similarity Calculation**: Cosine similarity is calculated between each resume and the job description to determine how similar each resume is to the job description.
5. **Ranking**: The resumes are ranked based on their cosine similarity scores. The most relevant resumes are shown at the top.

## How to Use

1. Clone this repository:
    bash
    git clone <repository_url>
    cd <repository_directory>
    

2. Run the Streamlit application:
    bash
    streamlit run Resume_app.py
    

3. Open the application in your web browser.

4. Upload resumes in PDF format and enter a job description.

5. The application will display the resumes ranked by their relevance to the job description.

## Code Explanation

### Main Imports

python
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


- **streamlit (st)**: Used for creating an interactive web interface.
- **PyPDF2 (PdfReader)**: Extracts text from PDF resumes.
- **pandas (pd)**: Used for storing and manipulating resume data.
- **TfidfVectorizer**: Converts text into numerical features using the Term Frequency-Inverse Document Frequency method.
- **cosine_similarity**: Computes the cosine similarity between the job description and resumes.

### Example Code

python
# Initialize Streamlit interface
st.title("AI Resume Screening & Candidate Ranking System")

# Upload job description and resumes
job_description = st.text_area("Job Description", "Enter job description here...")
uploaded_resumes = st.file_uploader("Upload Resumes", type="pdf", accept_multiple_files=True)

if uploaded_resumes:
    resumes_text = []
    for file in uploaded_resumes:
        # Extract text from PDF
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        resumes_text.append(text)

    # Vectorize the job description and resumes
    documents = [job_description] + resumes_text
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    # Rank resumes based on similarity
    similarity_scores = cosine_sim[0]
    ranked_resumes = sorted(zip(similarity_scores, uploaded_resumes), reverse=True)

    # Display ranked resumes
    for score, resume in ranked_resumes:
        st.write(f"Resume: {resume.name}, Similarity Score: {score}")


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



This README.md file includes:

- An overview of how the app works.
- The list of necessary dependencies and installation instructions.
- Code breakdown and example of how to run the app.
- How the key imports are used to extract resume text and rank them based on relevance.
