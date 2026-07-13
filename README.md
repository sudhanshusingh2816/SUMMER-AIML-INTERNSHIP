# AI Resume Analyzer + ATS Score + Interview Coach

## What it does
- Extracts text from an uploaded PDF resume
- Computes an ATS match score against a pasted job description (TF-IDF + cosine similarity)
- Shows missing keywords the resume should probably include
- Checks for standard resume sections (Education, Experience, Skills, etc.)
- Gives rule-based AI feedback (action verbs, quantifiable metrics, section gaps)
- Generates likely interview questions based on the resume + job description

## Step 1: Install dependencies
Open a terminal in this folder and run:
```
pip install -r requirements.txt
```
If that's slow/fails on some systems, install individually:
```
pip install streamlit pdfplumber scikit-learn plotly
```

## Step 2: Run the app locally
```
streamlit run app.py
```
This opens the app in your browser automatically (usually at http://localhost:8501).

## Step 3: Test it
1. Upload any PDF resume (yours works best for the demo)
2. Paste a real job description (copy one from LinkedIn/Naukri/Indeed)
3. Click "Analyze Resume"
4. You'll see: ATS score gauge, missing keywords, feedback, and interview questions

## Step 4: Deploy for free (so you have a live link, not just localhost)
1. Create a GitHub repo, push these 3 files: `app.py`, `requirements.txt`, `README.md`
2. Go to https://share.streamlit.io (Streamlit Community Cloud)
3. Sign in with GitHub, click "New app", select your repo and `app.py`
4. Click Deploy — you'll get a public URL in a couple of minutes

## Step 5: Presentation talking points
- **Problem**: Most resumes get filtered out by ATS software before a human ever sees them
- **Approach**: NLP-based text extraction + TF-IDF similarity scoring (AI/NLP technique) combined with rule-based analytics (DA technique) for feedback
- **Demo**: Use your own resume live — most convincing part of any presentation
- **Extension ideas** (mention as "future scope" if asked): plug in an LLM API (OpenAI/Hugging Face) for more natural-language feedback, support DOCX resumes, add a resume score history/tracking dashboard

## Notes
- Works best with text-based PDFs (not scanned/image resumes)
- No paid API required — fully self-contained and free to run
