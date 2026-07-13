import streamlit as st
import pdfplumber
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")

# ---------------------------------------------------------
# PREMIUM AI SAAS THEME (glassmorphism, per design spec)
# ---------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ---------- BACKGROUND ---------- */
.stApp {
    background:
        radial-gradient(ellipse 800px 600px at 15% 10%, rgba(79,140,255,0.16), transparent 60%),
        radial-gradient(ellipse 800px 600px at 85% 15%, rgba(139,92,246,0.16), transparent 60%),
        radial-gradient(ellipse 900px 700px at 50% 100%, rgba(56,189,248,0.08), transparent 60%),
        linear-gradient(180deg, #070B14 0%, #0B1120 45%, #111827 100%);
    background-attachment: fixed;
}

/* Corner blur orbs */
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    width: 420px; height: 420px;
    border-radius: 50%;
    filter: blur(120px);
    z-index: 0;
    pointer-events: none;
}
.stApp::before {
    top: -120px; left: -120px;
    background: rgba(79,140,255,0.25);
}
.stApp::after {
    bottom: -120px; right: -120px;
    background: rgba(139,92,246,0.22);
}

/* Main container: centered, generous whitespace */
.block-container {
    max-width: 1080px;
    padding: 3rem 2rem 4rem 2rem !important;
    margin: 0 auto;
    animation: fadeIn 0.7s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---------- TITLE ---------- */
.hero-wrap {
    position: relative;
    text-align: center;
    padding: 2rem 0 2.5rem 0;
}
.hero-glow {
    position: absolute;
    top: 50%; left: 50%;
    width: 480px; height: 240px;
    transform: translate(-50%, -50%);
    background: radial-gradient(ellipse, rgba(79,140,255,0.35), rgba(139,92,246,0.2) 55%, transparent 75%);
    filter: blur(50px);
    animation: floatGlow 6s ease-in-out infinite;
    z-index: 0;
}
@keyframes floatGlow {
    0%, 100% { transform: translate(-50%, -50%) scale(1); }
    50% { transform: translate(-50%, -50%) scale(1.12); }
}
.hero-title {
    position: relative;
    z-index: 1;
    font-size: 48px;
    font-weight: 800;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-title svg { filter: drop-shadow(0 0 12px rgba(79,140,255,0.6)); }
.hero-subtitle {
    position: relative;
    z-index: 1;
    color: #94A3B8;
    font-size: 16px;
    font-weight: 500;
    margin-top: 10px;
    letter-spacing: 0.2px;
}

/* ---------- GLASS CARDS (input columns) ---------- */
[data-testid="column"]:has([data-testid="stFileUploaderDropzone"]),
[data-testid="column"]:has(.stTextArea) {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 24px;
    padding: 1.75rem !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 250ms ease, box-shadow 250ms ease;
}
[data-testid="column"]:has([data-testid="stFileUploaderDropzone"]):hover,
[data-testid="column"]:has(.stTextArea):hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 44px rgba(79,140,255,0.18);
}

/* Card section headings (icon + label) */
.card-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #FFFFFF;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 14px;
}
.card-heading svg { color: #4F8CFF; }

[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    border: 1px dashed rgba(148,163,184,0.35) !important;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(148,163,184,0.25) !important;
    color: #E2E8F0 !important;
}

h2, h3 { color: #F1F5F9 !important; font-weight: 700 !important; }
p, label, .stMarkdown { color: #CBD5E1; }

/* ---------- BUTTON ---------- */
.stButton > button {
    background: linear-gradient(90deg, #4F8CFF, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    height: 55px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    box-shadow: 0 6px 24px rgba(79,140,255,0.35);
    transition: all 250ms ease;
}
.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 10px 34px rgba(139,92,246,0.5);
}

/* ---------- RESULT GLASS CARDS ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.28);
    padding: 0.5rem 0.25rem;
    margin-bottom: 1.75rem;
    transition: transform 250ms ease, box-shadow 250ms ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 40px rgba(79,140,255,0.15);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { color: #FFFFFF !important; }
[data-testid="stMetricLabel"] { color: #94A3B8 !important; }

/* Keyword tags */
.kw-tag {
    background: linear-gradient(90deg, #4F8CFF, #8B5CF6);
    color: #FFFFFF;
    padding: 7px 16px;
    border-radius: 999px;
    margin: 4px;
    display: inline-block;
    font-size: 13.5px;
    font-weight: 600;
    box-shadow: 0 3px 12px rgba(79,140,255,0.3);
}

/* Expander (interview questions) */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 0 0 12px 12px !important;
    color: #94A3B8 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    backdrop-filter: blur(12px);
}

hr { border-color: rgba(255,255,255,0.1) !important; }
[data-testid="stCaptionContainer"] { color: #64748B !important; text-align: center; }

/* ---------- RESPONSIVE ---------- */
@media (max-width: 768px) {
    .hero-title { font-size: 32px; }
    .block-container { padding: 2rem 1rem 3rem 1rem !important; }
    [data-testid="column"]:has([data-testid="stFileUploaderDropzone"]),
    [data-testid="column"]:has(.stTextArea) {
        padding: 1.2rem !important;
    }
}

</style>
""", unsafe_allow_html=True)

# Lucide-style inline SVG icons (stroke-based, no external dependency)
ICONS = {
    "brain": '<svg xmlns="http://www.w3.org/2000/svg" width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/></svg>',
    "upload": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    "filetext": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>',
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "sparkles": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.582a.5.5 0 0 1 0 .963L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/></svg>',
    "bot": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>',
    "target": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "key": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="7.5" cy="15.5" r="5.5"/><path d="m21 2-9.6 9.6"/><path d="m15.5 7.5 3 3L22 7l-3-3"/></svg>',
}


STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out over own same
shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this
those through to too under until up very was wasn't we we'd we'll we're we've were weren't
what what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves experience
years work working strong ability skills including etc using use used year looking
required requirements preferred responsibilities role team candidate join
""".split())

ACTION_VERBS = ["led", "built", "developed", "designed", "created", "managed",
                "implemented", "improved", "increased", "reduced", "achieved",
                "launched", "optimized", "analyzed", "collaborated", "delivered"]

RESUME_SECTIONS = ["education", "experience", "skills", "projects",
                    "certification", "summary", "objective"]

# Curated list of real technical/professional skills to match against,
# instead of relying on raw word-frequency (which picks up filler words).
SKILLS_DB = {
    # Languages
    "python": "Python", "java": "Java", "c++": "C++", "javascript": "JavaScript",
    "typescript": "TypeScript", "sql": "SQL", "r": "R", "html": "HTML", "css": "CSS",
    # AI / ML
    "machine learning": "Machine Learning", "deep learning": "Deep Learning",
    "artificial intelligence": "Artificial Intelligence", "nlp": "NLP",
    "natural language processing": "NLP", "computer vision": "Computer Vision",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch", "keras": "Keras",
    "scikit-learn": "Scikit-learn", "scikit learn": "Scikit-learn",
    "opencv": "OpenCV", "neural network": "Neural Networks",
    # Data
    "data analysis": "Data Analysis", "data visualization": "Data Visualization",
    "data science": "Data Science", "data engineering": "Data Engineering",
    "pandas": "Pandas", "numpy": "NumPy", "matplotlib": "Matplotlib",
    "power bi": "Power BI", "tableau": "Tableau", "excel": "Excel",
    "big data": "Big Data", "spark": "Spark", "hadoop": "Hadoop",
    "etl": "ETL", "statistics": "Statistics", "a/b testing": "A/B Testing",
    # Cloud & DevOps
    "aws": "AWS", "azure": "Azure", "gcp": "GCP", "docker": "Docker",
    "kubernetes": "Kubernetes", "ci/cd": "CI/CD", "git": "Git", "github": "GitHub",
    "linux": "Linux", "cloud computing": "Cloud Computing",
    # Web / Backend
    "rest api": "REST API", "fastapi": "FastAPI", "flask": "Flask",
    "django": "Django", "node.js": "Node.js", "react": "React",
    "streamlit": "Streamlit", "mongodb": "MongoDB", "mysql": "MySQL",
    "postgresql": "PostgreSQL", "microservices": "Microservices",
    # Practices
    "agile": "Agile", "scrum": "Scrum", "jira": "Jira",
    "unit testing": "Unit Testing", "pytest": "Pytest",
    "object oriented programming": "OOP", "data structures": "Data Structures",
    "algorithms": "Algorithms", "project management": "Project Management",
    # Soft skills
    "communication": "Communication", "leadership": "Leadership",
    "problem solving": "Problem Solving", "teamwork": "Teamwork",
    "team collaboration": "Team Collaboration", "time management": "Time Management",
    "critical thinking": "Critical Thinking", "analytical skills": "Analytical Skills",
}


# ---------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------
# CORE ATS LOGIC
# ---------------------------------------------------------
def get_ats_score(resume_text_raw, jd_text_raw):
    # Signal 1: TF-IDF cosine similarity (captures overall text/phrasing overlap)
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([clean_text(resume_text_raw), clean_text(jd_text_raw)])
    tfidf_score = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Signal 2: actual skill keyword overlap (what real ATS tools weigh heavily)
    jd_skills = find_skills_in_text(jd_text_raw)
    if jd_skills:
        resume_skills = set(find_skills_in_text(resume_text_raw))
        overlap_score = len([s for s in jd_skills if s in resume_skills]) / len(jd_skills)
    else:
        overlap_score = tfidf_score  # fall back if no known skills detected in JD

    # Weighted blend: keyword overlap matters more for ATS-style matching
    final_score = (0.35 * tfidf_score) + (0.65 * overlap_score)
    return round(final_score * 100, 2)


def find_skills_in_text(text):
    """Return list of (key, display_name) skills from SKILLS_DB found in text."""
    text_lower = " " + re.sub(r"[^a-z0-9\+\./\s]", " ", text.lower()) + " "
    found = []
    for key, display in SKILLS_DB.items():
        pattern = r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.append(display)
    return found


def get_top_keywords(text, top_n=25):
    # Prefer real matched skills; fall back to frequency if too few matches
    matched = find_skills_in_text(text)
    if len(matched) >= 5:
        return matched[:top_n]
    words = [w for w in clean_text(text).split() if w not in STOPWORDS and len(w) > 2]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(top_n)]


def get_missing_keywords(resume_text, jd_text, top_n=15):
    jd_skills = find_skills_in_text(jd_text)
    resume_skills = set(find_skills_in_text(resume_text))
    missing = [kw for kw in jd_skills if kw not in resume_skills]
    return missing[:top_n]


def check_sections(resume_text):
    text_lower = resume_text.lower()
    found = [s for s in RESUME_SECTIONS if s in text_lower]
    missing = [s for s in RESUME_SECTIONS if s not in text_lower]
    return found, missing


def check_action_verbs(resume_text):
    text_lower = resume_text.lower()
    found = [v for v in ACTION_VERBS if v in text_lower]
    return found


# ---------------------------------------------------------
# FEEDBACK GENERATION (rule-based "AI coach")
# ---------------------------------------------------------
def generate_feedback(resume_text, score, missing_keywords, missing_sections, action_verbs_found):
    feedback = []

    if score >= 75:
        feedback.append(f"Strong match — your resume aligns well with this job description ({score}%).")
    elif score >= 50:
        feedback.append(f"Moderate match ({score}%). Adding more role-specific keywords will boost this significantly.")
    else:
        feedback.append(f"Low match ({score}%). Your resume likely needs targeted keyword additions to pass ATS filters for this role.")

    if missing_keywords:
        top5 = ", ".join(missing_keywords[:5])
        feedback.append(f"Consider naturally adding these keywords if genuinely relevant: {top5}.")

    if missing_sections:
        feedback.append(f"Your resume appears to be missing clear section headers for: {', '.join(missing_sections)}. ATS systems rely heavily on labeled sections.")

    if len(action_verbs_found) < 4:
        feedback.append("Use more strong action verbs (e.g., led, built, improved, delivered) to make achievements stand out.")
    else:
        feedback.append("Good use of action verbs — this makes achievements sound impactful.")

    word_count = len(resume_text.split())
    if word_count < 150:
        feedback.append("Resume seems short. Consider adding more detail on projects and quantifiable achievements.")
    elif word_count > 1000:
        feedback.append("Resume is quite long — consider trimming to the most relevant, recent, and impactful points.")

    if not re.search(r"\d+\s*%|\d+\s*percent|\d+x|\$\d+|\d+\s*(users|clients|projects|hours|days)", resume_text.lower()):
        feedback.append("Add quantifiable metrics (%, numbers, scale) to achievements — they carry more weight than descriptions alone.")

    return feedback


# ---------------------------------------------------------
# INTERVIEW QUESTION GENERATION (template-based, with sample answers)
# ---------------------------------------------------------
def extract_relevant_sentence(resume_text, keyword):
    """Find the sentence in the resume that mentions this keyword, for grounding sample answers."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", resume_text)
    keyword_lower = keyword.lower()
    for s in sentences:
        s_clean = s.strip()
        if keyword_lower in s_clean.lower() and len(s_clean) > 15:
            return s_clean.rstrip(".") + "."
    return None


def generate_interview_questions(resume_text, jd_text, missing_keywords, resume_keywords):
    qa_pairs = []

    tech_pool = (resume_keywords[:10] if resume_keywords else []) + missing_keywords[:5]
    tech_pool = list(dict.fromkeys(tech_pool))[:6]  # dedupe, cap

    # --- Skill-based questions, grounded in actual resume lines where possible ---
    for kw in tech_pool[:4]:
        question = f"Can you walk me through your experience with {kw}?"
        evidence = extract_relevant_sentence(resume_text, kw)
        if evidence:
            answer = (
                f"Sample answer approach: Your resume mentions — \"{evidence}\" "
                f"Use this as your foundation: briefly explain the context (what the project/role was), "
                f"the specific way you used {kw}, and the measurable outcome or impact. "
                f"Keep it to 60-90 seconds and end with the result."
            )
        else:
            answer = (
                f"Sample answer approach: This wasn't explicitly found in your resume, so if you do have "
                f"experience with {kw}, mention where you used it (course, project, internship), what you built "
                f"or solved with it, and the outcome. If you don't have direct experience, be honest and briefly "
                f"mention a related skill or your plan to learn it."
            )
        qa_pairs.append({"question": question, "answer": answer})

    # --- Behavioral questions, guided by resume content when available ---
    metric_sentence = None
    metric_match = re.search(
        r"[^.]*\d+\s*(%|percent|x|users|clients|projects|hours|days)[^.]*\.", resume_text, re.IGNORECASE
    )
    if metric_match:
        metric_sentence = metric_match.group(0).strip()

    behavioral_bank = [
        {
            "question": "Tell me about a challenging project you worked on and how you handled it.",
            "answer": (
                "Sample answer approach: Use the STAR method — Situation (what was the project/context), "
                "Task (what was your specific responsibility), Action (what steps you took, tools you used), "
                "Result (the outcome, ideally with a number). "
                + (f"You could build this around: \"{metric_sentence}\"" if metric_sentence else
                   "Pick your most detailed project from the resume and quantify the result if you can.")
            ),
        },
        {
            "question": "Describe a time you had to learn a new skill quickly for a project.",
            "answer": (
                "Sample answer approach: Pick a skill or tool from your Skills/Projects section that you "
                "picked up recently. Explain why you needed it, how you learned it (docs, course, trial and "
                "error), and how you applied it successfully. Interviewers want to see your learning process, "
                "not just the outcome."
            ),
        },
        {
            "question": "How do you prioritize tasks when working under a tight deadline?",
            "answer": (
                "Sample answer approach: Describe a concrete method (e.g., breaking tasks down, focusing on "
                "high-impact items first, communicating early if something's at risk). If your resume mentions "
                "multiple projects or internships, reference how you balanced them."
            ),
        },
        {
            "question": "Tell me about a time you disagreed with a teammate — how did you resolve it?",
            "answer": (
                "Sample answer approach: Briefly describe the disagreement without placing blame, explain how "
                "you communicated your perspective, how you reached a resolution, and what you learned about "
                "collaboration. If you led a team (check your resume), reference that experience."
            ),
        },
        {
            "question": "What's an achievement from your resume you're most proud of, and why?",
            "answer": (
                "Sample answer approach: Choose the project or role with the strongest measurable result. "
                + (f"For example: \"{metric_sentence}\"" if metric_sentence else
                   "Walk through the challenge, your specific contribution, and the impact.")
                + " Explain briefly why it mattered to you personally, not just the numbers."
            ),
        },
    ]
    qa_pairs.extend(behavioral_bank[:3])

    # --- Role-fit question ---
    role_hint = get_top_keywords(jd_text, top_n=3)
    if role_hint:
        qa_pairs.append({
            "question": f"Why are you interested in a role involving {', '.join(role_hint)}?",
            "answer": (
                f"Sample answer approach: Connect 1-2 things from your background (skills, projects, or "
                f"coursework) to {', '.join(role_hint)}, then explain what draws you to this specific type of "
                f"work going forward. Keep it genuine and specific rather than generic enthusiasm."
            ),
        })

    return qa_pairs



# ---------------------------------------------------------
# GAUGE CHART
# ---------------------------------------------------------
def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "ATS Match Score", "font": {"size": 18, "color": "#CBD5E1"}},
        number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748B", "tickfont": {"color": "#64748B"}},
            "bar": {"color": "#4F8CFF"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(239,68,68,0.18)"},
                {"range": [50, 75], "color": "rgba(250,204,21,0.18)"},
                {"range": [75, 100], "color": "rgba(56,189,248,0.22)"},
            ],
        },
    ))
    fig.update_layout(
        height=280, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
    )
    return fig


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-glow"></div>
    <div class="hero-title">{ICONS['brain']} AI Resume Analyzer</div>
    <div class="hero-subtitle">Intelligent ATS Scoring &nbsp;•&nbsp; Resume Feedback &nbsp;•&nbsp; AI Interview Coach</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"<div class='card-heading'>{ICONS['upload']} Upload Resume (PDF)</div>", unsafe_allow_html=True)
    resume_file = st.file_uploader("Upload your resume", type=["pdf"], label_visibility="collapsed")

with col2:
    st.markdown(f"<div class='card-heading'>{ICONS['filetext']} Paste Job Description</div>", unsafe_allow_html=True)
    jd_text_input = st.text_area("Paste the job description here", height=200, label_visibility="collapsed", placeholder="Paste the job description here...")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
b1, b2, b3 = st.columns([1, 1.2, 1])
with b2:
    analyze_btn = st.button(f"Analyze Resume", type="primary", use_container_width=True)

if analyze_btn:
    if not resume_file or not jd_text_input.strip():
        st.warning("Please upload a resume PDF and paste a job description before analyzing.")
    else:
        with st.spinner("✨ Analyzing your resume..."):
            resume_text_raw = extract_text_from_pdf(resume_file)

            if not resume_text_raw.strip():
                st.error("Couldn't extract text from this PDF. Try a different file (avoid scanned/image-only resumes).")
                st.stop()

            score = get_ats_score(resume_text_raw, jd_text_input)
            missing_keywords = get_missing_keywords(resume_text_raw, jd_text_input)
            resume_keywords = get_top_keywords(resume_text_raw)
            found_sections, missing_sections = check_sections(resume_text_raw)
            action_verbs_found = check_action_verbs(resume_text_raw)

            feedback = generate_feedback(resume_text_raw, score, missing_keywords, missing_sections, action_verbs_found)
            questions = generate_interview_questions(resume_text_raw, jd_text_input, missing_keywords, resume_keywords)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.success("Analysis complete!")

        # --- Score + Quick Stats ---
        with st.container(border=True):
            g1, g2 = st.columns([1, 1.3])
            with g1:
                st.plotly_chart(create_gauge(score), use_container_width=True)
            with g2:
                st.markdown(f"<div class='card-heading'>{ICONS['target']} Quick Stats</div>", unsafe_allow_html=True)
                st.metric("Word Count", len(resume_text_raw.split()))
                st.metric("Sections Found", f"{len(found_sections)}/{len(RESUME_SECTIONS)}")
                st.metric("Action Verbs Used", len(action_verbs_found))

        # --- Missing Keywords ---
        with st.container(border=True):
            st.markdown(f"<div class='card-heading'>{ICONS['key']} Missing Keywords (from Job Description)</div>", unsafe_allow_html=True)
            if missing_keywords:
                tags_html = " ".join([f"<span class='kw-tag'>{kw}</span>" for kw in missing_keywords])
                st.markdown(f"<div style='line-height:2.6'>{tags_html}</div>", unsafe_allow_html=True)
            else:
                st.write("Great — no major missing keywords detected!")

        # --- Feedback ---
        with st.container(border=True):
            st.markdown(f"<div class='card-heading'>{ICONS['sparkles']} AI Resume Feedback</div>", unsafe_allow_html=True)
            for point in feedback:
                st.markdown(f"- {point}")

        # --- Interview Questions ---
        with st.container(border=True):
            st.markdown(f"<div class='card-heading'>{ICONS['bot']} Likely Interview Questions</div>", unsafe_allow_html=True)
            for i, qa in enumerate(questions, 1):
                with st.expander(f"Q{i}: {qa['question']}"):
                    st.write(qa["answer"])

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.divider()
st.caption("Built as an AIML internship project — Resume Analyzer using TF-IDF based ATS scoring, rule-based NLP feedback, and dynamic interview question generation.")
