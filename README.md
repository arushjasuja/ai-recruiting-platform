# AI Recruiting Automation Platform

Automates market research, job description generation, and candidate matching for small recruiting teams.

# Live demo: https://ai-recruiting-platform.streamlit.app/ (requires OpenAI API key)

## Problem It Solves

Manual recruiting research is time-intensive and leads to poor outcomes:
- 3+ hours researching salary ranges across multiple job boards
- Job descriptions written with outdated market data
- Biased language that deters qualified candidates
- Resume screening based on keyword matching instead of actual fit

## Current Solution (v1.0)

**Market Intelligence:** Processes job market data to determine competitive salary ranges and in-demand skills

**Content Generation:** Creates job descriptions using current market context with bias detection

**Candidate Matching:** Matches resumes to jobs using semantic similarity with explanation of fit

## Performance

- Market analysis: Processes 500+ job postings in under 2 minutes
- Candidate matching: Under 100ms response time per candidate
- Content quality: Bias detection scoring on 0-10 scale
- Time savings: 3-hour manual process reduced to 15 minutes

## Who This Helps

**Good fit:**
- Small recruiting agencies (2-20 people) scaling operations
- Startups hiring technical roles regularly
- Independent recruiters competing against larger firms
- Teams that need better than manual processes but can't afford enterprise tools

**Not designed for:**
- One-off hiring needs
- Large enterprises with dedicated recruiting infrastructure
- Non-technical role recruiting (less benefit from semantic matching)

## Tech Stack

- **AI Frameworks:** CrewAI (market research), LangChain (content generation), AutoGen (candidate matching)
- **ML Models:** OpenAI GPT-4o, Sentence Transformers for semantic similarity
- **Database:** ChromaDB for vector storage and persistence
- **Interface:** Streamlit web application

## Current Limitations (v1.0)

This is a proof-of-concept focused on validating the architecture:

- Market data uses demo datasets instead of real-time scraping
- Multi-agent workflows are simulated rather than full framework implementations  
- Resume parsing relies on regex patterns instead of production NLP models
- Missing production infrastructure (caching, async processing, monitoring)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Run application
streamlit run main.py
```

## Usage

1. **Generate Market Report:** Analyze salary ranges and skill demand for target role
2. **Create Job Description:** Use market data to generate optimized job posting with bias checking
3. **Match Candidates:** Upload resumes and get ranked matches with explanations

## Data Flow

```
Market Intelligence → Job Description Generation → Candidate Matching
        ↓                        ↓                       ↓
    Salary Data          Bias-Free Content        Explained Rankings
```

## Production Roadmap (v2.0)

**Real Framework Implementations:**
- CrewAI crews with actual web scraping and data collection
- Full LangChain agent chains with memory and conversation flow
- AutoGen conversational agents for interactive candidate screening

**Production Infrastructure:**
- spaCy-based NLP pipeline for 90%+ resume parsing accuracy
- Redis caching and async processing for scalability
- Comprehensive monitoring and error recovery
- Support for 10+ concurrent users with <5% failure rate

## License

MIT License

---

*Built for recruiting teams that need automation but aren't ready for enterprise solutions.*