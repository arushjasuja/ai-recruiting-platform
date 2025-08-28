from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime
import uuid

class Job(BaseModel):
    id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:8]}")
    title: str
    company: str
    location: str
    skills_required: List[str] = []
    salary_min: int = 0
    salary_max: int = 0
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

class Candidate(BaseModel):
    id: str = Field(default_factory=lambda: f"cand_{uuid.uuid4().hex[:8]}")
    name: str
    skills: List[str] = []
    experience_years: int = 0
    resume_text: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

class MarketReport(BaseModel):
    id: str = Field(default_factory=lambda: f"rpt_{uuid.uuid4().hex[:8]}")
    job_title: str
    location: str
    total_postings: int
    salary_range: Dict[str, int]
    top_skills: List[Dict[str, Any]]
    market_demand: str
    generated_at: datetime = Field(default_factory=datetime.now)

class MatchResult(BaseModel):
    candidate_id: str
    job_id: str
    similarity_score: float
    match_percentage: int
    explanation: str