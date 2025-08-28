import chromadb
import json
from typing import List
from datetime import datetime

class DatabaseManager:
    def __init__(self, path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.jobs = self.client.get_or_create_collection("jobs")
        self.candidates = self.client.get_or_create_collection("candidates")
        self.reports = self.client.get_or_create_collection("reports")
    
    def _flatten_data(self, data):
        """Convert complex types to JSON strings for ChromaDB"""
        flattened = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                flattened[key] = json.dumps(value)
            else:
                flattened[key] = value
        return flattened
    
    def _unflatten_data(self, data):
        """Convert JSON strings back to complex types"""
        unflattened = {}
        for key, value in data.items():
            if isinstance(value, str) and key in ['salary_range', 'top_skills', 'skills_required', 'skills']:
                try:
                    unflattened[key] = json.loads(value)
                except:
                    unflattened[key] = value
            else:
                unflattened[key] = value
        return unflattened
    
    def store_job(self, job):
        data = job.model_dump()
        data['created_at'] = job.created_at.isoformat()
        data = self._flatten_data(data)
        self.jobs.upsert(documents=[f"{job.title} {job.description}"], metadatas=[data], ids=[job.id])
    
    def store_candidate(self, candidate):
        data = candidate.model_dump()
        data['created_at'] = candidate.created_at.isoformat()
        data = self._flatten_data(data)
        self.candidates.upsert(documents=[candidate.resume_text], metadatas=[data], ids=[candidate.id])
    
    def store_report(self, report):
        data = report.model_dump()
        data['generated_at'] = report.generated_at.isoformat()
        data = self._flatten_data(data)
        self.reports.upsert(documents=[f"{report.job_title} {report.location}"], metadatas=[data], ids=[report.id])
    
    def get_all_jobs(self):
        from .models import Job
        results = self.jobs.get()
        jobs = []
        for meta in results['metadatas']:
            meta = self._unflatten_data(meta)
            meta['created_at'] = datetime.fromisoformat(meta['created_at'])
            jobs.append(Job(**meta))
        return jobs
    
    def get_all_candidates(self):
        from .models import Candidate
        results = self.candidates.get()
        candidates = []
        for meta in results['metadatas']:
            meta = self._unflatten_data(meta)
            meta['created_at'] = datetime.fromisoformat(meta['created_at'])
            candidates.append(Candidate(**meta))
        return candidates
    
    def get_reports(self):
        from .models import MarketReport
        results = self.reports.get()
        reports = []
        for meta in results['metadatas']:
            meta = self._unflatten_data(meta)
            meta['generated_at'] = datetime.fromisoformat(meta['generated_at'])
            reports.append(MarketReport(**meta))
        return reports