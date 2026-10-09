from app import pipeline, web_search, academic_search, query_generator, llm_analyzer
web_search.search_web = lambda q, limit=5: []
academic_search.search_academic_sources = lambda c, q: []
query_generator.generate_queries = lambda c, ctx='': []
from app.llm_analyzer import AnalysisResult
llm_analyzer.analyze_claim = lambda claim, *a, **k: AnalysisResult(claim.claim_id, 'SUPPORTED', 0.9, [], [], 'ok', [], '')
p = pipeline.ClaimGuardPipeline()
res = p.run('sample_data/A_Unified_LLM_Based_Framework_for_Resume_Job_Description_Semantic_Alignment_Using_PuterJS___Springer.pdf', None)
print(res is not None)
