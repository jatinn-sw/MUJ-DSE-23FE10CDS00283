from app import pipeline, web_search, academic_search, query_generator
web_search.search_web=lambda q,l=5: []
academic_search.search_academic_sources=lambda c,q: []
query_generator.generate_queries=lambda c,ctx='': []
res = pipeline.run_pipeline('sample_data/A_Unified_LLM_Based_Framework_for_Resume_Job_Description_Semantic_Alignment_Using_PuterJS___Springer.pdf', lambda s,m,p: None)
print(res is not None)
