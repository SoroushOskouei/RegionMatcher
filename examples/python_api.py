from region_matcher import InferenceOptions, run_inference

results = run_inference(
    InferenceOptions(
        checkpoint="models/model.pt",
        query="data/queries/query.jpg",
        gallery="data/gallery",
        output="outputs/example_run",
        top_k=5,
        device=None,  # Automatically selects CUDA when available.
    )
)

for rank, result in enumerate(results, start=1):
    print(rank, result.path, result.score)
