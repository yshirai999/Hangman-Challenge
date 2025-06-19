from agents.submission_agent import agent
from agents.agent import HangmanAgent
from agents.utils import HangmanTransformerModelV2, HangmanTransformerModelV3, data_path, model_path, checkpoint_path, results_path

if agent.mode != "Transformer":
    print(f"Current Agent mode: {agent.mode}")
    print("Switching to Transformer mode...")
    agent.mode = "Transformer"
    print(f"Switched Agent mode to: {agent.mode}")

# Ensure model is loaded and on GPU and run evaluation
if agent.TransformerModel==HangmanTransformerModelV3:
    m_path = checkpoint_path("v3_run/transformer_best.pt")
    log_path = results_path("logs/v3_eval/transformer_best.jsonl")
    agent.transformer_model = agent.load_transformer_model(
        HangmanTransformerModelV3,
        m_path,
        use_cuda=True,
    )
    agent.evaluate_transformer_prediction(
        use_blank_mask_for_transformer=True,
        eval_data_path=data_path("transformer_eval_data_from_subpattern_short.pkl"),
        log_path=log_path
    )
elif agent.TransformerModel==HangmanTransformerModelV2:
    print("Using HangmanTransformerModelV2")
    model_path = checkpoint_path("v2_run/transformer_best.pt")
    agent.transformer_model = agent.load_transformer_model(
        HangmanTransformerModelV2,
        model_path,
        use_cuda=True
    )
    agent.evaluate_transformer_prediction(
        eval_data_path=data_path("transformer_eval_data_from_subpattern.pkl")
    )

print(f"Evaluation completed for {model_path}.")