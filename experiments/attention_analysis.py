"""
This file contains the code that uses the output from pure inference, and then runs the same input over the model again (without generation) to compute attentions that the model saw during generation
"""

from utils.plotter import Plotter
from utils.prompt_llava import infer_model_with_attention

def attention_distribution_between_text_and_visual(levels):
    """
    This function computes the attention distribution placed from last_token to visual tokens vs text tokens
    """

    # These are the source -> target combinations of interest for us and hence we will store the mean for these
    key_pairs = [
                    ("last", "visual_subject"),
                    ("last", "visual_object"),
                    ("relation", "visual_subject"),
                    ("relation", "visual_object"),
                    ("last", "all_text"),
                    ("last", "all_visual")
    ]

    attn_results = infer_model_with_attention(levels, key_pairs, "visual")

    if not attn_results:
        return

    plotter = Plotter(experiment_name="attention_analysis")

    # Always save raw attention results to JSON via Plotter
    plotter.save_jsonl(attn_results, "attention_results_detailed.json")

    # Once I write down the json, I will then analyse them on a notebook
    # # 1) Layer-wise text vs image fractions averaged over all QAs
    # all_metrics = [r["attention_metrics"] for r in attn_results]
    # agg_metrics = plotter.aggregate_attention_metrics(all_metrics)
    # plotter.plot_layer_text_vs_image(agg_metrics)

    # # 2) Per‑relation and global head×layer grids
    # #    (plot_attention_by_relation expects the full attn_results list)
    # plotter.plot_attention_by_relation(attn_results)


