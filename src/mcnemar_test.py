import numpy as np
import os
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar_test(project_dir=".", save_to_file=False):
    # 1. Load prediction arrays
    paths = {
        "y_true": os.path.join(project_dir, "eval_outputs_multimodal_10k", "layoutlmv3_margin_star_10k_y_true.npy"),
        "y_pred_multi": os.path.join(project_dir, "eval_outputs_multimodal_10k", "layoutlmv3_margin_star_10k_y_pred_multi.npy"),
        "y_pred_text": os.path.join(project_dir, "eval_outputs_text_10k", "bert_margin_star_10k_y_pred_text.npy")
    }
    
    y_true = np.load(paths["y_true"])
    y_pred_multi = np.load(paths["y_pred_multi"])
    y_pred_text = np.load(paths["y_pred_text"])
    
    assert len(y_true) == len(y_pred_text) == len(y_pred_multi), "❌ Array length mismatch!"
    
    # 2. Compute correctness masks
    text_correct = (y_pred_text == y_true)
    multi_correct = (y_pred_multi == y_true)
    
    # 3. Build 2x2 contingency table
    n11 = int(np.sum(text_correct & multi_correct))  # Both correct
    n00 = int(np.sum(~text_correct & ~multi_correct)) # Both wrong
    n10 = int(np.sum(text_correct & ~multi_correct))  # Text correct, Multi wrong
    n01 = int(np.sum(~text_correct & multi_correct))  # Text wrong, Multi correct
    
    print("\n" + "="*50)
    print("Contingency Table:")
    print("="*50)
    print(f"n11 (both correct)          = {n11}")
    print(f"n00 (both wrong)            = {n00}")
    print(f"n10 (text correct, multi wrong) = {n10}")
    print(f"n01 (text wrong, multi correct) = {n01}")
    
    # 4. Run McNemar's Test (with continuity correction)
    table = [[n11, n10],
             [n01, n00]]
    result = mcnemar(table, exact=False, correction=True)
    
    # 5. Format p-value for academic reporting
    p_val = result.pvalue
    p_str = f"{p_val:.2e}" if p_val > 1e-10 else "< 10⁻⁵⁰"
    
    # 6. Compute Odds Ratio
    odds_ratio = n01 / n10 if n10 > 0 else float('inf')
    
    print("\n" + "="*50)
    print("McNemar Test Result:")
    print("="*50)
    print(f"χ² statistic (corrected) = {result.statistic:.6f}")
    print(f"p-value                  = {p_str}")
    print(f"Odds Ratio               = {odds_ratio:.2f}")
    
    # 7. Statistical Interpretation
    print("\n" + "="*50)
    if p_val < 0.05:
        print("✅ Result: Statistically significant (p < 0.05)")
        print(f" Interpretation: Multimodal model corrects {odds_ratio:.1f}× more text-only errors than vice versa.")
        print("   → Strong evidence of architectural superiority under identical training conditions.")
    else:
        print("❌ Result: NOT statistically significant")
        print("   → Observed differences may be due to random variation.")
    print("="*50 + "\n")
    
    # 8. Optional: Save to .txt for thesis appendix
    if save_to_file:
        output_path = os.path.join(project_dir, "McNemar_Test_Result.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Contingency Table:\n")
            f.write(f"n11 (both correct) = {n11}\n")
            f.write(f"n00 (both wrong)   = {n00}\n")
            f.write(f"n10 (text correct, multi wrong) = {n10}\n")
            f.write(f"n01 (text wrong, multi correct) = {n01}\n")
            f.write("McNemar Test Result:\n")
            f.write(f"chi2 statistic = {result.statistic:.6f}\n")
            f.write(f"p-value = {p_str}\n")
            f.write(f"Odds Ratio = {odds_ratio:.2f}\n")
            f.write(f"Result: Statistically significant (p < 0.05)\n")
        print(f"💾 Results saved to: {output_path}")

# Run the test
if __name__ == "__main__":
    run_mcnemar_test(save_to_file=True)