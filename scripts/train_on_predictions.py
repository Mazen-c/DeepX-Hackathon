"""Train local ABSA model weights based on LLM predictions."""

import argparse
import json
from pathlib import Path

import pandas as pd
from scripts.train_local_weights import train_and_save

def main():
    parser = argparse.ArgumentParser(description="Train model weights using predictions")
    parser.add_argument("--val-csv", default="Data/processed/val_clean.csv", help="Original validation CSV with review_clean")
    parser.add_argument("--predictions", default="Data/processed/val_predictions.json", help="Predictions JSON file")
    parser.add_argument("--weights", default="models/pseudo_labels_weights.joblib", help="Output weights file")
    parser.add_argument("--meta", default="models/pseudo_labels_weights.meta.json", help="Output metadata file")
    parser.add_argument("--output-csv", default="Data/processed/pseudo_labels_train.csv", help="Output CSV for the joined data")
    args = parser.parse_args()

    val_csv_path = Path(args.val_csv)
    pred_path = Path(args.predictions)
    weights_path = Path(args.weights)
    meta_path = Path(args.meta)
    output_csv_path = Path(args.output_csv)

    if not val_csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    # Load reviews
    df_val = pd.read_csv(val_csv_path, encoding="utf-8-sig")
    if "review_clean" not in df_val.columns and "review_text" in df_val.columns:
        df_val["review_clean"] = df_val["review_text"]
    
    # Ensure review_id is string for joining
    if "review_id" not in df_val.columns:
        df_val["review_id"] = df_val.index.astype(str)
    else:
        df_val["review_id"] = df_val["review_id"].astype(str)

    # Load predictions
    with pred_path.open("r", encoding="utf-8") as f:
        preds = json.load(f)

    # Build long format dataframe
    rows = []
    for p in preds:
        rid = str(p.get("review_id", ""))
        aspect_sentiments = p.get("aspect_sentiments", {})
        
        # Get the corresponding review text
        matching_row = df_val[df_val["review_id"] == rid]
        if matching_row.empty:
            continue
            
        review_clean = matching_row.iloc[0]["review_clean"]
        
        if not aspect_sentiments:
            rows.append({
                "review_id": rid,
                "review_clean": review_clean,
                "aspect": "general",
                "sentiment": "neutral"
            })
        else:
            for aspect, sentiment in aspect_sentiments.items():
                rows.append({
                    "review_id": rid,
                    "review_clean": review_clean,
                    "aspect": aspect,
                    "sentiment": sentiment
                })

    if not rows:
        raise ValueError("No matching rows found between CSV and predictions.")

    df_train = pd.DataFrame(rows)
    
    # Save the pseudo-labels CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Created pseudo-labels dataset with {len(df_train)} rows at {output_csv_path}")

    # Train and save weights
    print(f"Training new weights and saving to {weights_path}...")
    metadata = train_and_save(output_csv_path, weights_path, meta_path)
    print("Training complete. Metadata:")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
