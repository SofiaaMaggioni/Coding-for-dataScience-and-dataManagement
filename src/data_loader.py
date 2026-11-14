import os
import pandas as pd


# Build the default data paths from this file's location.
# This lets the script work even when it is launched outside the project folder.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
DEFAULT_CLEAN_DIR = os.path.join(PROJECT_DIR, "data", "clean")


def load_raw_data(data_dir=DEFAULT_RAW_DIR):
    """
    Load the CSV files containing recipes and interactions.
    """
    recipes_path = os.path.join(data_dir, "RAW_recipes.csv")
    interactions_path = os.path.join(data_dir, "RAW_interactions.csv")

    if not os.path.exists(recipes_path) or not os.path.exists(interactions_path):
        raise FileNotFoundError(
            f"CSV files not found in {data_dir}. Check that both files are present."
        )

    print("Loading data...")
    recipes_df = pd.read_csv(recipes_path)
    interactions_df = pd.read_csv(interactions_path)

    return recipes_df, interactions_df


def inspect_missing_values(df, dataset_name="Dataset"):
    """
    Count the missing values in each column.
    """
    missing_series = df.isna().sum()
    print(f"\n--- Missing values in {dataset_name} ---")
    print(missing_series[missing_series > 0])
    return missing_series


def clean_interactions(df_interactions):
    """
    Clean the interactions dataframe.

    Rows with a rating of 0 are reviews without a star rating, so they are
    excluded from the rating analysis.
    """
    clean_df = df_interactions.dropna(
        subset=["user_id", "recipe_id", "rating"]
    ).copy()

    clean_df["rating"] = clean_df["rating"].astype(float)

    clean_df = clean_df[clean_df["rating"] > 0]

    return clean_df


def clean_recipes(df_recipes):
    """
    Clean the recipes dataframe.

    Recipes without ratings are kept here. They can be excluded later when
    creating a ranking based on ratings.
    """
    clean_df = df_recipes.dropna(
        subset=["id", "name", "minutes", "n_ingredients"]
    ).copy()

    clean_df["minutes"] = clean_df["minutes"].astype(int)
    clean_df["n_ingredients"] = clean_df["n_ingredients"].astype(int)

    clean_df = clean_df[clean_df["minutes"] > 0]

    return clean_df


def save_clean_data(df_recipes, df_interactions, output_dir=DEFAULT_CLEAN_DIR):
    """
    Save the cleaned dataframes as CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    recipes_out = os.path.join(output_dir, "clean_recipes.csv")
    interactions_out = os.path.join(output_dir, "clean_interactions.csv")

    print(f"\nSaving cleaned files in {output_dir}...")
    df_recipes.to_csv(recipes_out, index=False)
    df_interactions.to_csv(interactions_out, index=False)
    print("Files saved!")


if __name__ == "__main__":
    raw_recipes, raw_interactions = load_raw_data()

    print(f"Initial recipes: {len(raw_recipes)}")
    print(f"Initial interactions: {len(raw_interactions)}")

    # Show missing values and recipes with invalid preparation times.
    inspect_missing_values(raw_recipes, "Recipes")
    print(
        "Recipes with invalid preparation time (<= 0):",
        (raw_recipes["minutes"] <= 0).sum(),
    )

    # Show missing values and interactions without a star rating.
    inspect_missing_values(raw_interactions, "Interactions")
    print(
        "Interactions without a star rating (rating = 0):",
        (raw_interactions["rating"] == 0).sum(),
    )

    clean_inter = clean_interactions(raw_interactions)
    clean_rec = clean_recipes(raw_recipes)

    print(f"\nValid recipes: {len(clean_rec)}")
    print(f"Rated interactions: {len(clean_inter)}")

    save_clean_data(clean_rec, clean_inter)
