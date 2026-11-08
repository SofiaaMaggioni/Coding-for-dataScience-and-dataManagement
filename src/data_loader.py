import os
import pandas as pd


def load_raw_data(data_dir="data/raw"):
    """
    Carica i file CSV di ricette e interazioni.
    """
    recipes_path = os.path.join(data_dir, "RAW_recipes.csv")
    interactions_path = os.path.join(data_dir, "RAW_interactions.csv")

    if not os.path.exists(recipes_path) or not os.path.exists(interactions_path):
        raise FileNotFoundError(
            f"File non trovati in {data_dir}. Verificare la presenza dei file CSV."
        )

    print("Caricamento in corso...")
    recipes_df = pd.read_csv(recipes_path)
    interactions_df = pd.read_csv(interactions_path)

    return recipes_df, interactions_df


def inspect_missing_values(df, dataset_name="Dataset"):
    """
    Mostra il conteggio dei valori mancanti per ogni colonna.
    """
    missing_series = df.isna().sum()
    print(f"\n--- Valori mancanti in {dataset_name} ---")
    print(missing_series[missing_series > 0])
    return missing_series


def clean_interactions(df_interactions, drop_zeros=True):
    """
    Pulisce il dataframe delle recensioni:
    - Rimuove le righe con valori mancanti nelle colonne chiave
    - Converte la colonna rating a float
    - Esclude le recensioni con rating pari a 0
    """
    # 1. NaN
    clean_df = df_interactions.dropna(subset=["user_id", "recipe_id", "rating"]).copy()

    # 2. Conversione tipo
    clean_df["rating"] = clean_df["rating"].astype(float)

    # 3. Filtraggio booleano (esclusione rating = 0)
    if drop_zeros:
        clean_df = clean_df[clean_df["rating"] > 0]

    return clean_df


def clean_recipes(df_recipes):
    """
    Pulisce il dataframe delle ricette:
    - Rimuove righe con id o nome mancante
    - Converte colonne numeriche a float/int
    - Rimuove valori di preparazione non realistici
    """
    # Rimozione righe con valori nulli essenziali
    clean_df = df_recipes.dropna(subset=["id", "name", "minutes", "n_ingredients"]).copy()

    # Conversione tipo
    clean_df["minutes"] = clean_df["minutes"].astype(int)
    clean_df["n_ingredients"] = clean_df["n_ingredients"].astype(int)

    # Filtraggio ricette con tempi validi (maggiori di zero)
    clean_df = clean_df[clean_df["minutes"] > 0]

    return clean_df


if __name__ == "__main__":
    raw_recipes, raw_interactions = load_raw_data()

    print(f"Righe ricette iniziali: {len(raw_recipes)}")
    print(f"Righe interazioni iniziali: {len(raw_interactions)}")

    # 1. Valori non validi in Recipes
    print("\nValori mancanti ricette diviso per categoria:")
    recipes_missing = raw_recipes.isna().sum()
    recipes_issues = recipes_missing[recipes_missing > 0].copy()
    recipes_issues["minutes non validi (<= 0)"] = (raw_recipes["minutes"] <= 0).sum()
    print(recipes_issues)

    # 2. Valori non validi in Interactions
    print("\nValori mancanti interazioni diviso per categoria:")
    interactions_missing = raw_interactions.isna().sum()
    interactions_issues = interactions_missing[interactions_missing > 0].copy()
    interactions_issues["rating pari a 0"] = (raw_interactions["rating"] == 0).sum()
    print(interactions_issues)

    # Esecuzione pulizia
    clean_inter = clean_interactions(raw_interactions, drop_zeros=True)
    clean_rec = clean_recipes(raw_recipes)

    print(f"\nRicette valide finali: {len(clean_rec)}")
    print(f"Interazioni valide finali: {len(clean_inter)}")
