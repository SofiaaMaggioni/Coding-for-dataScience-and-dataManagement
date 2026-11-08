if __name__ == "__main__":
    raw_recipes, raw_interactions = load_raw_data()

    print(f"Righe ricette iniziali: {len(raw_recipes)}")
    print(f"Righe interazioni iniziali: {len(raw_interactions)}")

    # Ispezione valori mancanti
    inspect_missing_values(raw_recipes, "Recipes")
    inspect_missing_values(raw_interactions, "Interactions")

    # Pulizia
    clean_inter = clean_interactions(raw_interactions, drop_zeros=True)
    clean_rec = clean_recipes(raw_recipes)

    print(f"\nInterazioni scartate: {len(raw_interactions) - len(clean_inter)}")
    print(f"Interazioni valide finali: {len(clean_inter)}")
    print(f"Ricette valide finali: {len(clean_rec)}")








if __name__ == "__main__":
    raw_recipes, raw_interactions = load_raw_data()

    print(f"Righe ricette iniziali: {len(raw_recipes)}")
    print(f"Righe interazioni iniziali: {len(raw_interactions)}")

    # Ispezione valori mancanti
    inspect_missing_values(raw_recipes, "Recipes")
    inspect_missing_values(raw_interactions, "Interactions")

    # Pulizia
    clean_inter = clean_interactions(raw_interactions, drop_zeros=True)
    clean_rec = clean_recipes(raw_recipes)

    print(f"\nInterazioni scartate: {len(raw_interactions) - len(clean_inter)}")
    print(f"Interazioni valide finali: {len(clean_inter)}")
    print(f"Ricette valide finali: {len(clean_rec)}")
