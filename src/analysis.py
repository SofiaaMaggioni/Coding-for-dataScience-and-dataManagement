"""Functions for summarising recipe ratings."""


def summarize_recipe_ratings(interactions_df):
    """
    Calculate the number and mean of valid ratings for each recipe.

    Ratings equal to 0 represent reviews without a star rating and are
    excluded. The returned dataframe has one row per recipe with at least one
    valid rating.

    Parameters
    ----------
    interactions_df : pandas.DataFrame
        Interactions dataframe containing ``recipe_id`` and ``rating``.

    Returns
    -------
    pandas.DataFrame
        A dataframe with ``recipe_id``, ``rating_count`` and ``rating_mean``.
    """
    valid_ratings = interactions_df[interactions_df["rating"] > 0]

    rating_summary = valid_ratings.groupby("recipe_id")["rating"].agg(
        rating_count="count",
        rating_mean="mean",
    )

    return rating_summary.reset_index()


def add_rating_summary(recipes_df, rating_summary):
    """
    Add rating counts and means to the recipes dataframe.

    All recipes are retained. Recipes without valid ratings have missing
    values in ``rating_count`` and ``rating_mean``.

    Parameters
    ----------
    recipes_df : pandas.DataFrame
        Recipes dataframe containing an ``id`` column.
    rating_summary : pandas.DataFrame
        Output of :func:`summarize_recipe_ratings`.

    Returns
    -------
    pandas.DataFrame
        Recipes joined with their rating counts and means.
    """
    return recipes_df.merge(
        rating_summary,
        how="left",
        left_on="id",
        right_on="recipe_id",
    )


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    recipes, interactions = load_raw_data()
    rating_summary = summarize_recipe_ratings(interactions)
    recipes_with_ratings = add_rating_summary(recipes, rating_summary)

    rated_recipes = recipes_with_ratings.dropna(subset=["rating_mean"])
    top_recipes = rated_recipes.sort_values(
        "rating_mean", ascending=False
    ).head(10)

    print(f"Number of recipes: {len(recipes_with_ratings):,}")
    print(f"Recipes with valid ratings: {len(rated_recipes):,}")
    print(f"Recipes without valid ratings: {len(recipes_with_ratings) - len(rated_recipes):,}")
    print(f"Mean of all valid ratings: {interactions.loc[interactions['rating'] > 0, 'rating'].mean():.2f}")
    print("\nTop 10 recipes by simple average rating:")
    print(
        top_recipes[["id", "name", "rating_count", "rating_mean"]]
        .to_string(index=False)
    )
