"""Functions for ranking recipes with a Bayesian lower bound."""

import numpy as np


def simple_average_ranking(recipes_df):
    """
    Rank recipes by their simple average rating.

    Recipes without valid ratings are excluded because they have no observed
    average rating.

    Parameters
    ----------
    recipes_df : pandas.DataFrame
        Recipes dataframe joined with the rating summary from
        :func:`src.analysis.add_rating_summary`.

    Returns
    -------
    pandas.DataFrame
        Rated recipes sorted from highest to lowest simple average.
    """
    rated_recipes = recipes_df.dropna(subset=["rating_mean"]).copy()
    return rated_recipes.sort_values("rating_mean", ascending=False)


def bayesian_average_ranking(recipes_df, global_mean, m=25):
    """
    Rank recipes by a Bayesian average shrunk toward the global mean.

    The parameter ``m`` represents the weight of the global mean, expressed
    as an equivalent number of ratings.

    Parameters
    ----------
    recipes_df : pandas.DataFrame
        Recipes dataframe joined with the rating summary from
        :func:`src.analysis.add_rating_summary`.
    global_mean : float
        Mean of all valid ratings.
    m : int, optional
        Weight assigned to the global mean. Defaults to 25.

    Returns
    -------
    pandas.DataFrame
        Rated recipes sorted from highest to lowest Bayesian average.
    """
    rated_recipes = recipes_df.dropna(subset=["rating_mean"]).copy()
    rating_count = rated_recipes["rating_count"]

    rated_recipes["bayesian_average_score"] = (
        rating_count * rated_recipes["rating_mean"] + m * global_mean
    ) / (rating_count + m)

    return rated_recipes.sort_values(
        "bayesian_average_score", ascending=False
    )


def bayesian_lb_ranking(recipes_df, z=1.65):
    """
    Rank rated recipes by a Bayesian lower bound for their mean rating.

    The calculation uses the number of ratings at each level from 1 to 5.
    A uniform prior adds one hypothetical rating at every level. The score is
    the estimated mean minus ``z`` times its estimated uncertainty.

    Parameters
    ----------
    recipes_df : pandas.DataFrame
        Recipes dataframe joined with the rating summary from
        :func:`src.analysis.add_rating_summary`.
    z : float, optional
        Multiplier controlling how conservative the lower bound is. The
        default value 1.65 follows Miller's approximate ranking method.

    Returns
    -------
    pandas.DataFrame
        Rated recipes sorted from highest to lowest Bayesian score.
    """
    rated_recipes = recipes_df.dropna(subset=["rating_count"]).copy()
    rating_count = rated_recipes["rating_count"]
    prior_total = rating_count + 5

    rating_mean = 0
    rating_second_moment = 0

    for rating in range(1, 6):
        rating_count_column = rated_recipes[f"rating_{rating}_count"] + 1
        rating_mean += rating * rating_count_column / prior_total
        rating_second_moment += rating**2 * rating_count_column / prior_total

    variance = (rating_second_moment - rating_mean**2) / (rating_count + 6)
    uncertainty = np.sqrt(variance)

    rated_recipes["bayesian_score"] = rating_mean - z * uncertainty

    return rated_recipes.sort_values("bayesian_score", ascending=False)


if __name__ == "__main__":
    from src.analysis import add_rating_summary, summarize_recipe_ratings
    from src.data_loader import load_raw_data

    recipes, interactions = load_raw_data()
    rating_summary = summarize_recipe_ratings(interactions)
    recipes_with_ratings = add_rating_summary(recipes, rating_summary)
    valid_ratings = interactions.loc[interactions["rating"] > 0, "rating"]
    global_mean = valid_ratings.mean()

    ranking = bayesian_lb_ranking(recipes_with_ratings)
    bayesian_average = bayesian_average_ranking(
        recipes_with_ratings, global_mean
    )
    average_ranking = simple_average_ranking(recipes_with_ratings)

    print("Top 10 recipes by Bayesian score:")
    print(
        ranking[["id", "name", "rating_count", "rating_mean", "bayesian_score"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 recipes by Bayesian average (m=25):")
    print(
        bayesian_average[
            ["id", "name", "rating_count", "rating_mean", "bayesian_average_score"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop 10 recipes by simple average rating:")
    print(
        average_ranking[["id", "name", "rating_count", "rating_mean"]]
        .head(10)
        .to_string(index=False)
    )
