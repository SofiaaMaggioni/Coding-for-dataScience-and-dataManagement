"""Functions for ranking recipes with a Bayesian lower bound."""

import numpy as np
import pandas as pd


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


def wilson_ranking(recipes_df, z=1.96):
    """
    Rank recipes by the Wilson lower bound for positive ratings.

    Ratings of 4 or 5 stars are considered positive; ratings from 1 to 3
    are considered non-positive. The default ``z`` gives a 95% confidence
    level for the Wilson interval.

    Parameters
    ----------
    recipes_df : pandas.DataFrame
        Recipes dataframe joined with the rating summary from
        :func:`src.analysis.add_rating_summary`.
    z : float, optional
        Normal-distribution multiplier. Defaults to 1.96 (95% confidence).

    Returns
    -------
    pandas.DataFrame
        Rated recipes sorted from highest to lowest Wilson lower bound.
    """
    rated_recipes = recipes_df.dropna(subset=["rating_count"]).copy()
    total_ratings = rated_recipes["rating_count"]
    positive_ratings = (
        rated_recipes["rating_4_count"] + rated_recipes["rating_5_count"]
    )
    positive_fraction = positive_ratings / total_ratings

    z_squared = z**2
    numerator = (
        positive_fraction
        + z_squared / (2 * total_ratings)
        - z
        * np.sqrt(
            positive_fraction * (1 - positive_fraction) / total_ratings
            + z_squared / (4 * total_ratings**2)
        )
    )
    denominator = 1 + z_squared / total_ratings

    rated_recipes["positive_rating_count"] = positive_ratings
    rated_recipes["positive_rating_fraction"] = positive_fraction
    rated_recipes["wilson_score"] = numerator / denominator

    return rated_recipes.sort_values("wilson_score", ascending=False)


def compare_top_rankings(rankings, top_n=10):
    """
    Compare recipe positions across ranking methods for their top results.

    The result contains every recipe appearing in the top ``top_n`` of at
    least one method, together with its position in each complete ranking.

    Parameters
    ----------
    rankings : dict
        Mapping of method names to ranked recipes dataframes. Each dataframe
        must contain an ``id`` and a ``name`` column.
    top_n : int, optional
        Number of top recipes to include from each method. Defaults to 10.

    Returns
    -------
    pandas.DataFrame
        Recipe names and their positions in the complete ranking for each
        method. Recipes outside a method's top ``top_n`` still show their
        position in that full ranking.
    """
    top_recipes = pd.concat(
        [ranking.head(top_n)[["id", "name"]] for ranking in rankings.values()]
    ).drop_duplicates(subset="id")

    comparison = top_recipes.copy()

    for method_name, ranking in rankings.items():
        method_ranks = ranking[["id"]].copy()
        method_ranks[f"{method_name}_rank"] = range(1, len(ranking) + 1)
        comparison = comparison.merge(method_ranks, on="id", how="left")

    return comparison


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
    wilson = wilson_ranking(recipes_with_ratings)
    ranking_comparison = compare_top_rankings(
        {
            "bayesian_lb": ranking,
            "bayesian_average": bayesian_average,
            "simple_average": average_ranking,
            "wilson": wilson,
        }
    )

    print("Top 10 recipes by Bayesian lower bound score:")
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

    print("\nTop 10 recipes by Wilson score (4-5 stars positive):")
    print(
        wilson[
            [
                "id",
                "name",
                "rating_count",
                "positive_rating_count",
                "positive_rating_fraction",
                "wilson_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nPositions of recipes in at least one top 10:")
    print(ranking_comparison.to_string(index=False))
