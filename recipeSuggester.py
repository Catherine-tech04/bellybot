import pandas as pd
import ast
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")

recipes = pd.read_csv("RAW_recipes.csv")

# Convert ingredients and tags from string to list
recipes["ingredients"] = recipes["ingredients"].apply(ast.literal_eval)
recipes["tags"] = recipes["tags"].apply(ast.literal_eval)

# Lowercase everything
recipes["ingredients"] = recipes["ingredients"].apply(
    lambda x: [i.lower() for i in x]
)
recipes["tags"] = recipes["tags"].apply(
    lambda x: [t.lower() for t in x]
)

# Convert ingredient list to single string
recipes["ingredients_clean"] = recipes["ingredients"].apply(
    lambda x: " ".join(x)
)

print("Dataset loaded. Total recipes:", len(recipes))

# -------------------------------
# 🔥 LOAD TRANSFORMER MODEL
# -------------------------------

print("Loading AI model (Sentence-BERT)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 📦 LOAD OR CREATE EMBEDDINGS
# -------------------------------

embedding_file = "recipe_embeddings.npy"

if os.path.exists(embedding_file):
    print("Loading saved embeddings...")
    recipe_embeddings = np.load(embedding_file)
else:
    print("Creating embeddings (first time only, may take time)...")
    recipe_embeddings = model.encode(
        recipes["ingredients_clean"].tolist(),
        show_progress_bar=True
    )
    np.save(embedding_file, recipe_embeddings)

print("Embeddings ready.")

# -------------------------------
# 🧠 LOAD USER PROFILE (RL MEMORY)
# -------------------------------

profile_file = "user_profile.npy"
pref_file = "user_preferences.json"

if os.path.exists(profile_file):
    user_profile_embedding = np.load(profile_file)
else:
    user_profile_embedding = None

if os.path.exists(pref_file):
    with open(pref_file, "r") as f:
        user_preferences = json.load(f)
else:
    user_preferences = {
        "ingredients": {},
        "tags": {}
    }

# -------------------------------
# 🔎 RECOMMEND FUNCTION
# -------------------------------

def recommend_by_ingredients(user_ingredients,
                             dietary_restriction=None,
                             meal_type=None,
                             top_n=5):

    user_text = " ".join(user_ingredients)
    user_embedding = model.encode([user_text])

    # STEP 1: At least one ingredient must match
    filtered_recipes = recipes[
        recipes["ingredients"].apply(
            lambda ing: any(
                user_ing in ingredient
                for ingredient in ing
                for user_ing in user_ingredients
            )
        )
    ]

    if filtered_recipes.empty:
        return "No recipes contain those ingredients."

    # STEP 2: Dietary filter
    if dietary_restriction:
        filtered_recipes = filtered_recipes[
            filtered_recipes["tags"].apply(
                lambda tags: dietary_restriction in tags
            )
        ]

    # STEP 3: Meal type filter
    if meal_type:
        filtered_recipes = filtered_recipes[
            filtered_recipes["tags"].apply(
                lambda tags: meal_type in tags
            )
        ]

    if filtered_recipes.empty:
        return "No recipes found after applying filters."

    # STEP 4: Count ingredient overlap
    filtered_recipes = filtered_recipes.copy()

    filtered_recipes["ingredient_match_count"] = filtered_recipes["ingredients"].apply(
        lambda ing: sum(
            any(user_ing in ingredient for ingredient in ing)
            for user_ing in user_ingredients
        )
    )

    filtered_indices = filtered_recipes.index
    filtered_embeddings = recipe_embeddings[filtered_indices]

    # STEP 5: Embedding similarity
    similarity_scores = cosine_similarity(
        user_embedding,
        filtered_embeddings
    ).flatten()

    filtered_recipes["similarity"] = similarity_scores

    # STEP 6: Base scoring
    filtered_recipes["final_score"] = (
        0.6 * filtered_recipes["similarity"] +
        0.4 * (filtered_recipes["ingredient_match_count"] / len(user_ingredients))
    )

    # STEP 7: Reinforcement Learning Boost
    global user_profile_embedding
    global user_preferences

    if user_profile_embedding is not None:
        profile_similarity = cosine_similarity(
            user_profile_embedding.reshape(1, -1),
            filtered_embeddings
        ).flatten()

        filtered_recipes["final_score"] += 0.3 * profile_similarity

    # Ingredient preference boost
    for ingredient, weight in user_preferences["ingredients"].items():
        filtered_recipes["final_score"] += filtered_recipes["ingredients"].apply(
            lambda ing: 0.05 * weight if ingredient in ing else 0
        )

    # Tag preference boost
    for tag, weight in user_preferences["tags"].items():
        filtered_recipes["final_score"] += filtered_recipes["tags"].apply(
            lambda t: 0.05 * weight if tag in t else 0
        )

    filtered_recipes = filtered_recipes.sort_values(
        by="final_score",
        ascending=False
    )

    return filtered_recipes.head(top_n)

# -------------------------------
# 🔁 INTERACTIVE LOOP
# -------------------------------

while True:

    print("\n-----------------------------------")

    user_input = input("Enter ingredients separated by commas: ")
    diet_input = input("Enter dietary restriction (or press Enter for none): ")
    meal_input = input("Enter meal type (dessert, breakfast, dinner, etc) or press Enter for none: ")

    user_ingredients = [item.strip().lower() for item in user_input.split(",")]

    if diet_input.strip() == "":
        diet_input = None

    if meal_input.strip() == "":
        meal_input = None

    results = recommend_by_ingredients(
        user_ingredients,
        dietary_restriction=diet_input,
        meal_type=meal_input,
        top_n=5
    )

    print("\nRecommended Recipes:")

    if isinstance(results, str):
        print(results)
    else:
        print(results[["name", "final_score"]])

        # Reinforcement learning feedback
        like_choice = input("\nEnter recipe number you liked (1-5) or press Enter to skip: ")

        if like_choice.strip() != "":
            try:
                index = int(like_choice) - 1
                liked_recipe = results.iloc[index]
                recipe_index = liked_recipe.name

                print("Learning from your choice...")

                liked_embedding = recipe_embeddings[recipe_index]

                if user_profile_embedding is None:
                    user_profile_embedding = liked_embedding
                else:
                    user_profile_embedding = (
                        0.7 * user_profile_embedding +
                        0.3 * liked_embedding
                    )

                np.save(profile_file, user_profile_embedding)

                # Update ingredient preference
                for ing in recipes.loc[recipe_index]["ingredients"]:
                    user_preferences["ingredients"][ing] = \
                        user_preferences["ingredients"].get(ing, 0) + 1

                # Update tag preference
                for tag in recipes.loc[recipe_index]["tags"]:
                    user_preferences["tags"][tag] = \
                        user_preferences["tags"].get(tag, 0) + 1

                with open(pref_file, "w") as f:
                    json.dump(user_preferences, f)

                print("Preference updated! 🎯")

            except:
                print("Invalid selection.")

    again = input("\nDo you want another recommendation? (yes/no): ").lower()
    if again != "yes":
        print("\nThank you for using the Adaptive AI Recipe Recommender!")
        break
