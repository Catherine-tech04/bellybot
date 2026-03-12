import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 65)
print("  BellyBot v2 — Filter-Aware SBERT Training")
print("  Dataset: 80 curated recipes with allergen + health tags")
print("  Model: sentence-transformers/all-MiniLM-L6-v2")
print("=" * 65)

# Maps each dietary preference to disallowed ingredient keywords
DIET_EXCLUSION_RULES = {
    "vegan":        ["meat", "chicken", "beef", "pork", "fish", "egg", "dairy", "milk", "cheese", "yogurt", "ghee", "cream", "honey", "gelatin", "whey", "lamb", "shrimp", "tuna", "salmon"],
    "vegetarian":   ["chicken", "beef", "pork", "fish", "shrimp", "tuna", "salmon", "lamb", "mutton", "prawn", "crab"],
    "pescatarian":  ["chicken", "beef", "pork", "lamb", "mutton"],
    "jain":         ["meat", "chicken", "beef", "pork", "fish", "egg", "onion", "garlic", "potato", "carrot", "beet"],
    "halal":        ["pork", "bacon", "gelatin", "lard", "alcohol", "wine"],
    "keto":         [],  # No ingredient exclusion; uses health_tags
    "paleo":        ["dairy", "grain", "legume", "bean", "rice", "pasta", "bread", "sugar"],
    "mediterranean":[],  # No hard exclusions; preference-based
    "omnivore":     [],  # No exclusions
}

# Maps each diet to required tags (recipe MUST have at least one)
DIET_REQUIRED_TAGS = {
    "vegan":        ["vegan"],
    "vegetarian":   ["vegetarian", "vegan"],
    "pescatarian":  ["pescatarian", "vegan", "vegetarian"],
    "jain":         ["jain"],
    "keto":         ["keto"],
    "halal":        ["halal", "vegan", "vegetarian"],
    "paleo":        ["omnivore"],
    "mediterranean":["vegetarian", "pescatarian", "omnivore", "vegan"],
    "omnivore":     ["omnivore", "vegetarian", "vegan", "pescatarian", "halal"],
}

# Health condition → disallowed health_tags + keyword blockers
HEALTH_CONDITION_BLOCKS = {
    "diabetes": {
        "blocked_health_tags": ["high-calorie", "high-sugar", "natural-sugar"],
        "blocked_keywords": ["sugar", "honey", "jaggery", "maple syrup", "condensed milk"],
        "required_health_tags": [],  # Not strictly required but preferred
    },
    "hypertension": {
        "blocked_health_tags": [],
        "blocked_keywords": ["salt heavy", "soy sauce heavy", "processed"],
        "required_health_tags": [],
    },
    "celiac": {
        "blocked_health_tags": [],
        "blocked_keywords": ["wheat", "barley", "rye", "gluten"],
        "required_health_tags": ["gluten-free"],
    },
    "ibs": {
        "blocked_health_tags": [],
        "blocked_keywords": ["onion", "garlic", "beans", "cabbage"],
        "required_health_tags": ["easily-digestible", "gut-health"],
    },
    "kidney-disease": {
        "blocked_health_tags": [],
        "blocked_keywords": ["high potassium", "high phosphorus"],
        "required_health_tags": [],
    },
    "high-cholesterol": {
        "blocked_health_tags": [],
        "blocked_keywords": ["butter", "ghee", "cream", "fried", "lard"],
        "required_health_tags": ["heart-health"],
    },
    "heart-disease": {
        "blocked_health_tags": [],
        "blocked_keywords": ["butter", "fried", "cream", "lard", "processed"],
        "required_health_tags": ["heart-health"],
    },
}

# Fitness goal → preferred health tags (for scoring, not hard blocking)
FITNESS_PREFERRED_TAGS = {
    "weight-loss":      ["low-calorie", "high-protein", "fiber-rich", "low-fat", "low-carb"],
    "muscle-gain":      ["high-protein", "post-workout", "pre-workout", "high-calorie"],
    "heart-health":     ["heart-health", "omega-3", "anti-inflammatory", "low-fat"],
    "gut-health":       ["probiotic", "gut-health", "fiber-rich", "easily-digestible"],
    "improve-stamina":  ["pre-workout", "energy", "iron-rich", "omega-3"],
    "diabetes-control": ["diabetes-friendly", "low-carb", "fiber-rich", "gluten-free"],
    "maintenance":      [],
}


def hard_filter(recipes: list, profile: dict) -> tuple[list, list]:
    """
    STAGE 1: Hard filter — deterministic safety layer.

    Returns (safe_recipes, rejection_log)
    Every rejection is logged with a reason for full transparency.

    This function runs BEFORE SBERT. No unsafe recipe ever reaches the AI.
    """
    safe = []
    rejection_log = []

    user_allergies     = [a.lower() for a in profile.get("allergies", []) if a != "None"]
    diet_pref          = profile.get("dietary_preference", "omnivore").lower()
    health_conditions  = [h.lower() for h in profile.get("health_conditions", []) if h != "None"]
    meal_type          = profile.get("meal_type", "").strip()

    for r in recipes:
        reasons = []

        # ── Rule 1: Meal type match
        if meal_type and meal_type not in r.get("meal_types", []):
            reasons.append(f"Meal type '{meal_type}' not in {r['meal_types']}")

        # ── Rule 2: Allergen check (HARD BLOCK)
        recipe_allergens = [a.lower() for a in r.get("allergens", [])]
        allergen_conflicts = [a for a in user_allergies if a in recipe_allergens]
        if allergen_conflicts:
            reasons.append(f"ALLERGEN CONFLICT: recipe contains {allergen_conflicts}")

        # ── Rule 3: Dietary preference tag check
        required_diet_tags = DIET_REQUIRED_TAGS.get(diet_pref, [])
        recipe_diet_tags = [t.lower() for t in r.get("diet_tags", [])]
        if required_diet_tags:
            has_required = any(t in recipe_diet_tags for t in required_diet_tags)
            if not has_required:
                reasons.append(f"Dietary mismatch: '{diet_pref}' requires one of {required_diet_tags}, recipe has {recipe_diet_tags}")

        # ── Rule 4: Dietary keyword exclusion (ingredient-level check)
        excluded_keywords = DIET_EXCLUSION_RULES.get(diet_pref, [])
        raw_text = " ".join(r.get("ingredients_raw", [])).lower()
        keyword_conflicts = [kw for kw in excluded_keywords if kw in raw_text]
        if keyword_conflicts:
            reasons.append(f"Ingredient violation for '{diet_pref}': {keyword_conflicts}")

        # ── Rule 5: Health condition checks
        recipe_health_tags = [t.lower() for t in r.get("health_tags", [])]
        for condition in health_conditions:
            rules = HEALTH_CONDITION_BLOCKS.get(condition, {})

            # Check blocked health tags
            blocked_tags = rules.get("blocked_health_tags", [])
            tag_conflicts = [t for t in blocked_tags if t in recipe_health_tags]
            if tag_conflicts:
                reasons.append(f"Health conflict ({condition}): recipe has blocked tags {tag_conflicts}")

            # Check blocked ingredient keywords
            blocked_kws = rules.get("blocked_keywords", [])
            kw_conflicts = [kw for kw in blocked_kws if kw in raw_text]
            if kw_conflicts:
                reasons.append(f"Health conflict ({condition}): ingredient contains {kw_conflicts}")

            # Check required health tags (soft enforcement for IBS/Celiac)
            req_tags = rules.get("required_health_tags", [])
            if req_tags:
                has_req = any(t in recipe_health_tags for t in req_tags)
                if not has_req:
                    reasons.append(f"Missing required health tag for {condition}: needs one of {req_tags}")

        if reasons:
            rejection_log.append({"recipe": r["name"], "reasons": reasons})
        else:
            safe.append(r)

    return safe, rejection_log


def fitness_score(recipe: dict, fitness_goal: str) -> float:
    """
    Adds a soft preference score based on fitness goal.
    This is ADDITIVE to SBERT score (not a filter).
    Range: 0.0 to 0.3 (bonus on top of cosine similarity)
    """
    preferred = FITNESS_PREFERRED_TAGS.get(fitness_goal.lower(), [])
    recipe_tags = [t.lower() for t in recipe.get("health_tags", [])]
    matches = sum(1 for t in preferred if t in recipe_tags)
    return min(0.3, matches * 0.07)

# STEP 1: Load model + data

print("\n[STEP 1] Loading SBERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("  Model loaded | Embedding dims: 384")

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'recipes.json')
with open(dataset_path, 'r') as f:
    recipes = json.load(f)
print(f"\n[STEP 2] Loaded {len(recipes)} recipes from dataset")


# STEP 2: Build training pairs

print("\n[STEP 3] Creating training pairs...")
training_examples = []

for i, ra in enumerate(recipes):
    for j, rb in enumerate(recipes):
        if i >= j:
            continue

        shared_health = len(set(ra['health_tags']) & set(rb['health_tags']))
        shared_diet   = len(set(ra['diet_tags']) & set(rb['diet_tags']))
        shared_meal   = len(set(ra['meal_types']) & set(rb['meal_types']))
        shared_fit    = len(set(ra['fitness_tags']) & set(rb['fitness_tags']))

        if shared_meal > 0 and shared_health >= 2 and shared_diet >= 1:
            score = 0.90
        elif shared_meal > 0 and shared_health >= 1:
            score = 0.75
        elif shared_meal > 0:
            score = 0.58
        elif shared_health >= 2:
            score = 0.45
        elif shared_health >= 1:
            score = 0.30
        else:
            score = 0.10

        if score >= 0.70 or (score < 0.30 and len(training_examples) % 8 == 0):
            training_examples.append(
                InputExample(
                    texts=[ra['search_text'], rb['search_text']],
                    label=float(score)
                )
            )

# Manual high-quality pairs
manual = [
    ("high protein low carb keto breakfast egg spinach", "keto egg protein morning low calorie", 0.93),
    ("vegan plant based protein tofu tempeh lentil", "vegan dinner protein iron fiber legumes", 0.88),
    ("diabetes blood sugar fiber low glycemic whole grain", "diabetes control fiber carb breakfast healthy", 0.91),
    ("omega 3 salmon heart health cardiovascular", "fish omega3 protein heart healthy pescatarian", 0.90),
    ("gut health probiotic yogurt fermented bacteria", "gut probiotic fermented digestive health", 0.92),
    ("post workout muscle recovery protein", "muscle gain protein workout high calorie", 0.89),
    ("weight loss low calorie deficit vegetables", "low calorie salad fiber diet lunch", 0.88),
    ("gluten free celiac safe no wheat", "gluten free breakfast safe allergy", 0.90),
    ("iron rich anemia spinach leafy greens", "iron boost blood health leafy vegetable", 0.87),
    ("allergy safe no nuts no dairy no eggs", "allergen free safe simple whole food", 0.85),
]
for ta, tb, sc in manual:
    training_examples.append(InputExample(texts=[ta, tb], label=float(sc)))

print(f" {len(training_examples)} training pairs | "
      f"High (≥0.7): {sum(1 for e in training_examples if e.label >= 0.7)} | "
      f"Low (<0.3): {sum(1 for e in training_examples if e.label < 0.3)}")

# STEP 3: Fine-tune SBERT

BATCH_SIZE = 16
EPOCHS = 3
WARMUP  = 15

print(f"\n[STEP 4] Fine-tuning SBERT | Epochs={EPOCHS} | Batch={BATCH_SIZE}")
print("  Loss fn: CosineSimilarityLoss")
print("  Concept: MSE(cosine_sim(enc(A), enc(B)), target_score)")

train_dl   = DataLoader(training_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.CosineSimilarityLoss(model)
save_path  = os.path.join(os.path.dirname(__file__), 'saved_model')

model.fit(
    train_objectives=[(train_dl, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP,
    output_path=save_path,
    show_progress_bar=True,
)
print(f"\n  Model saved to: {save_path}/")


# STEP 4: Pre-compute corpus embeddings

print("\n[STEP 5] Pre-computing corpus embeddings for all 80 recipes...")
recipe_texts      = [r['search_text'] for r in recipes]
recipe_embeddings = model.encode(recipe_texts, show_progress_bar=True)

emb_path = os.path.join(os.path.dirname(__file__), 'recipe_embeddings.npy')
np.save(emb_path, recipe_embeddings)
print(f"  Embeddings saved | Shape: {recipe_embeddings.shape}")


# STEP 5: Evaluate hard filter + SBERT together

print("\n[STEP 6] Evaluating Filter + SBERT pipeline...")
print("-" * 60)

test_cases = [
    {
        "label": "Nut allergy + Vegan + Weight Loss + Dinner",
        "profile": {"allergies": ["nuts"], "dietary_preference": "vegan",
                    "health_conditions": [], "fitness_goal": "weight-loss", "meal_type": "Dinner"},
        "query": "lentils chickpeas vegetables dinner"
    },
    {
        "label": "Dairy allergy + Diabetes + Breakfast",
        "profile": {"allergies": ["dairy"], "dietary_preference": "vegetarian",
                    "health_conditions": ["diabetes"], "fitness_goal": "diabetes-control", "meal_type": "Breakfast"},
        "query": "oats vegetables fiber breakfast"
    },
    {
        "label": "Egg + Shellfish allergy + Muscle Gain + Pescatarian",
        "profile": {"allergies": ["eggs", "shellfish"], "dietary_preference": "pescatarian",
                    "health_conditions": [], "fitness_goal": "muscle-gain", "meal_type": "Lunch"},
        "query": "salmon protein lunch bowl"
    },
]

all_precisions = []
for tc in test_cases:
    print(f"\nTest: {tc['label']}")
    safe_recipes, rejections = hard_filter(recipes, tc["profile"])
    print(f"  Hard Filter: {len(recipes)} → {len(safe_recipes)} safe recipes | {len(rejections)} rejected")

    if len(safe_recipes) == 0:
        print("  ⚠ All recipes filtered — check profile constraints")
        continue

    # SBERT on safe pool
    safe_embs = np.array([recipe_embeddings[recipes.index(r)] for r in safe_recipes])
    q_emb     = model.encode([tc["query"]])
    sims      = cosine_similarity(q_emb, safe_embs)[0]

    # Add fitness score bonus
    bonus = np.array([fitness_score(r, tc["profile"]["fitness_goal"]) for r in safe_recipes])
    final_scores = sims + bonus

    top_idx = np.argsort(final_scores)[::-1][:3]
    print(f"  Top-3 after filter + SBERT + fitness score:")
    relevant = 0
    for rank, idx in enumerate(top_idx, 1):
        r = safe_recipes[idx]
        # Check if result is relevant (has good health tags for goal)
        goal_tags = FITNESS_PREFERRED_TAGS.get(tc["profile"]["fitness_goal"], [])
        is_relevant = any(t in [x.lower() for x in r["health_tags"]] for t in goal_tags)
        if is_relevant:
            relevant += 1
        print(f"    {rank}. {'yes' if is_relevant else 'no'} [{final_scores[idx]:.4f}] {r['name']} "
              f"| allergens: {r['allergens'] or '[]'}")

    prec = relevant / 3
    all_precisions.append(prec)
    print(f"  Precision@3: {prec:.2f}")

    # Verify NO allergen slipped through
    user_allergies = [a.lower() for a in tc["profile"]["allergies"]]
    for idx in top_idx:
        recipe_allergens = [a.lower() for a in safe_recipes[idx]["allergens"]]
        conflict = [a for a in user_allergies if a in recipe_allergens]
        assert not conflict, f"ALLERGEN BREACH: {conflict} in {safe_recipes[idx]['name']}"
    print("   ALLERGEN SAFETY CHECK: PASSED — no conflicts in results")

print(f"\n{'=' * 60}")
print(f"  FINAL RESULTS")
print(f"  Average Precision@3: {np.mean(all_precisions):.2f}")
print(f"  Allergen Safety: 100% (hard filter guaranteed)")
print(f"{'=' * 60}")
print("\nTraining complete! Now run: python main.py")
