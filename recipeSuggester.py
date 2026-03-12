"""
================================================================================
BellyBot v2 — FastAPI Backend (Filter-First Architecture)
================================================================================
"""

import json, os, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="BellyBot v2 API",
    description="Filter-First AI Recipe System: Hard Filter → SBERT → LLM",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DIET_REQUIRED_TAGS = {
    "vegan":         ["vegan"],
    "vegetarian":    ["vegetarian", "vegan"],
    "pescatarian":   ["pescatarian", "vegan", "vegetarian"],
    "jain":          ["jain"],
    "keto":          ["keto"],
    "halal":         ["halal", "vegan", "vegetarian"],
    "paleo":         ["omnivore"],
    "mediterranean": ["vegetarian", "pescatarian", "omnivore", "vegan"],
    "omnivore":      ["omnivore", "vegetarian", "vegan", "pescatarian", "halal"],
}

DIET_EXCLUSION_RULES = {
    "vegan":      ["meat","chicken","beef","pork","fish","egg","dairy","milk","cheese",
                   "yogurt","ghee","cream","honey","gelatin","whey","lamb","shrimp","tuna","salmon"],
    "vegetarian": ["chicken","beef","pork","fish","shrimp","tuna","salmon","lamb","mutton","prawn","crab"],
    "pescatarian":["chicken","beef","pork","lamb","mutton"],
    "jain":       ["meat","chicken","beef","pork","fish","egg","onion","garlic","potato","carrot","beet"],
    "halal":      ["pork","bacon","gelatin","lard","alcohol","wine"],
    "keto":       [],
    "paleo":      ["dairy","grain","legume","bean","rice","pasta","bread","sugar"],
    "mediterranean":[],
    "omnivore":   [],
}

HEALTH_CONDITION_BLOCKS = {
    "diabetes": {
        "blocked_health_tags": ["high-calorie", "high-sugar", "natural-sugar"],
        "blocked_keywords":    ["refined sugar", "jaggery", "maple syrup", "condensed milk"],
    },
    "hypertension": {
        "blocked_health_tags": [],
        "blocked_keywords":    [],
    },
    "celiac": {
        "blocked_health_tags": [],
        "blocked_keywords":    ["wheat", "barley", "rye"],
        "required_health_tags": ["gluten-free"],
    },
    "ibs": {
        "blocked_health_tags": [],
        "blocked_keywords":    [],
        "required_health_tags": ["easily-digestible", "gut-health"],
    },
    "high-cholesterol": {
        "blocked_health_tags": [],
        "blocked_keywords":    ["butter", "lard", "cream"],
    },
    "heart-disease": {
        "blocked_health_tags": [],
        "blocked_keywords":    ["butter", "lard", "cream", "fried"],
    },
    "kidney-disease": {
        "blocked_health_tags": [],
        "blocked_keywords":    [],
    },
    "pcos": {
        "blocked_health_tags": ["high-sugar"],
        "blocked_keywords":    ["sugar", "refined flour"],
    },
    "thyroid": {
        "blocked_health_tags": [],
        "blocked_keywords":    [],
    },
}

FITNESS_PREFERRED_TAGS = {
    "weight-loss":      ["low-calorie","high-protein","fiber-rich","low-fat","low-carb"],
    "muscle-gain":      ["high-protein","post-workout","pre-workout","high-calorie"],
    "heart-health":     ["heart-health","omega-3","anti-inflammatory","low-fat"],
    "gut-health":       ["probiotic","gut-health","fiber-rich","easily-digestible"],
    "improve-stamina":  ["pre-workout","energy","iron-rich","omega-3"],
    "diabetes-control": ["diabetes-friendly","low-carb","fiber-rich","gluten-free"],
    "maintenance":      [],
}

print("\n🚀 BellyBot v2 — Filter-First Architecture Starting...")
print("─" * 55)

MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'saved_model')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'recipes.json')
EMB_PATH     = os.path.join(os.path.dirname(__file__), 'recipe_embeddings.npy')

print("[1/3] Loading SBERT model...")
try:
    sbert = SentenceTransformer(MODEL_PATH if os.path.exists(MODEL_PATH) else
                                'sentence-transformers/all-MiniLM-L6-v2')
    print(" SBERT ready")
except Exception as e:
    raise RuntimeError(f"Model load failed: {e}")

print("[2/3] Loading dataset + embeddings...")
with open(DATASET_PATH) as f:
    RECIPES = json.load(f)

try:
    CORPUS_EMBEDDINGS = np.load(EMB_PATH)
    print(f" {len(RECIPES)} recipes | embeddings {CORPUS_EMBEDDINGS.shape}")
except FileNotFoundError:
    print("  ⚙ Computing embeddings...")
    CORPUS_EMBEDDINGS = sbert.encode([r['search_text'] for r in RECIPES], show_progress_bar=True)
    np.save(EMB_PATH, CORPUS_EMBEDDINGS)
    print(f"  Computed {CORPUS_EMBEDDINGS.shape}")

print("[3/3] Groq client...")
groq_client = Groq(api_key="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")
print("  Ready\n" + "─" * 55)
print("  API docs: http://localhost:8000/docs\n")


class HealthProfile(BaseModel):
    age:                  Optional[int]   = None
    gender:               Optional[str]   = None
    height_cm:            Optional[float] = None
    weight_kg:            Optional[float] = None
    activity_level:       Optional[str]   = None
    fitness_goal:         Optional[str]   = None
    dietary_preference:   Optional[str]   = None
    health_conditions:    List[str]       = []
    allergies:            List[str]       = []


class RecipeRequest(BaseModel):
    ingredients:  str
    meal_type:    str
    servings:     int = 2
    profile:      HealthProfile


class RejectedRecipe(BaseModel):
    name:    str
    reasons: List[str]


class NutritionInfo(BaseModel):
    calories: int
    protein:  str
    carbs:    str
    fats:     str
    fiber:    str


class Recipe(BaseModel):
    name:            str
    tagline:         str
    nutrition:       NutritionInfo
    prep_time:       str
    cook_time:       str
    difficulty:      str
    health_score:    int
    goal_alignment:  str
    ingredients:     List[str]
    steps:           List[str]
    ai_note:         str
    warnings:        List[str]
    sbert_score:     float
    fitness_bonus:   float


class RecipeResponse(BaseModel):
    recipes:            List[Recipe]
    sbert_top_matches:  List[str]
    rejection_log:      List[RejectedRecipe]
    safe_pool_size:     int
    total_corpus_size:  int
    pipeline_steps:     List[str]


def hard_filter(recipes: list, profile: HealthProfile, meal_type: str):
    safe       = []
    rejections = []

    user_allergies    = [a.lower() for a in profile.allergies if a.lower() != "none"]
    health_conditions = [h.lower() for h in profile.health_conditions if h.lower() != "none"]
    diet_pref         = (profile.dietary_preference or "omnivore").lower()

    for r in recipes:
        reasons = []
        raw_text          = " ".join(r.get("ingredients_raw", [])).lower()
        recipe_allergens  = [a.lower() for a in r.get("allergens", [])]
        recipe_diet_tags  = [t.lower() for t in r.get("diet_tags", [])]
        recipe_health_tags= [t.lower() for t in r.get("health_tags", [])]
        recipe_meal_types = r.get("meal_types", [])

        if meal_type and meal_type not in recipe_meal_types:
            reasons.append(f"Meal type '{meal_type}' ≠ recipe supports {recipe_meal_types}")

        conflicts = [a for a in user_allergies if a in recipe_allergens]
        if conflicts:
            reasons.append(f" ALLERGEN: {conflicts}")

        req_tags = DIET_REQUIRED_TAGS.get(diet_pref, [])
        if req_tags and not any(t in recipe_diet_tags for t in req_tags):
            reasons.append(f"Diet '{diet_pref}' needs tag in {req_tags}, recipe has {recipe_diet_tags}")

        blocked_kws = DIET_EXCLUSION_RULES.get(diet_pref, [])
        kw_hits = [kw for kw in blocked_kws if kw in raw_text]
        if kw_hits:
            reasons.append(f"Diet '{diet_pref}' blocks ingredients: {kw_hits}")

        for cond in health_conditions:
            rules = HEALTH_CONDITION_BLOCKS.get(cond, {})
            bad_tags = [t for t in rules.get("blocked_health_tags", []) if t in recipe_health_tags]
            if bad_tags:
                reasons.append(f"Condition '{cond}' blocks tags: {bad_tags}")
            bad_kws = [k for k in rules.get("blocked_keywords", []) if k in raw_text]
            if bad_kws:
                reasons.append(f"Condition '{cond}' blocks ingredients: {bad_kws}")
            req_ht = rules.get("required_health_tags", [])
            if req_ht and not any(t in recipe_health_tags for t in req_ht):
                reasons.append(f"Condition '{cond}' requires one of: {req_ht}")

        if reasons:
            rejections.append({"name": r["name"], "reasons": reasons})
        else:
            safe.append(r)

    return safe, rejections


def get_fitness_bonus(recipe: dict, fitness_goal: str) -> float:
    preferred = FITNESS_PREFERRED_TAGS.get((fitness_goal or "maintenance").lower(), [])
    tags = [t.lower() for t in recipe.get("health_tags", [])]
    return min(0.30, sum(0.07 for t in preferred if t in tags))


def sbert_rank(query: str, safe_recipes: list, fitness_goal: str) -> list:
    q_emb = sbert.encode([query])
    safe_indices = [RECIPES.index(r) for r in safe_recipes if r in RECIPES]
    safe_embs = CORPUS_EMBEDDINGS[safe_indices] if safe_indices else \
                sbert.encode([r['search_text'] for r in safe_recipes])
    sims  = cosine_similarity(q_emb, safe_embs)[0]
    bonus = np.array([get_fitness_bonus(r, fitness_goal) for r in safe_recipes])
    final = sims + bonus
    top_idx = np.argsort(final)[::-1][:3]
    return [(safe_recipes[i], float(sims[i]), float(bonus[i])) for i in top_idx]


def bmi_str(weight, height):
    if not weight or not height:
        return "unknown"
    b = weight / ((height / 100) ** 2)
    cat = ("Underweight" if b < 18.5 else "Normal" if b < 25 else "Overweight" if b < 30 else "Obese")
    return f"{b:.1f} ({cat})"


@app.get("/")
def root():
    return {
        "status": "running",
        "version": "2.0",
        "architecture": "HardFilter → SBERT → Groq LLM",
        "corpus_size": len(RECIPES),
        "embedding_dims": int(CORPUS_EMBEDDINGS.shape[1]),
    }


@app.post("/generate", response_model=RecipeResponse)
async def generate(request: RecipeRequest):
    steps = []

    steps.append(f"[1/3] Hard filter: checking {len(RECIPES)} recipes...")
    safe, rejections = hard_filter(RECIPES, request.profile, request.meal_type)
    steps.append(
        f"[1/3] Filter complete: {len(RECIPES) - len(safe)} rejected, "
        f"{len(safe)} safe | Allergens blocked: "
        f"{[a for a in request.profile.allergies if a.lower() != 'none']}"
    )

    if len(safe) == 0:
        raise HTTPException(
            status_code=400,
            detail="No recipes match your profile constraints. Try relaxing filters."
        )

    steps.append(f"[2/3] SBERT encoding query → cosine similarity on {len(safe)} safe recipes...")
    query = f"{request.meal_type} {request.ingredients} " \
            f"{request.profile.dietary_preference or ''} {request.profile.fitness_goal or ''}"

    top3 = sbert_rank(query, safe, request.profile.fitness_goal or "maintenance")
    steps.append(
        f"[2/3] Top SBERT match: '{top3[0][0]['name']}' "
        f"(cosine: {top3[0][1]:.4f}, fitness bonus: +{top3[0][2]:.4f})"
    )

    profile = request.profile
    sbert_context = "\n".join([
        f"  {i+1}. [sim:{sc:.3f} + bonus:{bn:.3f}] {r['name']} "
        f"| allergens:{r['allergens'] or 'none'} "
        f"| health:{r['health_tags'][:3]}"
        for i, (r, sc, bn) in enumerate(top3)
    ])

    system_prompt = """You are BellyBot v2, an expert AI nutritionist using a filter-first architecture.
CRITICAL: All recipes returned to you have ALREADY passed hard allergen + dietary + health filtering.
Your job is to create personalized recipes using the retrieved context and the user's full health profile.
Return ONLY valid JSON, no markdown, no preamble.

JSON schema:
{
  "recipes": [
    {
      "name": "string",
      "tagline": "string",
      "nutrition": {"calories": int, "protein": "Xg", "carbs": "Xg", "fats": "Xg", "fiber": "Xg"},
      "prep_time": "string",
      "cook_time": "string",
      "difficulty": "Easy|Medium|Hard",
      "health_score": int (0-100),
      "goal_alignment": "string",
      "ingredients": ["string"],
      "steps": ["string"],
      "ai_note": "string",
      "warnings": ["string"]
    }
  ]
}"""

    user_prompt = f"""## SBERT Semantic Retrieval (Stage 2 — Safe Pool RAG Context):
{sbert_context}

## User Health Profile:
Age: {profile.age} | Gender: {profile.gender} | BMI: {bmi_str(profile.weight_kg, profile.height_cm)}
Activity: {profile.activity_level} | Goal: {profile.fitness_goal}
Diet: {profile.dietary_preference} | Conditions: {', '.join(profile.health_conditions) or 'None'}
ALLERGIES (HARD BLOCKED): {', '.join(profile.allergies) or 'None'}

## Dynamic Request:
Meal Type: {request.meal_type} | Servings: {request.servings}
Available Ingredients: {request.ingredients}

Generate exactly 3 personalized {request.meal_type} recipes.
Use ONLY available ingredients + pantry staples (water, salt, basic oil, minimal spices).
All allergens listed above are ALREADY guaranteed excluded — confirm this in your ai_note.
Adapt nutrition to fitness goal: {profile.fitness_goal}."""

    steps.append("[3/3] Groq LLM generating recipes with SBERT RAG context + health profile...")

    msg = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2800,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    raw     = msg.choices[0].message.content
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    parsed  = json.loads(cleaned)

    for i, recipe in enumerate(parsed["recipes"]):
        r, sc, bn = top3[min(i, len(top3)-1)]
        recipe["sbert_score"]   = sc
        recipe["fitness_bonus"] = bn

    steps.append(f" {len(parsed['recipes'])} recipes generated | "
                 f"Health scores: {[r['health_score'] for r in parsed['recipes']]}")

    return RecipeResponse(
        recipes=parsed["recipes"],
        sbert_top_matches=[
            f"{r['name']} (sim:{sc:.3f}, bonus:{bn:.3f})"
            for r, sc, bn in top3
        ],
        rejection_log=[RejectedRecipe(**rj) for rj in rejections],
        safe_pool_size=len(safe),
        total_corpus_size=len(RECIPES),
        pipeline_steps=steps,
    )


@app.post("/filter/preview")
async def filter_preview(request: RecipeRequest):
    safe, rejections = hard_filter(RECIPES, request.profile, request.meal_type)
    return {
        "total": len(RECIPES),
        "safe":  len(safe),
        "rejected": len(rejections),
        "rejection_log": rejections[:10],
        "safe_recipes": [{"name": r["name"], "allergens": r["allergens"],
                          "diet_tags": r["diet_tags"]} for r in safe[:10]],
    }


@app.get("/sbert/similarity")
def similarity(text_a: str, text_b: str):
    a, b  = sbert.encode([text_a, text_b])
    score = float(cosine_similarity([a], [b])[0][0])
    return {
        "text_a": text_a, "text_b": text_b,
        "cosine_similarity": round(score, 4),
        "interpretation": ("Very similar" if score > 0.8 else "Similar" if score > 0.6
                           else "Somewhat related" if score > 0.4 else "Different"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
