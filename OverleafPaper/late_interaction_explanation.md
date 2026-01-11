# Deep Dive: Late Interaction Path Ranking (with CWQ Example)

## The Core Problem
Traditional retrieval methods usually choose one of two extremes:
1.  **Bi-Encoders (Fast, Low detail):** Squash the whole question and the whole path into single vectors (one dot per text).
    *   *Issue:* You lose fine details. "Character" in a movie might be confused with "Character" in a book or game if the sentence is compressed.
2.  **Cross-Encoders (Slow, High detail):** Feed the question and path together into BERT.
    *   *Issue:* Extremely expensive. You have to run the heavy model for *every single* candidate path.

## The Solution: Late Interaction (MaxSim)

We use a "Goldilocks" approach called **Late Interaction** (inspired by ColBERT). It keeps the precision of word-by-word comparison but runs fast like a bi-encoder.

### Real-World Example from CWQ
We use a complex multi-constraint question (`WebQTest-415`):

*   **Question ($q$):** "What movie with film character named Mr. Woodson did Tupac star in?"
    *   *Note:* This question has TWO constraints: (1) Character "Mr. Woodson" appearing, AND (2) Tupac starring.
*   **Actual Data Path ($p$):**
    ```text
    (Mr. Woodson) --[film.film_character.portrayed_in_films]--> (m.03jps9y) --[film.performance.film]--> (Gridlock'd)
    ```
    *Note: `m.03jps9y` is an intermediate "Performance" node representing the specific acting role in that specific film.*

#### Step 1: Tokenize & Encode
Instead of squashing the sentence, we keep a vector for **every single token**.

*   **Question Tokens:** `['What', 'movie', 'with', 'film', 'character', 'named', 'Mr', '.', 'Woodson', 'Tupac', 'star', 'in', '?']`
*   **Path Tokens:** `['film', '.', 'film', '_', 'character', '.', 'portrayed', '_', 'in', '_', 'films', 'film', '.' 'performance', '.', 'film']`

#### Step 2: The "Interaction" Matrix
We compare **every** question token against **every** path token.

Imagine the question token **"character"**. We calculate its similarity score (dot product) with every token in the path:

| Path Token | Similarity Score w/ "character" | Interpretation |
| :--- | :--- | :--- |
| `film` | 0.3 | Broadly related (movie domain) |
| `portrayed` | 0.6 | Related action (characters get portrayed) |
| `performance` | 0.4 | Related concept (acting role) |
| `character` | **0.95** | **Exact Match!** (from `film_character`) |
| `films` | 0.35 | Broadly related (movie domain) |

#### Step 3: MaxSim (Find the Champion)
For the question token **"character"**, the model scans the whole path and picks the *winner*:
> Max Score for "character" = **0.95** (matches `character` from `film_character`)

Now typical "distractor" words like **"star"**:
> Max Score for "star" = **0.75** (matches `performance` or `portrayed`)
> *Even though "star" isn't in the path, the embedding for "performance" is semantically close to "starring role".*

#### Step 4: Total Scoring
We sum up the "Champion" scores.
$$ S(q, p) = 0.95 (\text{character}) + 0.75 (\text{star}) + 0.90 (\text{movie/film}) + \dots $$

**Why matches matter:**
Consider a WRONG path that connects Tupac to a movie via a song, which also appears in the dataset:
*   **Alternative Path:** `(Tupac) --[music.featured_artist.recordings]--> (Song) --[music.recording.releases]--> (Gridlock'd Soundtrack)`
*   When we score **"character"** against this path, the tokens are `music`, `artist`, `recordings`.
*   Score: "character" matches `artist` (0.3).
*   **Result:** The MaxSim score for "character" drops from **0.95** to **0.3**. The total path score decreases significantly, correctly rejecting the "Soundtrack" path because it lacks the "film character" concept required by the question.

![Late Interaction Diagram](late_interaction_diagram_1767851671038.png)

## Structural Awareness (Hop Embeddings)
We've added **Hop Embeddings** ($\mathbf{u}_k$):
*   `film.film_character.portrayed_in_films` gets a **[HOP 1]** tag.
*   `film.performance.film` gets a **[HOP 2]** tag.

This helps the model distinguish:
1.  `Character -> [PORTRAYED_IN] -> Performance` (Hop 1)
2.  `Performance -> [IS_FILM] -> Movie` (Hop 2)

This structure confirms the directionality: we are going FROM a character TO a movie.
