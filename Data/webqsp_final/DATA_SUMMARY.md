# WebQSP Final Dataset Summary

## Overview

The `webqsp_final` dataset contains question-answering data based on the WebQuestionsSP (WebQSP) dataset, enriched with knowledge graph subgraphs and relation paths. The dataset is designed for training and evaluating knowledge graph-based question answering models.

## Dataset Statistics

### File Sizes
- **train.jsonl**: 940.11 MB
- **val.jsonl**: 79.48 MB  
- **test.jsonl**: 549.32 MB
- **train.parquet**: 252 MB
- **val.parquet**: 22 MB
- **test.parquet**: 132 MB
- **vocab.json**: 43 MB

### Dataset Splits
- **Training set**: 2,826 samples (parquet format)
- **Validation set**: Available in both JSONL and Parquet formats
- **Test set**: Available in both JSONL and Parquet formats

## Data Formats

### 1. JSONL Format (`train.jsonl`, `val.jsonl`, `test.jsonl`)

Each line in the JSONL files is a JSON object representing a single question-answer pair with associated knowledge graph information.

#### Structure:
```json
{
  "id": "WebQTrn-0",
  "question": "what is the name of justin bieber brother",
  "answer": ["Jaxon Bieber"],
  "q_entity": ["Justin Bieber"],
  "a_entity": ["Jaxon Bieber"],
  "graph": [
    ["entity1", "relation1", "entity2"],
    ["entity2", "relation2", "entity3"],
    ...
  ],
  "paths": [
    {
      "full_path": "(Justin Bieber) --[people.person.parents]--> (Jeremy Bieber) --[people.person.children]--> (Jaxon Bieber)",
      "relation_chain": "people.person.parents -> people.person.children",
      "entities": ["Justin Bieber", "Jeremy Bieber", "Jaxon Bieber"],
      "relations": ["people.person.parents", "people.person.children"]
    },
    ...
  ]
}
```

#### Field Descriptions:
- **id**: Unique identifier for the question (e.g., "WebQTrn-0")
- **question**: Natural language question string
- **answer**: List of answer entities (can be multiple answers)
- **q_entity**: List of question entities (entities mentioned in the question)
- **a_entity**: List of answer entities (same as answer, but in entity format)
- **graph**: List of knowledge graph triples `[subject, relation, object]` extracted from the subgraph relevant to the question
  - Each triple is represented as `[head_entity, relation, tail_entity]`
  - Typical subgraph contains thousands of triples (e.g., ~9,000 triples per question)
- **paths**: List of reasoning paths from question entity to answer entity
  - Each path contains:
    - `full_path`: Human-readable path representation
    - `relation_chain`: Sequence of relations in the path
    - `entities`: Ordered list of entities in the path
    - `relations`: Ordered list of relations in the path

### 2. Parquet Format (`train.parquet`, `val.parquet`, `test.parquet`)

The Parquet files contain the same data as JSONL files but in a columnar format optimized for efficient data loading. The structure is identical, with the following columns:

- `id`: Question identifier
- `question`: Question text
- `answer`: Answer entities (JSON string)
- `q_entity`: Question entities (JSON string)
- `a_entity`: Answer entities (JSON string)
- `graph`: Knowledge graph triples (JSON string)
- `paths`: Reasoning paths (JSON string)

### 3. Vocabulary File (`vocab.json`)

The vocabulary file contains mappings for entities and relations to integer indices.

#### Structure:
```json
{
  "entity2idx": {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
    "<MASK>": 4,
    "entity_name": index,
    ...
  },
  "relation2idx": {
    "<PAD>": 0,
    "<UNK>": 1,
    "<MASK>": 2,
    "relation.name": index,
    ...
  }
}
```

#### Statistics:
- **Entity vocabulary size**: 1,316,471 entities
- **Relation vocabulary size**: 6,097 relations
- **Special tokens**: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<MASK>` for entities; `<PAD>`, `<UNK>`, `<MASK>` for relations

### 4. GSR Data (`gsr_data/`)

The GSR (Graph Subgraph Relation) data directory contains processed data for relation-based training.

#### Files:
- **gsr_training_data.jsonl**: Training data with relation patterns
- **gsr_val_data.jsonl**: Validation data with relation patterns
- **subgraph_index.json**: Index of relation patterns and their statistics

#### GSR Data Format:
Each line in GSR JSONL files contains:
```json
{
  "question": "what is the name of justin bieber brother",
  "subgraph_id": "path_people_person_parents_people_person_children",
  "relations": [
    "people.person.parents",
    "people.person.children"
  ]
}
```

#### Subgraph Index Structure:
The `subgraph_index.json` contains:
- **patterns**: Dictionary of relation patterns
  - Each pattern has:
    - `subgraph_id`: Unique identifier for the pattern
    - `relations`: List of relations in the pattern
    - `relation_pattern`: Regex-like pattern string
    - `example_count`: Number of examples using this pattern
    - `example_triples`: Sample triples demonstrating the pattern
    - `answer_types`: List of answer types (if available)

- **Total patterns**: 5,059 unique relation patterns

## Sample Data Examples

### Example 1: Simple Question
```json
{
  "id": "WebQTrn-0",
  "question": "what is the name of justin bieber brother",
  "answer": ["Jaxon Bieber"],
  "q_entity": ["Justin Bieber"],
  "a_entity": ["Jaxon Bieber"],
  "graph": [
    ["Justin Bieber", "people.person.parents", "Jeremy Bieber"],
    ["Jeremy Bieber", "people.person.children", "Jaxon Bieber"],
    ...
  ],
  "paths": [
    {
      "full_path": "(Justin Bieber) --[people.person.parents]--> (Jeremy Bieber) --[people.person.children]--> (Jaxon Bieber)",
      "relation_chain": "people.person.parents -> people.person.children",
      "entities": ["Justin Bieber", "Jeremy Bieber", "Jaxon Bieber"],
      "relations": ["people.person.parents", "people.person.children"]
    }
  ]
}
```

### Example 2: Location-based Question
```json
{
  "question": "what time zone am i in cleveland ohio",
  "subgraph_id": "path_location_location_time_zones",
  "relations": ["location.location.time_zones"]
}
```

## Relation Types

The dataset uses Freebase relation names with hierarchical naming:
- **People relations**: `people.person.*` (e.g., `people.person.parents`, `people.person.children`, `people.person.nationality`)
- **Location relations**: `location.location.*` (e.g., `location.location.containedby`, `location.location.time_zones`)
- **Film relations**: `film.film.*` (e.g., `film.film.directed_by`, `film.film.starring`)
- **Music relations**: `music.artist.*`, `music.recording.*`
- **Award relations**: `award.award_winner.*`, `award.award_nomination.*`
- And many more domain-specific relations

## Usage Notes

1. **Data Loading**: 
   - Use Parquet files for efficient batch loading in training pipelines
   - Use JSONL files for streaming or when working with specific tools that require line-by-line processing

2. **Graph Size**: 
   - Subgraphs can be very large (thousands of triples per question)
   - Consider filtering or sampling when memory is constrained

3. **Paths**: 
   - Not all questions have paths (some may have empty `paths` arrays)
   - Paths represent reasoning chains from question entities to answer entities

4. **Multiple Answers**: 
   - Some questions have multiple valid answers
   - The `answer` and `a_entity` fields are lists to accommodate this

5. **Entity Format**: 
   - Entities use Freebase entity names (e.g., "Justin Bieber", "m.0xxx" for mid identifiers)
   - Some entities may have special identifiers starting with "m." or "g."

## Complex Examples

The dataset contains many questions with complex reasoning requirements. Here are examples of different types of complexity:

### Multi-Path Questions

Many questions have multiple reasoning paths (94.5% of training samples have 3+ paths). This indicates that there are often multiple ways to answer the same question using different relation chains.

#### Example 1: Family Relationship Question
**Question**: "what is the name of justin bieber brother"

**Answer**: `["Jaxon Bieber"]`

**Number of paths**: 3

**Sample paths**:
1. **Direct path**: `(Justin Bieber) --[people.person.parents]--> (Jeremy Bieber) --[people.person.children]--> (Jaxon Bieber)`
   - Relations: `people.person.parents -> people.person.children`

2. **Profession-based path**: `(Justin Bieber) --[people.person.profession]--> (Musician) <--[people.person.profession]-- (Jeremy Bieber) --[people.person.children]--> (Jaxon Bieber)`
   - Relations: `people.person.profession -> people.person.profession -> people.person.children`

3. **Award-based path**: `(Justin Bieber) --[award.award_winner.awards_won]--> (m.0yrkc0l) --[award.award_honor.award_winner]--> (Justin Bieber) --[people.person.parents]--> (Jeremy Bieber) --[people.person.children]--> (Jaxon Bieber)`
   - Relations: `award.award_winner.awards_won -> award.award_honor.award_winner -> people.person.parents -> people.person.children`

#### Example 2: Character Identification Question
**Question**: "what character did natalie portman play in star wars"

**Answer**: `["Padmé Amidala"]`

**Number of paths**: 3

**Sample paths**:
1. **Gender-based path**: `(Natalie Portman) --[people.person.gender]--> (Female) <--[fictional_universe.fictional_character.gender]-- (Padmé Amidala)`
2. **Image-based path**: `(Natalie Portman) --[common.topic.image]--> (Natalie Portman) --[people.person.gender]--> (Female) <--[fictional_universe.fictional_character.gender]-- (Padmé Amidala)`
3. **Dating-based path**: `(Natalie Portman) --[base.popstra.celebrity.dated]--> (m.065q1bw) --[base.popstra.dated.participant]--> (Natalie Portman) --[people.person.gender]--> (Female) <--[fictional_universe.fictional_character.gender]-- (Padmé Amidala)`

#### Statistics:
- **2,671 questions** (94.5%) have 3+ paths
- **681 questions** (24.1%) have 10+ paths
- **Maximum paths observed**: 100 paths for a single question

### Complex Answer Questions

Some questions have multiple valid answers, requiring the model to identify all correct entities.

#### Example 1: Multiple Countries
**Question**: "which countries border the us"

**Answers**: `["Mexico", "Canada"]`

**Number of paths**: 6

#### Example 2: Multiple Locations
**Question**: "where is rome italy located on a map"

**Answers**: `["Italy", "Lazio", "Province of Rome"]`

**Number of paths**: 9

#### Example 3: Extensive Lists
**Question**: "what books did beverly cleary right"

**Answers**: 376 different book titles (including translations and editions)

**Number of paths**: 100

**Sample answers**: 
- "Ramona and Her Mother"
- "Dear Mr. Henshaw"
- "Henry Huggins"
- "The Mouse and the Motorcycle"
- "Beezus and Ramona"
- ... (371 more)

#### Example 4: Tourist Attractions
**Question**: "what to see near sedona arizona"

**Answers**: 11 different locations:
- "Bell Rock"
- "Sycamore Canyon"
- "Slide Rock State Park"
- "Sedona Airport"
- "Seven Canyons"
- "Cathedral Rock"
- "Sedona International Film Festival"
- "Oak Creek Canyon"
- "Chapel of the Holy Cross"
- "Red Rock State Park"
- "Honanki"

**Number of paths**: 33

#### Statistics:
- **1,363 questions** (48.2%) have multiple answers or complex answer structures
- **349 questions** (12.3%) have 10+ answers
- **Maximum answers observed**: 3,688 answers for "what songs did mozart write"

### Complex Relation Chains

Some questions require reasoning through multiple relation hops to reach the answer.

#### Example 1: Government and Currency
**Question**: "what kind of money to take to bahamas"

**Answer**: `["Bahamian dollar"]`

**Complex path** (4 relations):
```
(Bahamas) --[government.governmental_jurisdiction.governing_officials]--> 
(m.0_mv93_) --[government.government_position_held.governmental_body]--> 
(Senate of the Bahamas) --[government.governmental_body.jurisdiction]--> 
(Bahamas) --[location.country.currency_used]--> 
(Bahamian dollar)
```

**Relation chain**: `government.governmental_jurisdiction.governing_officials -> government.government_position_held.governmental_body -> government.governmental_body.jurisdiction -> location.country.currency_used`

#### Example 2: Location via Airport
**Question**: "what country is the grand bahama island in"

**Complex path** (3 relations):
```
(Grand Bahama) --[location.location.nearby_airports]--> 
(West End Airport) --[location.location.containedby]--> 
(Grand Bahama) --[location.location.containedby]--> 
(Bahamas)
```

**Relation chain**: `location.location.nearby_airports -> location.location.containedby -> location.location.containedby`

#### Example 3: Film Character via Multiple Hops
**Question**: "what character did john noble play in lord of the rings"

**Complex path** (4 relations):
```
(John Noble) --[film.actor.film]--> 
(m.0h1ldcx) --[film.performance.actor]--> 
(John Noble) --[people.person.gender]--> 
(Male) <--[fictional_universe.fictional_character.gender]-- 
(Denethor II)
```

**Relation chain**: `film.actor.film -> film.performance.actor -> people.person.gender -> fictional_universe.fictional_character.gender`

#### Statistics:
- **2,657 questions** (94.0%) have at least one path with 3+ relations
- Most complex paths contain 3-4 relations
- Paths can include bidirectional reasoning (indicated by `<--` in path notation)

### Combined Complexity

Many questions exhibit multiple types of complexity simultaneously:

- **Multi-path + Complex answers**: Questions with many paths leading to multiple answers
- **Complex relations + Multi-path**: Questions requiring long reasoning chains with multiple alternative paths
- **All three**: Questions like "what books did beverly cleary right" with 100 paths, 376 answers, and various relation chains

## Related Files

- **GSR Training Data**: Processed data focusing on relation patterns for relation-only training
- **Subgraph Index**: Pre-computed relation patterns and their statistics for efficient pattern matching

