# KGQA RAG System Evaluation Report

**Generated at**: 2025-12-19 16:11:39

## Summary Metrics

### Recall@K
| K | Recall |
|---|--------|
| 1 | 0.0748 |
| 3 | 0.1300 |
| 5 | 0.1618 |
| 10 | 0.2412 |
| 20 | 0.3870 |
| 50 | 0.6276 |
| 100 | 0.7275 |

### Hits@K
| K | Hits |
|---|------|
| 1 | 0.0861 |
| 3 | 0.1529 |
| 5 | 0.1946 |
| 10 | 0.2903 |
| 20 | 0.4425 |
| 50 | 0.6891 |
| 100 | 0.7836 |

### MRR: 0.1565

## Timing Analysis

- **Average Retrieval Time**: 14.29 ms
- **Average Reranking Time**: 0.00 ms
- **Average Total Time**: 14.29 ms
- **Throughput**: 69.97 queries/sec

## Example Results

### Example 1
**Question**: what does jamaican people speak

**Ground Truth Paths**: 
```
  - location.country.official_language
  - location.country.languages_spoken
```

**Top Retrieved Paths**:
```
  1. people.ethnicity.languages_spoken (score: 0.0208)
  2. people.person.ethnicity (score: 0.0205)
  3. people.person.languages -> language.human_language.countries_spoken_in (score: 0.0201)
  4. tv.tv_program.country_of_origin -> people.person.nationality (score: 0.0200)
  5. people.person.languages -> people.person.languages (score: 0.0199)
```

### Example 2
**Question**: what did james k polk do before he was president

**Ground Truth Paths**: 
```
  - government.politician.government_positions_held -> government.government_position_held.office_position_or_title
  - government.politician.government_positions_held -> government.government_position_held.office_position_or_title
  - government.politician.government_positions_held -> government.government_position_held.office_position_or_title
```

**Top Retrieved Paths**:
```
  1. government.us_president.vice_president (score: 0.0328)
  2. government.us_vice_president.to_president (score: 0.0323)
  3. law.invention.inventor (score: 0.0220)
  4. government.political_district.representatives -> government.government_position_held.office_holder (score: 0.0220)
  5. film.actor.film -> film.film.starring (score: 0.0216)
```

### Example 3
**Question**: where is jamarcus russell from

**Ground Truth Paths**: 
```
  - people.person.place_of_birth
```

**Top Retrieved Paths**:
```
  1. location.statistical_region.places_imported_from -> location.imports_and_exports.imported_from (score: 0.0219)
  2. fictional_universe.fictional_character.places_lived (score: 0.0179)
  3. film.film.country -> people.person.nationality (score: 0.0177)
  4. film.film_location.featured_in_films -> film.film.country (score: 0.0167)
  5. fictional_universe.fictional_character.parents (score: 0.0167)
```

### Example 4
**Question**: where was george washington carver from

**Ground Truth Paths**: 
```
  - people.person.place_of_birth
```

**Top Retrieved Paths**:
```
  1. location.statistical_region.places_imported_from -> location.imports_and_exports.imported_from (score: 0.0269)
  2. fictional_universe.fictional_character.places_lived (score: 0.0214)
  3. government.governmental_body.jurisdiction -> people.person.place_of_birth (score: 0.0200)
  4. base.aareas.schema.administrative_area.capital -> location.location.people_born_here (score: 0.0194)
  5. fictional_universe.fictional_character.occupation -> people.person.profession (score: 0.0190)
```

### Example 5
**Question**: what else did ben franklin invent

**Ground Truth Paths**: 
```
  - law.inventor.inventions
  - law.inventor.inventions
  - law.inventor.inventions
  - base.argumentmaps.innovator.original_ideas
```

**Top Retrieved Paths**:
```
  1. symbols.namesake.named_after -> people.person.religion (score: 0.0235)
  2. visual_art.visual_artist.art_forms (score: 0.0217)
  3. time.event.included_in_event -> base.supertopics.supertopic.related_topics (score: 0.0214)
  4. tv.tv_program_creator.programs_created (score: 0.0213)
  5. tv.tv_program.program_creator -> fictional_universe.fictional_character_creator.fictional_characters_created (score: 0.0203)
```

### Example 6
**Question**: who was richard nixon married to

**Ground Truth Paths**: 
```
  - people.person.nationality -> people.person.nationality
```

**Top Retrieved Paths**:
```
  1. book.author.series_written_or_contributed_to (score: 0.0237)
  2. fictional_universe.fictional_character.romantically_involved_with -> fictional_universe.romantic_involvement.partner (score: 0.0236)
  3. government.us_president.vice_president (score: 0.0218)
  4. government.us_vice_president.to_president (score: 0.0212)
  5. fictional_universe.fictional_character.places_lived (score: 0.0205)
```

### Example 7
**Question**: who is governor of ohio 2011

**Ground Truth Paths**: 
```
  - location.administrative_division.first_level_division_of -> people.person.nationality
```

**Top Retrieved Paths**:
```
  1. government.governmental_body.body_this_is_a_component_of (score: 0.0280)
  2. location.us_state.capital -> people.person.place_of_birth (score: 0.0245)
  3. government.governmental_body.jurisdiction -> people.person.place_of_birth (score: 0.0233)
  4. people.person.place_of_birth (score: 0.0227)
  5. location.country.form_of_government (score: 0.0222)
```

### Example 8
**Question**: who was vice president after kennedy died

**Ground Truth Paths**: 
```
  - government.us_president.vice_president
```

**Top Retrieved Paths**:
```
  1. government.us_vice_president.to_president (score: 0.0325)
  2. government.us_president.vice_president (score: 0.0323)
  3. symbols.namesake.named_after -> people.deceased_person.place_of_death (score: 0.0290)
  4. symbols.namesake.named_after -> people.person.religion (score: 0.0257)
  5. symbols.namesake.named_after (score: 0.0243)
```

### Example 9
**Question**: where is the fukushima daiichi nuclear plant located

**Ground Truth Paths**: 
```
  - location.location.containedby
  - location.location.containedby
```

**Top Retrieved Paths**:
```
  1. film.film.featured_film_locations (score: 0.0236)
  2. event.disaster.areas_affected (score: 0.0233)
  3. film.film_location.featured_in_films -> film.film.country (score: 0.0225)
  4. film.film.country -> organization.organization.geographic_scope (score: 0.0214)
  5. fictional_universe.fictional_universe.locations -> fictional_universe.fictional_organization.sub_organization_in_fiction (score: 0.0213)
```

### Example 10
**Question**: what countries are part of the uk

**Ground Truth Paths**: 
```
  - location.country.first_level_divisions
  - location.country.first_level_divisions
  - base.aareas.schema.administrative_area.administrative_children
  - location.country.first_level_divisions
```

**Top Retrieved Paths**:
```
  1. olympics.olympic_participating_country.olympics_participated_in -> olympics.olympic_games.participating_countries (score: 0.0315)
  2. location.location.partially_contains -> geography.river.basin_countries (score: 0.0312)
  3. geography.lake.basin_countries (score: 0.0310)
  4. location.country.form_of_government -> government.form_of_government.countries (score: 0.0305)
  5. geography.river.basin_countries (score: 0.0304)
```
