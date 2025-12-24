# Combined KGQA System Evaluation Report

**Generated at**: 2025-12-20 20:44:52

## Summary Metrics

### Path Retrieval
| K | Recall | Hits |
|---|--------|------|
| 1 | 0.2483 | 0.2733 |
| 3 | 0.4075 | 0.4667 |
| 5 | 0.4615 | 0.5300 |
| 10 | 0.5047 | 0.5633 |
| 20 | 0.5047 | 0.5633 |
| 50 | 0.5047 | 0.5633 |
| 100 | 0.5047 | 0.5633 |

**MRR**: 0.3814

### Source Contribution
- Retrieved paths in top-k: 100.0%
- Generated paths in top-k: 0.0%

### Answer Quality
- Exact Match: 0.3767
- F1 Score: 0.5635
- Entity Overlap: 0.4915

### Timing
- Retrieval: 9.76 ms
- Generation: 0.00 ms
- Reranking: 49.11 ms
- LLM: 477.58 ms
- **Total**: 536.45 ms

## Example Results

### Example 1
**Question**: what does jamaican people speak

**Topic Entity**: Jamaica

**Top Retrieved Paths**:
```
  1. people.ethnicity.languages_spoken (score: 0.5547)
  2. language.human_language.countries_spoken_in (score: 0.5430)
  3. location.country.languages_spoken (score: 0.4922)
  4. location.country.languages_spoken -> language.human_language.countries_spoken_in (score: 0.4551)
  5. people.person.languages -> language.human_language.countries_spoken_in (score: 0.3943)
```

**Generated Answer**: Jamaican English, Jamaican Creole

**Ground Truth Answers**: Jamaican English, Jamaican Creole English Language

### Example 2
**Question**: what did james k polk do before he was president

**Topic Entity**: James K. Polk

**Top Retrieved Paths**:
```
  1. government.politician.government_positions_held -> government.government_position_held.office_position_or_title (score: 0.0328)
  2. government.us_vice_president.to_president (score: 0.0289)
  3. government.politician.government_positions_held -> government.government_position_held.basic_title (score: 0.0282)
  4. government.politician.government_positions_held -> government.government_position_held.jurisdiction_of_office (score: 0.0250)
  5. government.us_president.vice_president (score: 0.0172)
```

**Generated Answer**: United States Representative, Governor of Tennessee, Speaker of the U.S. House

**Ground Truth Answers**: United States Representative, Governor of Tennessee, Speaker of the United States House of Representatives

### Example 3
**Question**: where is jamarcus russell from

**Topic Entity**: JaMarcus Russell

**Top Retrieved Paths**:
```
  1. people.person.place_of_birth -> music.artist.origin (score: 0.1896)
  2. people.person.place_of_birth (score: 0.0690)
  3. people.person.place_of_birth -> location.location.contains (score: 0.0382)
  4. people.person.place_of_birth -> location.location.containedby (score: 0.0177)
  5. location.us_state.capital -> people.person.place_of_birth (score: 0.0141)
```

**Generated Answer**: Mobile, Alabama

**Ground Truth Answers**: Mobile

### Example 4
**Question**: where was george washington carver from

**Topic Entity**: George Washington Carver

**Top Retrieved Paths**:
```
  1. people.person.place_of_birth -> location.location.contains (score: 0.1217)
  2. people.person.place_of_birth (score: 0.1082)
  3. people.person.place_of_birth -> location.location.containedby (score: 0.0421)
  4. people.person.place_of_birth -> people.person.place_of_birth (score: 0.0362)
  5. government.governmental_body.jurisdiction -> people.person.place_of_birth (score: 0.0357)
```

**Generated Answer**: United States of America

**Ground Truth Answers**: Diamond

### Example 5
**Question**: what else did ben franklin invent

**Topic Entity**: Benjamin Franklin

**Top Retrieved Paths**:
```
  1. law.invention.inventor (score: 0.0127)
  2. law.inventor.us_patents (score: 0.0110)
  3. law.inventor.inventions (score: 0.0101)
  4. fictional_universe.fictional_character.occupation (score: 0.0084)
  5. visual_art.artwork.art_form (score: 0.0059)
```

**Generated Answer**: Lightning rod, Bifocals, Franklin stove, Glass harmonica

**Ground Truth Answers**: Lightning rod, Glass harmonica, Bifocals, Franklin stove

### Example 6
**Question**: who was richard nixon married to

**Topic Entity**: Richard Nixon

**Top Retrieved Paths**:
```
  1. people.person.spouse_s -> people.marriage.spouse (score: 0.0379)
  2. fictional_universe.fictional_character.romantically_involved_with -> fictional_universe.romantic_involvement.partner (score: 0.0052)
  3. people.person.spouse_s -> people.marriage.location_of_ceremony (score: 0.0027)
  4. government.us_president.vice_president (score: 0.0011)
  5. government.us_vice_president.to_president (score: 0.0006)
```

**Generated Answer**: Pat Nixon

**Ground Truth Answers**: Pat Nixon

### Example 7
**Question**: who is governor of ohio 2011

**Topic Entity**: Ohio

**Top Retrieved Paths**:
```
  1. government.politician.government_positions_held -> government.government_position_held.basic_title (score: 0.0087)
  2. government.politician.government_positions_held -> government.government_position_held.office_position_or_title (score: 0.0078)
  3. government.political_district.representatives -> government.government_position_held.office_holder (score: 0.0057)
  4. government.us_president.vice_president (score: 0.0022)
  5. government.governmental_body.members -> government.government_position_held.office_holder (score: 0.0021)
```

**Generated Answer**: Mike DeWine

**Ground Truth Answers**: John Kasich, Ted Strickland, Return J. Meigs, Jr.

### Example 8
**Question**: who was vice president after kennedy died

**Topic Entity**: John F. Kennedy

**Top Retrieved Paths**:
```
  1. government.us_president.vice_president (score: 0.0610)
  2. government.us_vice_president.to_president (score: 0.0293)
  3. base.obamabase.cabinet_member.cabinet_position (score: 0.0033)
  4. government.politician.government_positions_held -> government.government_position_held.basic_title (score: 0.0033)
  5. government.politician.government_positions_held -> government.government_position_held.office_position_or_title (score: 0.0031)
```

**Generated Answer**: Lyndon B. Johnson

**Ground Truth Answers**: Lyndon B. Johnson

### Example 9
**Question**: where is the fukushima daiichi nuclear plant located

**Topic Entity**: Fukushima Daiichi Nuclear Power Plant

**Top Retrieved Paths**:
```
  1. government.governmental_body.jurisdiction -> people.person.place_of_birth (score: 0.0023)
  2. base.locations.states_and_provences.cities_within -> location.location.people_born_here (score: 0.0013)
  3. people.person.place_of_birth -> location.location.contains (score: 0.0011)
  4. base.aareas.schema.administrative_area.subdividing_place (score: 0.0011)
  5. base.aareas.schema.administrative_area.subdivides_place (score: 0.0009)
```

**Generated Answer**: Fukushima Prefecture

**Ground Truth Answers**: Japan, Okuma

### Example 10
**Question**: what countries are part of the uk

**Topic Entity**: United Kingdom

**Top Retrieved Paths**:
```
  1. location.country.administrative_divisions (score: 0.0141)
  2. location.location.partially_contains -> location.country.administrative_divisions (score: 0.0116)
  3. location.country.form_of_government -> government.form_of_government.countries (score: 0.0082)
  4. location.administrative_division.country (score: 0.0067)
  5. base.locations.continents.countries_within (score: 0.0062)
```

**Generated Answer**: Wales, Scotland, England, Northern Ireland

**Ground Truth Answers**: Wales, Northern Ireland, Scotland, England
