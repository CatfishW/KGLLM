# Combined KGQA System Evaluation Report

**Generated at**: 2025-12-19 23:42:37

## Summary Metrics

### Path Retrieval
| K | Recall | Hits |
|---|--------|------|
| 1 | 0.2754 | 0.3102 |
| 3 | 0.4146 | 0.4721 |
| 5 | 0.4692 | 0.5350 |
| 10 | 0.5203 | 0.5832 |
| 20 | 0.5203 | 0.5832 |
| 50 | 0.5203 | 0.5832 |
| 100 | 0.5203 | 0.5832 |

**MRR**: 0.4034

### Source Contribution
- Retrieved paths in top-k: 100.0%
- Generated paths in top-k: 0.0%

### Answer Quality
- Exact Match: 0.0000
- F1 Score: 0.0000
- Entity Overlap: 0.0000

### Timing
- Retrieval: 5.82 ms
- Generation: 0.00 ms
- Reranking: 54.64 ms
- LLM: 0.00 ms
- **Total**: 60.45 ms

## Example Results

### Example 1
**Question**: what does jamaican people speak

**Topic Entity**: Jamaica

**Top Retrieved Paths**:
```
  1. people.ethnicity.languages_spoken (score: 0.5547)
  2. language.human_language.countries_spoken_in (score: 0.5430)
  3. location.country.languages_spoken (score: 0.4922)
  4. people.person.languages -> language.human_language.countries_spoken_in (score: 0.3943)
  5. people.person.languages -> people.person.languages (score: 0.0200)
```

**Generated Answer**: 

**Ground Truth Answers**: Jamaican English, Jamaican Creole English Language

### Example 2
**Question**: what did james k polk do before he was president

**Topic Entity**: James K. Polk

**Top Retrieved Paths**:
```
  1. government.us_vice_president.to_president (score: 0.0295)
  2. government.us_president.vice_president (score: 0.0170)
  3. government.political_district.representatives -> government.government_position_held.office_holder (score: 0.0085)
  4. base.politicalconventions.presidential_nominee.nominated_at (score: 0.0055)
  5. location.us_state.capital -> people.person.place_of_birth (score: 0.0022)
```

**Generated Answer**: 

**Ground Truth Answers**: United States Representative, Governor of Tennessee, Speaker of the United States House of Representatives

### Example 3
**Question**: where is jamarcus russell from

**Topic Entity**: JaMarcus Russell

**Top Retrieved Paths**:
```
  1. location.us_state.capital -> people.person.place_of_birth (score: 0.0138)
  2. location.location.people_born_here -> people.person.religion (score: 0.0070)
  3. people.person.place_of_birth -> sports.sports_team_location.teams (score: 0.0064)
  4. sports.sports_team.location (score: 0.0042)
  5. people.person.nationality -> sports.sports_team_location.teams (score: 0.0035)
```

**Generated Answer**: 

**Ground Truth Answers**: Mobile

### Example 4
**Question**: where was george washington carver from

**Topic Entity**: George Washington Carver

**Top Retrieved Paths**:
```
  1. government.governmental_body.jurisdiction -> people.person.place_of_birth (score: 0.0359)
  2. location.us_state.capital -> people.person.place_of_birth (score: 0.0202)
  3. location.administrative_division.country -> people.person.nationality (score: 0.0096)
  4. base.aareas.schema.administrative_area.capital -> location.location.people_born_here (score: 0.0095)
  5. royalty.monarch.royal_line -> royalty.royal_line.monarchs_from_this_line (score: 0.0023)
```

**Generated Answer**: 

**Ground Truth Answers**: Diamond

### Example 5
**Question**: what else did ben franklin invent

**Topic Entity**: Benjamin Franklin

**Top Retrieved Paths**:
```
  1. organization.organization.founders (score: 0.0244)
  2. astronomy.astronomical_discovery.discoverer (score: 0.0159)
  3. law.invention.inventor (score: 0.0127)
  4. law.inventor.us_patents (score: 0.0110)
  5. law.inventor.inventions (score: 0.0101)
```

**Generated Answer**: 

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

**Generated Answer**: 

**Ground Truth Answers**: Pat Nixon

### Example 7
**Question**: who is governor of ohio 2011

**Topic Entity**: Ohio

**Top Retrieved Paths**:
```
  1. government.politician.government_positions_held -> government.government_position_held.basic_title (score: 0.0087)
  2. government.political_district.representatives -> government.government_position_held.office_holder (score: 0.0057)
  3. government.us_president.vice_president (score: 0.0022)
  4. government.governmental_body.jurisdiction -> people.person.place_of_birth (score: 0.0010)
  5. location.country.form_of_government -> government.form_of_government.countries (score: 0.0007)
```

**Generated Answer**: 

**Ground Truth Answers**: John Kasich, Ted Strickland, Return J. Meigs, Jr.

### Example 8
**Question**: who was vice president after kennedy died

**Topic Entity**: John F. Kennedy

**Top Retrieved Paths**:
```
  1. government.us_president.vice_president (score: 0.0610)
  2. government.us_vice_president.to_president (score: 0.0293)
  3. base.obamabase.cabinet_member.cabinet_position (score: 0.0033)
  4. symbols.namesake.named_after -> people.deceased_person.place_of_death (score: 0.0008)
  5. base.locations.states_and_provences.cities_within -> people.deceased_person.place_of_death (score: 0.0007)
```

**Generated Answer**: 

**Ground Truth Answers**: Lyndon B. Johnson

### Example 9
**Question**: where is the fukushima daiichi nuclear plant located

**Topic Entity**: Fukushima Daiichi Nuclear Power Plant

**Top Retrieved Paths**:
```
  1. location.location.contains (score: 0.0090)
  2. event.disaster.areas_affected (score: 0.0037)
  3. time.event.locations -> location.location.contains (score: 0.0027)
  4. fictional_universe.fictional_universe.organizations (score: 0.0025)
  5. location.location.contains_major_portion_of (score: 0.0014)
```

**Generated Answer**: 

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

**Generated Answer**: 

**Ground Truth Answers**: Wales, Northern Ireland, Scotland, England
