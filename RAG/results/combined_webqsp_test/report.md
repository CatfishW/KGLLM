# Combined KGQA System Evaluation Report

**Generated at**: 2025-12-20 19:39:43

## Summary Metrics

### Path Retrieval
| K | Recall | Hits |
|---|--------|------|
| 1 | 0.1775 | 0.1900 |
| 3 | 0.3320 | 0.3900 |
| 5 | 0.4037 | 0.4800 |
| 10 | 0.4607 | 0.5200 |
| 20 | 0.4607 | 0.5200 |
| 50 | 0.4607 | 0.5200 |
| 100 | 0.4607 | 0.5200 |

**MRR**: 0.2990

### Source Contribution
- Retrieved paths in top-k: 69.1%
- Generated paths in top-k: 30.9%

### Answer Quality
- Exact Match: 0.4400
- F1 Score: 0.6153
- Entity Overlap: 0.5289

### Timing
- Retrieval: 15.23 ms
- Generation: 542.09 ms
- Reranking: 76.44 ms
- LLM: 664.50 ms
- **Total**: 1298.26 ms

## Example Results

### Example 1
**Question**: what does jamaican people speak

**Topic Entity**: Jamaica

**Top Retrieved Paths**:
```
  1. people.ethnicity.languages_spoken (score: 0.5547)
  2. language.human_language.countries_spoken_in (score: 0.5488)
  3. location.country.languages_spoken (score: 0.4922)
  4. people.person.languages -> language.human_language.countries_spoken_in (score: 0.3923)
  5. location.country.official_language -> location.country.languages_spoken -> location.country.languages_spoken -> location.country.languages_spoken -> location.country.official_language -> location.country.languages_spoken (score: 0.0223)
```

**Generated Answer**: Jamaican English, Jamaican Creole English Language

**Ground Truth Answers**: Jamaican English, Jamaican Creole English Language

### Example 2
**Question**: what did james k polk do before he was president

**Topic Entity**: James K. Polk

**Top Retrieved Paths**:
```
  1. government.us_vice_president.to_president (score: 0.0295)
  2. government.us_president.vice_president (score: 0.0170)
  3. people.person.profession -> common.image.appears_in_topic_gallery -> common.topic.subjects -> government.governmental_body.offices_positions -> sports.sports_team.championships -> government.government_position_held.appointed_by (score: 0.0119)
  4. government.political_district.representatives -> government.government_position_held.office_holder (score: 0.0084)
  5. baseball.baseball_team.team_stats -> base.lightweight.profession.professions_similar -> fictional_universe.fictional_character.rank -> location.country.languages_spoken -> book.author.works_written -> people.person.profession (score: 0.0058)
```

**Generated Answer**: Actually, the candidate list only includes George M

**Ground Truth Answers**: United States Representative, Governor of Tennessee, Speaker of the United States House of Representatives

### Example 3
**Question**: where is jamarcus russell from

**Topic Entity**: JaMarcus Russell

**Top Retrieved Paths**:
```
  1. location.us_state.capital -> people.person.place_of_birth (score: 0.0140)
  2. people.person.place_of_birth -> people.deceased_person.place_of_death -> film.film.subjects -> location.country.languages_spoken -> people.person.nationality -> sports.sports_league_draft_pick.player (score: 0.0109)
  3. people.person.place_of_birth -> celebrities.celebrity.substance_abuse_problems -> media_common.quotation.author -> people.profession.part_of_professional_field -> base.schemastaging.athlete_extra.salary -> sports.sports_league_draft_pick.player (score: 0.0099)
  4. location.location.people_born_here -> people.person.religion (score: 0.0070)
  5. people.person.place_of_birth -> sports.sports_team_location.teams (score: 0.0064)
```

**Generated Answer**: Mobile

**Ground Truth Answers**: Mobile

### Example 4
**Question**: where was george washington carver from

**Topic Entity**: George Washington Carver

**Top Retrieved Paths**:
```
  1. government.governmental_body.jurisdiction -> people.person.place_of_birth (score: 0.0357)
  2. people.person.place_of_birth -> people.person.place_of_birth -> people.cause_of_death.people -> people.person.education -> base.aareas.schema.administrative_area.administrative_parent -> people.person.parents (score: 0.0213)
  3. location.us_state.capital -> people.person.place_of_birth (score: 0.0205)
  4. people.person.place_of_birth -> education.education.institution -> protected_sites.natural_or_cultural_preservation_agency.natural_or_cultural_listings_maintained -> people.person.place_of_birth -> people.person.education -> people.person.place_of_birth (score: 0.0121)
  5. location.administrative_division.country -> people.person.nationality (score: 0.0096)
```

**Generated Answer**: Missouri

**Ground Truth Answers**: Diamond

### Example 5
**Question**: what else did ben franklin invent

**Topic Entity**: Benjamin Franklin

**Top Retrieved Paths**:
```
  1. law.inventor.inventions -> law.inventor.inventions -> language.language_family.geographic_distribution -> influence.influence_node.influenced -> law.inventor.inventions -> base.argumentmaps.innovator.original_ideas (score: 0.0453)
  2. law.inventor.inventions -> law.inventor.inventions -> law.inventor.inventions -> law.inventor.inventions -> organization.organization.founders -> law.inventor.inventions (score: 0.0309)
  3. base.argumentmaps.thing_of_disputed_value.disparagement -> basketball.basketball_player_stats.team -> location.country.official_language -> location.location.contains -> law.inventor.inventions -> law.inventor.inventions (score: 0.0270)
  4. organization.organization.founders (score: 0.0248)
  5. sports.professional_sports_team.owner_s -> location.mailing_address.citytown -> language.human_language.region -> people.ethnicity.people -> law.inventor.inventions -> sports.sports_award.team (score: 0.0172)
```

**Generated Answer**: Lightning rod, Bifocals, Franklin stove, Glass harmonica

**Ground Truth Answers**: Lightning rod, Glass harmonica, Bifocals, Franklin stove

### Example 6
**Question**: who was richard nixon married to

**Topic Entity**: Richard Nixon

**Top Retrieved Paths**:
```
  1. people.person.spouse_s -> people.marriage.spouse (score: 0.0388)
  2. fictional_universe.fictional_character.romantically_involved_with -> fictional_universe.romantic_involvement.partner (score: 0.0051)
  3. people.person.spouse_s -> people.marriage.location_of_ceremony (score: 0.0026)
  4. people.person.children -> freebase.valuenotation.has_value -> people.person.nationality -> base.popstra.dated.participant -> people.person.spouse_s -> base.popstra.dated.participant (score: 0.0019)
  5. government.us_president.vice_president (score: 0.0011)
```

**Generated Answer**: Pat Nixon

**Ground Truth Answers**: Pat Nixon

### Example 7
**Question**: who is governor of ohio 2011

**Topic Entity**: Ohio

**Top Retrieved Paths**:
```
  1. government.politician.government_positions_held -> government.government_position_held.basic_title (score: 0.0086)
  2. government.political_district.representatives -> government.government_position_held.office_holder (score: 0.0057)
  3. government.us_president.vice_president (score: 0.0022)
  4. biology.organism_classification.higher_classification -> government.government_position_held.office_holder -> people.person.places_lived -> sports.sports_league_draft_pick.team -> meteorology.tropical_cyclone.strongest_storm_of -> sports.multi_event_tournament.competitors (score: 0.0011)
  5. government.governmental_body.jurisdiction -> people.person.place_of_birth (score: 0.0010)
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
  3. government.us_president.vice_president -> government.us_president.vice_president -> military.military_command.military_commander -> government.us_president.vice_president -> government.government_position_held.office_holder -> government.us_president.vice_president (score: 0.0276)
  4. government.us_president.vice_president -> media_common.quotation.author -> book.written_work.author -> government.us_president.vice_president -> book.author.works_written -> government.us_president.vice_president (score: 0.0130)
  5. government.us_president.vice_president -> award.award_category.nominees -> film.performance.actor -> government.us_president.vice_president -> base.popstra.rehab_stay.patient -> people.person.children (score: 0.0038)
```

**Generated Answer**: Lyndon B. Johnson

**Ground Truth Answers**: Lyndon B. Johnson

### Example 9
**Question**: where is the fukushima daiichi nuclear plant located

**Topic Entity**: Fukushima Daiichi Nuclear Power Plant

**Top Retrieved Paths**:
```
  1. location.location.contains (score: 0.0089)
  2. location.location.containedby -> location.location.containedby -> location.location.containedby -> location.location.contains -> organization.organization.founders -> location.location.containedby (score: 0.0070)
  3. event.disaster.areas_affected (score: 0.0036)
  4. time.event.locations -> location.location.contains (score: 0.0028)
  5. fictional_universe.fictional_universe.organizations (score: 0.0026)
```

**Generated Answer**: Fukushima Prefecture

**Ground Truth Answers**: Japan, Okuma

### Example 10
**Question**: what countries are part of the uk

**Topic Entity**: United Kingdom

**Top Retrieved Paths**:
```
  1. location.country.administrative_divisions (score: 0.0140)
  2. location.location.partially_contains -> location.country.administrative_divisions (score: 0.0116)
  3. location.country.form_of_government -> government.form_of_government.countries (score: 0.0082)
  4. location.administrative_division.country (score: 0.0068)
  5. location.country.first_level_divisions -> base.aareas.schema.administrative_area.administrative_children -> organization.organization.founders -> location.country.first_level_divisions -> geography.river.basin_countries -> base.aareas.schema.administrative_area.administrative_children (score: 0.0065)
```

**Generated Answer**: Wales, Scotland, England, Northern Ireland

**Ground Truth Answers**: Wales, Northern Ireland, Scotland, England
