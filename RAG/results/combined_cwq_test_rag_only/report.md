# Combined KGQA System Evaluation Report

**Generated at**: 2025-12-20 08:52:59

## Summary Metrics

### Path Retrieval
| K | Recall | Hits |
|---|--------|------|
| 1 | 0.0806 | 0.1067 |
| 3 | 0.1434 | 0.2033 |
| 5 | 0.2001 | 0.3033 |
| 10 | 0.2616 | 0.3967 |
| 20 | 0.2616 | 0.3967 |
| 50 | 0.2616 | 0.3967 |
| 100 | 0.2616 | 0.3967 |

**MRR**: 0.1845

### Source Contribution
- Retrieved paths in top-k: 100.0%
- Generated paths in top-k: 0.0%

### Answer Quality
- Exact Match: 0.2433
- F1 Score: 0.4278
- Entity Overlap: 0.0000

### Timing
- Retrieval: 17.79 ms
- Generation: 0.00 ms
- Reranking: 61.57 ms
- LLM: 275.52 ms
- **Total**: 354.88 ms

## Example Results

### Example 1
**Question**: Lou Seal is the mascot for the team that last won the World Series when?

**Topic Entity**: ['Lou Seal']

**Top Retrieved Paths**:
```
  1. sports.mascot.team -> sports.sports_team.team_mascot (score: 0.7607)
  2. sports.sports_championship_event.champion -> sports.sports_team.team_mascot (score: 0.7251)
  3. sports.sports_facility.teams -> sports.sports_team.team_mascot (score: 0.6772)
  4. sports.mascot.team -> sports.sports_team.sport (score: 0.6152)
  5. sports.mascot.team (score: 0.5659)
```

**Generated Answer**: 2010

**Ground Truth Answers**: ['2014 World Series']

### Example 2
**Question**: What is the predominant religion where the leader is Ovadia Yosef?

**Topic Entity**: ['Ovadia Yosef']

**Top Retrieved Paths**:
```
  1. religion.religious_organization.leaders -> religion.religious_organization_leadership.jurisdiction (score: 0.4609)
  2. religion.religious_leader.religious_leadership -> religion.religious_organization_leadership.jurisdiction (score: 0.2200)
  3. religion.type_of_place_of_worship.religion -> religion.religion.notable_figures (score: 0.0857)
  4. religion.religious_practice.practice_of -> base.popstra.religion.membership -> base.popstra.religion_choice.member -> government.political_appointer.appointees -> government.government_position_held.office_holder (score: 0.0583)
  5. religion.religion.notable_figures -> organization.organization_founder.organizations_founded -> organization.organization.founders -> government.us_president.vice_president (score: 0.0421)
```

**Generated Answer**: Orthodox Judaism

**Ground Truth Answers**: ['Judaism']

### Example 3
**Question**: What country bordering France contains an airport that serves Nijmegen?

**Topic Entity**: ['Nijmegen', 'Country', 'France']

**Top Retrieved Paths**:
```
  1. location.country.capital -> location.location.nearby_airports (score: 0.4648)
  2. location.location.nearby_airports -> location.location.containedby -> location.country.form_of_government (score: 0.1592)
  3. aviation.airline.airports_served -> aviation.airline_airport_presence.cities_served (score: 0.1471)
  4. location.location.nearby_airports (score: 0.1321)
  5. aviation.airline.airports_served -> aviation.airline_airport_presence.airport (score: 0.0676)
```

**Generated Answer**: Netherlands

**Ground Truth Answers**: ['Germany']

### Example 4
**Question**: Which nation has the Alta Verapaz Department and is in Central America?

**Topic Entity**: ['Central America', 'Alta Verapaz Department', 'Country']

**Top Retrieved Paths**:
```
  1. location.administrative_division.country -> location.country.capital (score: 0.1688)
  2. location.country.capital -> location.administrative_division.country (score: 0.1023)
  3. location.country.capital (score: 0.0592)
  4. government.governmental_body.jurisdiction -> location.administrative_division.country (score: 0.0213)
  5. location.country.administrative_divisions (score: 0.0164)
```

**Generated Answer**: Guatemala

**Ground Truth Answers**: ['Guatemala']

### Example 5
**Question**: Which man is the leader of the country that uses Libya, Libya, Libya as its national anthem?

**Topic Entity**: ['Libya, Libya, Libya', 'Prime Minister of Libya']

**Top Retrieved Paths**:
```
  1. government.national_anthem.national_anthem_of -> government.national_anthem_of_a_country.country -> government.governmental_jurisdiction.governing_officials -> government.government_position_held.office_holder (score: 0.7505)
  2. music.composition.lyricist -> people.person.nationality -> government.governmental_jurisdiction.governing_officials -> government.government_position_held.office_holder (score: 0.0050)
  3. location.location.containedby -> base.locations.countries.cities_within -> visual_art.art_subject.art_series_on_the_subject -> visual_art.art_series.artist -> influence.influence_node.influenced_by (score: 0.0032)
  4. location.country.capital (score: 0.0028)
  5. religion.religion.notable_figures -> visual_art.art_subject.artwork_on_the_subject -> visual_art.artwork.art_subject -> government.us_president.vice_president (score: 0.0024)
```

**Generated Answer**: Prime Minister of Libya

**Ground Truth Answers**: ['Abdullah al-Thani']

### Example 6
**Question**: What educational institution has a football sports team named Northern Colorado Bears is in Greeley, Colorado?

**Topic Entity**: ['College/University', 'Northern Colorado Bears football', 'Greeley']

**Top Retrieved Paths**:
```
  1. education.educational_institution.sports_teams (score: 0.9067)
  2. sports.school_sports_team.school -> location.location.containedby (score: 0.9014)
  3. sports.mascot.team -> sports.sports_team.location (score: 0.8311)
  4. education.school_mascot.school -> location.location.containedby (score: 0.6621)
  5. sports.school_sports_team.school -> education.educational_institution.mascot (score: 0.6372)
```

**Generated Answer**: University of Northern Colorado

**Ground Truth Answers**: ['University of Northern Colorado']

### Example 7
**Question**: What language is spoken in the location that appointed Michelle Bachelet to a governmental position speak?

**Topic Entity**: ['Michelle Bachelet']

**Top Retrieved Paths**:
```
  1. government.government_office_or_title.jurisdiction -> location.country.languages_spoken (score: 0.4417)
  2. government.governmental_body.body_this_is_a_component_of -> government.governmental_body.members -> government.government_position_held.appointed_by (score: 0.3960)
  3. location.country.form_of_government -> government.form_of_government.countries -> location.country.languages_spoken (score: 0.2925)
  4. government.government.government_for -> location.country.languages_spoken (score: 0.2690)
  5. government.governmental_body.jurisdiction -> location.country.languages_spoken (score: 0.2351)
```

**Generated Answer**: Spanish

**Ground Truth Answers**: ['Aymara language', 'Mapudungun Language', 'Rapa Nui Language', 'Spanish Language', 'Puquina Language']

### Example 8
**Question**: What type of government is used in the country with Northern District?

**Topic Entity**: ['Northern District']

**Top Retrieved Paths**:
```
  1. government.government.government_for -> location.country.form_of_government (score: 0.3191)
  2. location.country.form_of_government (score: 0.2766)
  3. law.court.jurisdiction -> location.country.form_of_government (score: 0.2766)
  4. government.governmental_body.jurisdiction -> location.country.form_of_government (score: 0.2751)
  5. language.human_language.main_country -> location.country.form_of_government (score: 0.2720)
```

**Generated Answer**: Federal Republic

**Ground Truth Answers**: ['Parliamentary system']

### Example 9
**Question**: In which countries do the people speak Portuguese, where the child labor percentage was once 1.8?

**Topic Entity**: ['Portuguese Language']

**Top Retrieved Paths**:
```
  1. language.human_language.countries_spoken_in -> base.aareas.schema.administrative_area.administrative_children (score: 0.5137)
  2. language.human_language.main_country -> location.country.languages_spoken (score: 0.1919)
  3. language.human_language.countries_spoken_in -> location.country.first_level_divisions (score: 0.1160)
  4. location.country.languages_spoken -> language.human_language.main_country (score: 0.1067)
  5. location.country.languages_spoken -> language.human_language.language_family (score: 0.0833)
```

**Generated Answer**: Brazil, Angola, Mozambique

**Ground Truth Answers**: ['Mozambique']

### Example 10
**Question**: The people from the country that contains Nord-Ouest Department speak what languages today?

**Topic Entity**: ['Nord-Ouest Department']

**Top Retrieved Paths**:
```
  1. language.human_language.countries_spoken_in -> location.country.languages_spoken (score: 0.1766)
  2. language.human_language.countries_spoken_in (score: 0.1002)
  3. location.country.official_language -> language.human_language.region (score: 0.0637)
  4. language.human_language.countries_spoken_in -> location.country.official_language (score: 0.0637)
  5. people.ethnicity.languages_spoken -> language.human_language.main_country (score: 0.0628)
```

**Generated Answer**: French, Creole

**Ground Truth Answers**: ['Haitian Creole', 'French']
