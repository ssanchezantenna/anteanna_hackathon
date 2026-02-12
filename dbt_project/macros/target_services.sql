{#-
  Macro: target_services

  Returns the 8 target streaming services used throughout the project.
  Two modes:
    - target_services()        -> SQL IN-list string for use in WHERE clauses
    - target_services_list()   -> Jinja list for iteration in templates
-#}

{% macro target_services_list() %}
  {%- set services = [
    'Amazon Prime Video',
    'Apple TV+',
    'Disney+',
    'Max',
    'Netflix',
    'Hulu',
    'Peacock',
    'Paramount+'
  ] -%}
  {{ return(services) }}
{% endmacro %}


{% macro target_services() %}
  (
    'Amazon Prime Video',
    'Apple TV+',
    'Disney+',
    'Max',
    'Netflix',
    'Hulu',
    'Peacock',
    'Paramount+'
  )
{% endmacro %}
