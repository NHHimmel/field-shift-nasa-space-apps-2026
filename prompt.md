## Problem
Farmers face increasingly complex decisions as temperatures shift, rainfall becomes less predictable, and soil health declines across many regions. Water shortages, erosion, nutrient depletion, and extreme weather events add further uncertainty. Selecting new crops to add to a rotation can be difficult, because a crop that improves soil health in one place may consume too much water, introduce disease, conflict with local markets, or fall outside expected growing seasons somewhere else.

This challenge is inspired in part by multigenerational farming experience — many families have lived through seasons where better tools for tracking changing conditions could have helped identify healthier, more resilient rotation choices. Today, NASA’s long‑term environmental datasets offer new ways to understand local growing conditions, and SI techniques open opportunities to translate that data into practical, farmer‑friendly insights.

## Prompt
so first of all the code that exists right now is done by us and
our model/code will generate an output in JSON format containing numerical values. The backend will then make an LLM call and ask it to explain those values based on the JSON, following the structure you define for the website. We'll also instruct the LLM to ensure that the numerical values remain exactly as they are and are not altered in any way.

your task is to see the code understand it and make a website that will help farmers to understand what crop to plant in the future. 

Plan it how you would do it first then ask me questions if necessary.

This website will be build with fastapi and react. I don't know if we need a database or not, I don't even know if we should keep e login page as most of the farmers may find it troublesome. I am not sure, you can suggest me.

The websites ui will be clean and simple. It would be in both bangla and english.

Before starting out talk to me about what you want to do and what stuff do i need to install first.